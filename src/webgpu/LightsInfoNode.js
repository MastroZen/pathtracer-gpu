import { storage, uniform, uniformArray, texture } from 'three/tsl';
import { StorageBufferAttribute, HalfFloatType, Vector4 } from 'three/webgpu';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';
import { AtlasTexture } from './AtlasTexture.js';
import { LightsInfoUniformStruct } from '../uniforms/LightsInfoUniformStruct.js';
import { lightStruct, lightRecordStruct } from './nodes/structs.wgsl.js';
import { sampleTexelFunc } from './nodes/utils.wgsl.js';
import {
	RECT_AREA_LIGHT_TYPE,
	CIRC_AREA_LIGHT_TYPE,
	SPOT_LIGHT_TYPE,
	DIR_LIGHT_TYPE,
	POINT_LIGHT_TYPE,
	SUN_DISC_LIGHT_TYPE,
	EMITTER_LIGHT_TYPE,
	LIGHT_FAR_DISTANCE,
	intersectsRectangleFn,
	intersectsCircleFn,
	randomAreaLightSampleFn,
	randomSpotLightSampleFn,
	randomSphereLightSampleFn,
	sphereLightHitFn,
	sunLightHitFn,
	sunLightOneMinusCosFn,
	sunLightRadianceFn,
	getSpotAttenuationFn,
	getDistanceAttenuationFn,
} from './nodes/lights.wgsl.js';
import { sampleUniformConeFunc } from './nodes/sampling.wgsl.js';
import { EMITTER_STRIDE } from './emitters.js';

export class LightsInfoNode extends LightsInfoUniformStruct {

	constructor() {

		super();

		// lights packed into a storage buffer of Light structs
		this.countNode = uniform( this.count, 'uint' );
		this.buffer = new StorageBufferAttribute( new Float32Array( 2 * lightStruct.getLength() ), lightStruct.getLength() );
		this.bufferNode = storage( this.buffer, lightStruct ).toReadOnly().setName( 'lights' );

		// ies profiles packed into an atlas alongside their placement rects
		this.iesAtlas = new AtlasTexture( { type: HalfFloatType } );
		this.iesProfilesNode = texture( this.iesAtlas.texture );
		// named for the same reason as bvh_textureInfo: an id in the name is a new shader
		this.iesInfoNode = uniformArray( this.iesAtlas.textureInfo, 'uvec4' ).setName( 'iesInfo' );

		// the EMITTER TABLE: emissive triangles as lights (emitters.js), four vec4 each and drawn
		// by power. Its count is a uniform of its own, zero when no mesh emits
		this.emitterCountNode = uniform( 0, 'uint' );
		// its power (area times luminance, summed) and its bounding sphere: how NEE weighs the whole
		// table against the other lights when it picks one
		this.emitterPowerNode = uniform( 0 );
		this.emitterBoundsNode = uniform( new Vector4() );
		this.emitterBuffer = new StorageBufferAttribute( new Float32Array( 2 * EMITTER_STRIDE ), 4 );
		this.emitterBufferNode = storage( this.emitterBuffer, 'vec4' ).toReadOnly().setName( 'emitters' );

		this._initFns();

	}

	updateFrom( renderer, lights ) {

		// the unique set of ies textures referenced by the lights' "iesMap" fields
		const iesTextures = Array.from( new Set( lights.map( l => l.iesMap ).filter( t => t ) ) );
		const changed = super.updateFrom( lights, iesTextures );

		this.iesAtlas.setTextures( renderer, iesTextures );
		this.iesProfilesNode.value = this.iesAtlas.texture;

		const stride = lightStruct.getLength();
		const count = this.count;
		const capacity = Math.max( count, 2 );

		// resize the buffer to the exact light count, keeping the same binding node
		if ( this.buffer.array.length !== capacity * stride ) {

			this.buffer = new StorageBufferAttribute( new Float32Array( capacity * stride ), stride );
			this.bufferNode.value = this.buffer;

		}

		// the texture's packed float layout already matches lightStruct's std layout, so copy it in
		const src = this.tex.image.data;
		this.buffer.array.set( src.subarray( 0, count * stride ) );

		// rewrite the int fields ( lightType, iesProfile ) as i32 bits
		const intView = new Int32Array( this.buffer.array.buffer );
		for ( let i = 0; i < count; i ++ ) {

			const base = i * stride;
			intView[ base + 3 ] = Math.round( src[ base + 3 ] );
			intView[ base + 21 ] = Math.round( src[ base + 21 ] );

		}

		this.buffer.needsUpdate = true;
		this.countNode.value = this.count;

		return changed;

	}

	// the table built by collectEmitters, uploaded as is
	updateEmitters( table ) {

		const { data, count } = table;
		if ( this.emitterBuffer.array.length < data.length ) {

			this.emitterBuffer = new StorageBufferAttribute( new Float32Array( data.length ), 4 );
			this.emitterBufferNode.value = this.emitterBuffer;

		}

		this.emitterBuffer.array.set( data );
		this.emitterBuffer.needsUpdate = true;
		this.emitterCountNode.value = count;
		this.emitterPowerNode.value = table.totalPower;
		this.emitterBoundsNode.value.fromArray( table.bounds );

	}

	_initFns() {

		const { bufferNode, iesProfilesNode, iesInfoNode, countNode, emitterCountNode, emitterBufferNode, emitterPowerNode, emitterBoundsNode } = this;

		// profiles are sampled out of an atlas so filtering must resolve tile-relative wrapping
		const sampleIesTexelFn = sampleTexelFunc( iesInfoNode, iesProfilesNode, 'sampleIesTexel' );

		// the angular attenuation of a spot along a direction toward the light, cone or IES. One
		// function for the sample and for the hit of a sized spot: MIS weighs the two, and they must
		// see the same lamp
		const spotAttenuationFn = wgslTagFn/* wgsl */`
			fn spotAttenuation( light: ${ lightStruct }, direction: vec3f ) -> f32 {

				let spotNormal = normalize( cross( light.u, light.v ) );
				let cosTheta = dot( direction, spotNormal );
				var attenuation: f32;
				if ( light.iesProfile >= 0 ) {

					// tilt angle off the forward axis and twist angle around it, with the
					// tilt axis clamped and the twist axis wrapped ( wrapS = 1, wrapT = 0 )
					let tiltAngle = acos( cosTheta ) / PI;
					let twistAngle = ( atan2( dot( direction, light.v ), dot( direction, light.u ) ) + PI ) / ( 2.0 * PI );
					let packedProfile = ( 1 << 26 ) | light.iesProfile;
					attenuation = ${ sampleIesTexelFn }( vec2f( tiltAngle, twistAngle ), packedProfile, 0.0 ).r;

				} else {

					attenuation = ${ getSpotAttenuationFn }( light.coneCos, light.penumbraCos, cosTheta );

				}

				return attenuation;

			}
		`;

		// uniformly pick a light and sample it
		this.randomLightSample = wgslTagFn/* wgsl */`
			fn randomLightSample( lightIndex: u32, rayOrigin: vec3f, ruv: vec2f ) -> ${ lightRecordStruct } {

				let light = ${ bufferNode }[ lightIndex ];

				var result: ${ lightRecordStruct };
				if ( light.lightType == ${ SPOT_LIGHT_TYPE } ) {

					result = ${ randomSpotLightSampleFn }( light, rayOrigin, ruv );
					result.emission *= ${ spotAttenuationFn }( light, result.direction );

				} else if ( light.lightType == ${ POINT_LIGHT_TYPE } && light.radius > 0.0 ) {

					// ── SIZED POINT LIGHT ── a sphere around the position packed in the u slot
					result = ${ randomSphereLightSampleFn }( light, light.u, rayOrigin, ruv );

				} else if ( light.lightType == ${ POINT_LIGHT_TYPE } ) {

					// the point light's world position is packed into the u slot
					let lightRay = light.u - rayOrigin;
					let lightDist = length( lightRay );
					let cutoffDistance = light.distance;
					var distanceFalloff = 1.0 / max( pow( lightDist, light.decay ), 0.01 );
					if ( cutoffDistance > 0.0 ) {

						let window = clamp( 1.0 - pow( lightDist / cutoffDistance, 4.0 ), 0.0, 1.0 );
						distanceFalloff *= window * window;

					}

					result.direction = normalize( lightRay );
					result.dist = lightDist;
					result.pdf = 1.0;
					result.emission = light.color * light.intensity * distanceFalloff;
					result.lightType = light.lightType;

				} else if ( light.lightType == ${ DIR_LIGHT_TYPE } ) {

					// the directional light's direction is packed into the u slot
					result.dist = ${ LIGHT_FAR_DISTANCE };
					result.direction = light.u;
					result.pdf = 1.0;
					result.emission = light.color * light.intensity;
					result.lightType = light.lightType;
					// ── SUN CONE ── the sun has an angular diameter, and the radius slot holds the
					// tangent of half of it: the disc it would be at unit distance, as Cycles kept it
					// once. Zero keeps the delta light. With a size the record carries the radiance of
					// the disc and the pdf of its cone, which a bsdf-sampled ray finds too (sunLightHit),
					// so the sun is seen in a mirror and weighed with MIS in a glossy reflection.
					if ( light.radius > 0.0 ) {

						let oneMinusCos = ${ sunLightOneMinusCosFn }( light );
						result.direction = ${ sampleUniformConeFunc }( light.u, oneMinusCos, ruv );
						result.pdf = 1.0 / ( 2.0 * PI * oneMinusCos );
						result.emission = ${ sunLightRadianceFn }( light );
						result.lightType = ${ SUN_DISC_LIGHT_TYPE };

					}

				} else {

					result = ${ randomAreaLightSampleFn }( light, rayOrigin, ruv );

				}

				return result;

			}
		`;

		// -- THE WEIGHTS OF THE NEE CHOICE -- an estimate of the irradiance each slot brings to a point
		// (see neeSlotProbability): a lamp its intensity over d^2 (never nearer than its radius or a
		// centimetre), a spot the same through its cone, the sun its strength, an area light its
		// radiance times its area over d^2 on the side it faces, the emitter table its power over the
		// squared distance to its bounding sphere. The environment is weighed by the kernel, which owns
		// its uniforms. The same functions serve the pick and the hit: MIS needs both to agree
		this.lightSlotWeight = wgslTagFn/* wgsl */`
			fn lightSlotWeight( index: u32, x: vec3f ) -> f32 {

				let light = ${ bufferNode }[ index ];
				let power = dot( light.color, vec3f( 0.2126, 0.7152, 0.0722 ) ) * light.intensity;
				var weight = power;
				if ( light.lightType == ${ POINT_LIGHT_TYPE } || light.lightType == ${ SPOT_LIGHT_TYPE } ) {

					let center = select( light.position, light.u, light.lightType == ${ POINT_LIGHT_TYPE } );
					let toLight = center - x;
					let dist = sqrt( dot( toLight, toLight ) );
					weight = power * ${ getDistanceAttenuationFn }( max( dist, max( light.radius, 0.01 ) ), light.distance, light.decay );
					if ( light.lightType == ${ SPOT_LIGHT_TYPE } ) {

						weight *= ${ spotAttenuationFn }( light, toLight / max( dist, 1e-20 ) );

					}

				} else if ( light.lightType == ${ RECT_AREA_LIGHT_TYPE } || light.lightType == ${ CIRC_AREA_LIGHT_TYPE } ) {

					let normal = normalize( cross( light.u, light.v ) );
					let toLight = light.position - x;
					let distSq = dot( toLight, toLight );
					let cosLight = max( dot( toLight / sqrt( max( distSq, 1e-20 ) ), normal ), 0.0 );
					weight = power * light.area * cosLight / max( distSq, light.area );

				}

				return max( weight, 0.0 );

			}
		`;

		// a light that emits nothing is not a slot: sampling it would only spend samples
		this.lightSlotActive = wgslTagFn/* wgsl */`
			fn lightSlotActive( index: u32 ) -> bool {

				let light = ${ bufferNode }[ index ];
				return dot( light.color, vec3f( 0.2126, 0.7152, 0.0722 ) ) * light.intensity > 0.0;

			}
		`;

		this.emitterSlotWeight = wgslTagFn/* wgsl */`
			fn emitterSlotWeight( x: vec3f ) -> f32 {

				let bounds = ${ emitterBoundsNode };
				let toCenter = bounds.xyz - x;
				return ${ emitterPowerNode } / max( dot( toCenter, toCenter ), max( bounds.w * bounds.w, 1e-8 ) );

			}
		`;

		// the sum of the weights and the number of active slots at a point
		this.neeTotals = wgslTagFn/* wgsl */`
			fn neeTotals( x: vec3f, envWeight: f32 ) -> vec2f {

				var total = 0.0;
				var activeCount = 0.0;
				for ( var i = 0u; i < ${ countNode }; i ++ ) {

					total += ${ this.lightSlotWeight }( i, x );
					activeCount += select( 0.0, 1.0, ${ this.lightSlotActive }( i ) );

				}
				if ( envWeight > 0.0 ) {

					total += envWeight;
					activeCount += 1.0;

				}
				if ( ${ emitterCountNode } > 0u ) {

					total += ${ this.emitterSlotWeight }( x );
					activeCount += 1.0;

				}

				return vec2f( total, activeCount );

			}
		`;

		// -- NEE ON THE EMITTER TABLE -- a triangle drawn by power, then a point uniform on it. The
		// pdf is the density per unit area of that triangle's material (luminance over the power of
		// the table) turned into solid angle, the number MaterialKernel rebuilds when a bsdf ray hits
		// the same point. A zero pdf is a point seen from the side the raycast refuses, or edge-on
		this.sampleEmitter = wgslTagFn/* wgsl */`
			fn sampleEmitter( rayOrigin: vec3f, u: f32, ruv: vec2f ) -> ${ lightRecordStruct } {

				// the first triangle whose cumulative weight passes u
				var lo = 0u;
				var hi = ${ emitterCountNode } - 1u;
				loop {

					if ( lo >= hi ) {

						break;

					}
					let mid = ( lo + hi ) / 2u;
					if ( ${ emitterBufferNode }[ mid * 4u ].w > u ) {

						hi = mid;

					} else {

						lo = mid + 1u;

					}

				}

				let a = ${ emitterBufferNode }[ lo * 4u ];
				let b = ${ emitterBufferNode }[ lo * 4u + 1u ];
				let c = ${ emitterBufferNode }[ lo * 4u + 2u ];
				let radiance = ${ emitterBufferNode }[ lo * 4u + 3u ];

				// uniform on the triangle: the square root keeps the density flat
				let su = sqrt( ruv.x );
				let p = a.xyz * ( 1.0 - su ) + b.xyz * ( su * ( 1.0 - ruv.y ) ) + c.xyz * ( su * ruv.y );
				let toPoint = p - rayOrigin;
				let distSq = dot( toPoint, toPoint );
				let dist = sqrt( distSq );
				let direction = toPoint / max( dist, 1e-20 );
				let normal = normalize( cross( b.xyz - a.xyz, c.xyz - a.xyz ) );

				// the cosine at the emitter on the side the raycast accepts: c.w is 1 front, -1 back,
				// 0 both, already turned around for a mirroring transform
				let cosLight = dot( normal, - direction );
				let facing = select( cosLight * c.w, abs( cosLight ), c.w == 0.0 );

				var lightRec: ${ lightRecordStruct };
				lightRec.lightType = ${ EMITTER_LIGHT_TYPE };
				lightRec.direction = direction;
				// the shadow ray stops short of the emitter, or its own triangle would shadow it
				lightRec.dist = dist * ( 1.0 - 1e-4 );
				lightRec.emission = radiance.xyz;
				lightRec.pdf = 0.0;
				if ( facing > 1e-6 && distSq > 1e-12 ) {

					lightRec.pdf = b.w * distSq / facing;

				}

				return lightRec;

			}
		`;

		// forward intersection of a ray with a single light, used for MIS: the area lights, the sphere
		// of a point or spot light with a radius, and the disc of a sun with an angle. Delta lights
		// cannot be hit, and their samples take full weight. A sun disc is at LIGHT_FAR_DISTANCE:
		// the caller takes it only when the ray escapes the scene
		this.intersectLightAtIndex = wgslTagFn/* wgsl */`
			fn intersectLightAtIndex( rayOrigin: vec3f, rayDirection: vec3f, index: u32, lightRec: ptr<function, ${ lightRecordStruct }> ) -> bool {

				let light = ${ bufferNode }[ index ];
				var hit = false;

				if ( light.lightType == ${ RECT_AREA_LIGHT_TYPE } || light.lightType == ${ CIRC_AREA_LIGHT_TYPE } ) {

					var u = light.u;
					var v = light.v;
					let normal = normalize( cross( u, v ) );

					// only front-facing area lights can be hit
					if ( dot( normal, rayDirection ) > 0.0 ) {

						u *= 1.0 / dot( u, u );
						v *= 1.0 / dot( v, v );

						var dist = - 1.0;
						if ( light.lightType == ${ RECT_AREA_LIGHT_TYPE } ) {

							dist = ${ intersectsRectangleFn }( light.position, normal, u, v, rayOrigin, rayDirection );

						} else {

							dist = ${ intersectsCircleFn }( light.position, normal, u, v, rayOrigin, rayDirection );

						}

						if ( dist > 0.0 ) {

							let cosTheta = dot( rayDirection, normal );
							lightRec.dist = dist;
							lightRec.pdf = ( dist * dist ) / ( light.area * cosTheta );
							lightRec.emission = light.color * light.intensity;
							lightRec.direction = rayDirection;
							lightRec.lightType = light.lightType;
							hit = true;

						}

					}

				} else if ( ( light.lightType == ${ POINT_LIGHT_TYPE } || light.lightType == ${ SPOT_LIGHT_TYPE } ) && light.radius > 0.0 ) {

					// the point light keeps its position in the u slot, the spot in position
					let center = select( light.position, light.u, light.lightType == ${ POINT_LIGHT_TYPE } );
					var sphereRec = ${ sphereLightHitFn }( light, center, rayOrigin, rayDirection );
					if ( sphereRec.pdf > 0.0 ) {

						if ( light.lightType == ${ SPOT_LIGHT_TYPE } ) {

							sphereRec.emission *= ${ spotAttenuationFn }( light, rayDirection );

						}
						*lightRec = sphereRec;
						hit = true;

					}

				} else if ( light.lightType == ${ DIR_LIGHT_TYPE } && light.radius > 0.0 ) {

					let sunRec = ${ sunLightHitFn }( light, rayDirection );
					if ( sunRec.pdf > 0.0 ) {

						*lightRec = sunRec;
						hit = true;

					}

				}

				return hit;

			}
		`;

	}

	dispose() {

		this.tex.dispose();
		this.iesAtlas.dispose();

	}

}
