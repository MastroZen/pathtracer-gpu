import { wgslFn } from 'three/tsl';
import { constants, lightStruct, lightRecordStruct } from './structs.wgsl.js';
import { sampleUniformConeFunc, sinSqrToOneMinusCosFunc } from './sampling.wgsl.js';

// Light type tags matching LightsInfoUniformStruct's packing. The environment is treated as an
// additional light kind so env + analytic lights share one NEE path.
export const RECT_AREA_LIGHT_TYPE = 0;
export const CIRC_AREA_LIGHT_TYPE = 1;
export const SPOT_LIGHT_TYPE = 2;
export const DIR_LIGHT_TYPE = 3;
export const POINT_LIGHT_TYPE = 4;
export const ENVIRONMENT_LIGHT_TYPE = 5;
export const LIGHT_FAR_DISTANCE = 1e30;

// tolerance for comparing a shadow hit distance to the sampled light distance
export const LIGHT_EPSILON = 1e-5;

// light kinds that are also bsdf-sampled and so take MIS-weighted NEE - punctual lights take full weight
export const isMISWeightLightFn = wgslFn( /* wgsl */ `

	fn isMISWeightLight( lightType: i32 ) -> bool {

		return lightType == ${ ENVIRONMENT_LIGHT_TYPE } || lightType == ${ CIRC_AREA_LIGHT_TYPE } || lightType == ${ RECT_AREA_LIGHT_TYPE };

	}

` );

export const getSpotAttenuationFn = wgslFn( /* wgsl */ `

	fn getSpotAttenuation( coneCosine: f32, penumbraCosine: f32, angleCosine: f32 ) -> f32 {

		return smoothstep( coneCosine, penumbraCosine, angleCosine );

	}

` );

export const getDistanceAttenuationFn = wgslFn( /* wgsl */ `

	fn getDistanceAttenuation( lightDistance: f32, cutoffDistance: f32, decayExponent: f32 ) -> f32 {

		// based upon Frostbite 3 Moving to Physically-based Rendering
		// https://seblagarde.files.wordpress.com/2015/07/course_notes_moving_frostbite_to_pbr_v32.pdf
		var distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), EPSILON );
		if ( cutoffDistance > 0.0 ) {

			let window = clamp( 1.0 - pow( lightDistance / cutoffDistance, 4.0 ), 0.0, 1.0 );
			distanceFalloff *= window * window;

		}

		return distanceFalloff;

	}

`, [ constants ] );

// Ray/plane intersection constrained to a rectangle centered at "center" spanned by u, v.
// Returns the hit distance along the ray, or a negative value when there is no hit.
export const intersectsRectangleFn = wgslFn( /* wgsl */ `

	fn intersectsRectangle( center: vec3f, normal: vec3f, u: vec3f, v: vec3f, rayOrigin: vec3f, rayDirection: vec3f ) -> f32 {

		let t = dot( center - rayOrigin, normal ) / dot( rayDirection, normal );
		if ( t > EPSILON ) {

			let p = rayOrigin + rayDirection * t;
			let vi = p - center;

			let a1 = dot( u, vi );
			if ( abs( a1 ) <= 0.5 ) {

				let a2 = dot( v, vi );
				if ( abs( a2 ) <= 0.5 ) {

					return t;

				}

			}

		}

		return - 1.0;

	}

`, [ constants ] );

// Ray/plane intersection constrained to a circle centered at "position" spanned by u, v.
// Returns the hit distance along the ray, or a negative value when there is no hit.
export const intersectsCircleFn = wgslFn( /* wgsl */ `

	fn intersectsCircle( position: vec3f, normal: vec3f, u: vec3f, v: vec3f, rayOrigin: vec3f, rayDirection: vec3f ) -> f32 {

		let t = dot( position - rayOrigin, normal ) / dot( rayDirection, normal );
		if ( t > EPSILON ) {

			let hit = rayOrigin + rayDirection * t;
			let vi = hit - position;

			let a1 = dot( u, vi );
			let a2 = dot( v, vi );
			if ( length( vec2f( a1, a2 ) ) <= 0.5 ) {

				return t;

			}

		}

		return - 1.0;

	}

`, [ constants ] );

// Samples a random point on a rectangular or circular area light and forms the LightRecord.
export const randomAreaLightSampleFn = wgslFn( /* wgsl */ `

	fn randomAreaLightSample( light: Light, rayOrigin: vec3f, ruv: vec2f ) -> LightRecord {

		var randomPos = vec3f( 0.0 );
		if ( light.lightType == ${ RECT_AREA_LIGHT_TYPE } ) {

			randomPos = light.position + light.u * ( ruv.x - 0.5 ) + light.v * ( ruv.y - 0.5 );

		} else if ( light.lightType == ${ CIRC_AREA_LIGHT_TYPE } ) {

			let r = 0.5 * sqrt( ruv.x );
			let theta = ruv.y * 2.0 * PI;
			randomPos = light.position + light.u * ( r * cos( theta ) ) + light.v * ( r * sin( theta ) );

		}

		let toLight = randomPos - rayOrigin;
		let lightDistSq = dot( toLight, toLight );
		let dist = sqrt( lightDistSq );
		let direction = toLight / dist;
		let lightNormal = normalize( cross( light.u, light.v ) );

		var lightRec: LightRecord;
		lightRec.lightType = light.lightType;
		lightRec.emission = light.color * light.intensity;
		lightRec.dist = dist;
		lightRec.direction = direction;

		// points behind or edge-on to the light receive no contribution
		let cosTheta = dot( direction, lightNormal );
		if ( cosTheta <= 0.0 ) {

			lightRec.pdf = 0.0;

		} else {

			lightRec.pdf = lightDistSq / ( light.area * cosTheta );

		}

		return lightRec;

	}

`, [ lightStruct, lightRecordStruct, constants ] );

// ── SIZED LIGHTS: the SPHERE of Cycles ──
//
// A point or spot light with a radius is a sphere of uniform radiance, and a shading point sees
// a cap of it: the direction is drawn uniformly inside the cone that cap fills, which is the
// estimator of point_light_sample and spot_light_sample in Cycles 4.0 and later. It never runs
// with a zero radius - the callers keep the delta light, so a scene without sizes renders
// exactly as before.
//
// The radiance of the sphere is the intensity over pi r^2, and it fills a solid angle of
// 2 pi ( 1 - cos ): together that is the intensity over d^2 times 2 / ( 1 + cos ). The factor is
// one far from the light and two touching it, and it rides on top of the distance attenuation
// so that decay and cutoff distance keep their meaning.
//
// Inside the sphere there is no cap to sample: the light stays the point at its center. Cycles
// samples the hemisphere there.
//
// No early return, on purpose: an early return inside a function the tracing loop inlines has
// already cost a lost device on the D3D11 backend.
export const randomSphereLightSampleFn = wgslFn( /* wgsl */ `

	fn randomSphereLightSample( light: Light, center: vec3f, rayOrigin: vec3f, ruv: vec2f ) -> LightRecord {

		let toCenter = center - rayOrigin;
		let distSq = dot( toCenter, toCenter );
		let dist = sqrt( distSq );
		let axis = toCenter / max( dist, EPSILON );
		let radiusSq = light.radius * light.radius;
		let distanceAttenuation = getDistanceAttenuation( dist, light.distance, light.decay );

		var lightRec: LightRecord;
		lightRec.lightType = light.lightType;
		lightRec.pdf = 1.0;
		lightRec.dist = dist;
		lightRec.direction = axis;
		lightRec.emission = light.color * light.intensity * distanceAttenuation;

		if ( distSq > radiusSq ) {

			let oneMinusCos = sinSqrToOneMinusCos( radiusSq / distSq );
			let direction = sampleUniformCone( axis, oneMinusCos, ruv );

			// law of cosines: the near side of the sphere along the sampled direction
			let cosTheta = dot( direction, axis );
			lightRec.dist = dist * cosTheta - sqrt( max( radiusSq - distSq + distSq * cosTheta * cosTheta, 0.0 ) );
			lightRec.direction = direction;
			lightRec.emission *= 2.0 / ( 2.0 - oneMinusCos );

		}

		return lightRec;

	}

`, [ lightStruct, lightRecordStruct, constants, getDistanceAttenuationFn, sinSqrToOneMinusCosFunc, sampleUniformConeFunc ] );

// Samples a spot light with distance falloff. Angular ( cone or IES ) attenuation is applied by
// the caller, on the sampled direction as Cycles does: with a radius the edge of the cone blurs
// too, not only the shadow.
//
// With a radius the spot is the sphere of the point light. It was a disc here, pushed forward
// along the axis by radius / tan( angle ) so that it filled the cone: a narrow spot of 10 cm put
// its emitter a metre in front of the lamp, inside whatever was there.
export const randomSpotLightSampleFn = wgslFn( /* wgsl */ `

	fn randomSpotLightSample( light: Light, rayOrigin: vec3f, ruv: vec2f ) -> LightRecord {

		var lightRec: LightRecord;
		if ( light.radius > 0.0 ) {

			lightRec = randomSphereLightSample( light, light.position, rayOrigin, ruv );

		} else {

			let toLight = light.position - rayOrigin;
			let lightDistSq = dot( toLight, toLight );
			let dist = sqrt( lightDistSq );

			let direction = toLight / max( dist, EPSILON );
			let distanceAttenuation = getDistanceAttenuation( dist, light.distance, light.decay );

			lightRec.lightType = light.lightType;
			lightRec.dist = dist;
			lightRec.direction = direction;
			lightRec.emission = light.color * light.intensity * distanceAttenuation;
			lightRec.pdf = 1.0;

		}

		return lightRec;

	}

`, [ lightStruct, lightRecordStruct, constants, getDistanceAttenuationFn, randomSphereLightSampleFn ] );
