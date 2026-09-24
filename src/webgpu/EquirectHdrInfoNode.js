import { DataTexture, FloatType, NearestFilter, RedFormat, Vector3 } from 'three';
import { Matrix3 } from 'three/webgpu';
import { texture, sampler, uniform } from 'three/tsl';
import { EquirectHdrInfoUniform } from '../uniforms/EquirectHdrInfoUniform.js';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';
import { constants, environmentSampleStruct } from './nodes/structs.wgsl.js';
import { equirectDirectionToUvFn, equirectUvToDirectionFn, sampleUniformConeFunc } from './nodes/sampling.wgsl.js';

// ── THE SUN WEIGHT, the one of Cycles ──
//
// With a sun set, a sample picks the sun cone with this probability and the map
// otherwise, and the pdf of either is the MIXTURE of the two. It is sun_weight = 4
// against map_weight = 1 in the background light of Cycles ("empirical value").
export const SUN_GUIDING_WEIGHT = 4 / 5;

/** A single channel float texture read with textureLoad: no filtering, no sampler. */
function cdfTexture( data, width, height ) {

	const tex = new DataTexture( data, width, height, RedFormat, FloatType );
	tex.minFilter = NearestFilter;
	tex.magFilter = NearestFilter;
	tex.generateMipmaps = false;
	tex.needsUpdate = true;
	return tex;

}

export class EquirectHdrInfoNode extends EquirectHdrInfoUniform {

	constructor() {

		super();

		// environment map + importance-sampling CDF textures, each with a sampler
		this.mapNode = texture( this.map );
		this.mapSampler = sampler( this.mapNode );

		// ── THE CDFs, searched instead of the inverse tables ──
		//
		// The inverse tables return the CENTER of a texel, so a texel that holds a
		// lot of energy - a sun - turns into a point light. These are the inclusive
		// CDFs themselves, in 32 bits: a half float near one keeps three digits,
		// and a row of a thousand texels needs four.
		this.marginalCdf = cdfTexture( new Float32Array( [ 1 ] ), 1, 1 );
		this.conditionalCdf = cdfTexture( new Float32Array( [ 1 ] ), 1, 1 );
		this.marginalCdfNode = texture( this.marginalCdf );
		this.conditionalCdfNode = texture( this.conditionalCdf );

		// scalar parameters that assemble into the EnvironmentInfo struct in the shader
		this.rotationNode = uniform( new Matrix3() );
		this.intensityNode = uniform( 1 );
		this.totalSumNode = uniform( this.totalSum );

		// ── SUN GUIDING: a cone of directions sampled on its own ──
		//
		// A small bright disc in the environment - the sun of a sky texture - is
		// sampled as the cone it fills, as background_sun_sample does in Cycles. The
		// direction is in MAP space, the space of equirectDirectionToUv; the weight
		// is zero when there is no sun, and then the map is the only strategy.
		this.sunDirectionNode = uniform( new Vector3( 0, 1, 0 ) );
		this.sunOneMinusCosNode = uniform( 0 );
		this.sunWeightNode = uniform( 0 );

		this._initFns();

	}

	getPixelWeight( r, g, b, row, height ) {

		// weight the pixel contribution by its spherical solid angle.
		const theta = Math.PI * ( row + 0.5 ) / height;
		return super.getPixelWeight( r, g, b ) * Math.sin( theta );

	}

	/**
	 * Guides samples toward a sun: a cone around a direction in MAP space, of the
	 * given half angle in radians. Null turns it off. The radiance of the disc is
	 * not this object's business: whoever draws it wraps sampleColor, and the
	 * kernels read both strategies through that one function.
	 */
	setSunGuiding( direction, halfAngle = 0, weight = SUN_GUIDING_WEIGHT ) {

		if ( ! direction || ! ( halfAngle > 0 ) ) {

			this.sunWeightNode.value = 0;
			return;

		}

		this.sunDirectionNode.value.copy( direction ).normalize();
		// one minus the cosine, written so that it keeps its digits at a quarter of a degree
		this.sunOneMinusCosNode.value = 2 * Math.sin( halfAngle / 2 ) ** 2;
		this.sunWeightNode.value = weight;

	}

	dispose() {

		super.dispose();
		this.marginalCdf.dispose();
		this.conditionalCdf.dispose();

	}

	updateFrom( envMap ) {

		super.updateFrom( envMap );

		const {
			mapNode,
			totalSumNode,
			marginalCdf,
			conditionalCdf,
		} = this;

		const { width, height } = this.map.image;
		marginalCdf.image = { width: height, height: 1, data: this.cdfMarginal };
		marginalCdf.needsUpdate = true;
		conditionalCdf.image = { width, height, data: this.cdfConditional };
		conditionalCdf.needsUpdate = true;

		// refresh values in place on the existing nodes so no rebuild is required
		mapNode.value = this.map;
		totalSumNode.value = this.totalSum;

	}

	_initFns() {

		const {
			mapNode,
			mapSampler,
			marginalCdfNode,
			conditionalCdfNode,
			totalSumNode,
			rotationNode,
			intensityNode,
			sunDirectionNode,
			sunOneMinusCosNode,
			sunWeightNode,
		} = this;

		this.sampleColor = wgslTagFn/* wgsl */`
			fn sampleEnv( direction: vec3f ) -> vec4f {

				let sampleDir = ${ rotationNode } * direction;
				let mapUv = ${ equirectDirectionToUvFn }( sampleDir );
				let col = textureSampleLevel( ${ mapNode }, ${ mapSampler }, mapUv, 0 );

				return vec4f( ${ intensityNode } * col.rgb, col.a );

			}
		`;

		// A point of the map drawn EXACTLY from the piecewise constant distribution:
		// the texel from a binary search of the CDF, and the position inside it from
		// where the random number falls between the two CDF values - the inverse
		// lerp of background_map_sample in Cycles. The inverse tables this replaces
		// gave the texel center, always.
		//
		// ── AND THE SAMPLE CARRIES THE PROBABILITY OF ITS OWN TEXEL (z) ──
		//
		// Reading it back from the direction is not the same thing: uv -> direction ->
		// uv in single precision trigonometry lands in the NEIGHBOUR texel for 0.76%
		// of the samples on this GPU (31 in 4096, measured against a JS twin). Next
		// to a sun the neighbour is sky, a thousand times less likely, and the
		// sample weighs a thousand times more: a floor at 77% saturated pixels.
		const envMapSampleUv = wgslTagFn/* wgsl */`
			fn envMapSampleUv( r: vec2f ) -> vec3f {

				let size = textureDimensions( ${ conditionalCdfNode } );

				// the row, from the marginal CDF: the first entry above the random number
				var lo = 0u;
				var hi = size.y - 1u;
				loop {

					if ( lo >= hi ) {

						break;

					}
					let mid = ( lo + hi ) / 2u;
					if ( textureLoad( ${ marginalCdfNode }, vec2u( mid, 0u ), 0 ).x > r.x ) {

						hi = mid;

					} else {

						lo = mid + 1u;

					}

				}
				let y = lo;
				let yPrev = max( y, 1u ) - 1u;
				let cdfYPrev = select( 0.0, textureLoad( ${ marginalCdfNode }, vec2u( yPrev, 0u ), 0 ).x, y > 0u );
				let cdfY = textureLoad( ${ marginalCdfNode }, vec2u( y, 0u ), 0 ).x;
				let dv = clamp( ( r.x - cdfYPrev ) / max( cdfY - cdfYPrev, 1e-20 ), 0.0, 0.99999 );

				// the column, from the conditional CDF of that row
				lo = 0u;
				hi = size.x - 1u;
				loop {

					if ( lo >= hi ) {

						break;

					}
					let mid = ( lo + hi ) / 2u;
					if ( textureLoad( ${ conditionalCdfNode }, vec2u( mid, y ), 0 ).x > r.y ) {

						hi = mid;

					} else {

						lo = mid + 1u;

					}

				}
				let x = lo;
				let xPrev = max( x, 1u ) - 1u;
				let cdfXPrev = select( 0.0, textureLoad( ${ conditionalCdfNode }, vec2u( xPrev, y ), 0 ).x, x > 0u );
				let cdfX = textureLoad( ${ conditionalCdfNode }, vec2u( x, y ), 0 ).x;
				let du = clamp( ( r.y - cdfXPrev ) / max( cdfX - cdfXPrev, 1e-20 ), 0.0, 0.99999 );

				let texelProbability = max( ( cdfY - cdfYPrev ) * ( cdfX - cdfXPrev ), 0.0 );
				return vec3f( ( f32( x ) + du ) / f32( size.x ), ( f32( y ) + dv ) / f32( size.y ), texelProbability );

			}
		`;

		// The density of the map strategy, from the SAME CDFs it samples: the texel
		// probability over the solid angle of that point. Using the luminance of the
		// filtered map here would give a pdf that is not the one the samples follow.
		const envMapPdf = wgslTagFn/* wgsl */`
			${ [ constants ] }
			fn envMapPdf( mapDir: vec3f ) -> f32 {

				let size = textureDimensions( ${ conditionalCdfNode } );
				let uv = ${ equirectDirectionToUvFn }( mapDir );
				let x = min( u32( uv.x * f32( size.x ) ), size.x - 1u );
				let y = min( u32( uv.y * f32( size.y ) ), size.y - 1u );
				let xPrev = max( x, 1u ) - 1u;
				let yPrev = max( y, 1u ) - 1u;
				let pY = textureLoad( ${ marginalCdfNode }, vec2u( y, 0u ), 0 ).x
					- select( 0.0, textureLoad( ${ marginalCdfNode }, vec2u( yPrev, 0u ), 0 ).x, y > 0u );
				let pX = textureLoad( ${ conditionalCdfNode }, vec2u( x, y ), 0 ).x
					- select( 0.0, textureLoad( ${ conditionalCdfNode }, vec2u( xPrev, y ), 0 ).x, x > 0u );
				let sinTheta = sin( uv.y * PI );
				let pdf = max( pY * pX, 0.0 ) * f32( size.x * size.y ) / ( 2.0 * PI * PI * max( sinTheta, 1e-8 ) );
				return select( 0.0, pdf, sinTheta > 0.0 );

			}
		`;

		// The density of the MIXTURE, in map space: what one sample of sampleEnvDir
		// follows whichever strategy drew it, and what an escaping ray weighs against.
		const envMixturePdf = wgslTagFn/* wgsl */`
			${ [ constants, envMapPdf ] }
			fn envMixturePdf( direction: vec3f ) -> f32 {

				let mapDir = normalize( direction );

				let w = ${ sunWeightNode };
				let oneMinusCos = max( ${ sunOneMinusCosNode }, 1e-12 );
				let inCone = w > 0.0 && dot( mapDir, ${ sunDirectionNode } ) >= 1.0 - oneMinusCos;
				let sunPdf = select( 0.0, 1.0 / ( 2.0 * PI * oneMinusCos ), inCone );
				return w * sunPdf + ( 1.0 - w ) * ${ envMapPdf }( mapDir );

			}
		`;

		this.sampleDir = wgslTagFn/* wgsl */`
			${ [ constants, envMapSampleUv, envMapPdf, envMixturePdf ] }
			fn sampleEnvDir( r: vec2f ) -> ${ environmentSampleStruct } {

				var result: ${ environmentSampleStruct };

				// ── SUN OR MAP: one random number picks, and is rescaled to be used again ──
				let w = ${ sunWeightNode };
				var rr = r;
				var mapDir: vec3f;
				var pdf: f32;
				if ( rr.x < w ) {

					rr.x = rr.x / w;
					mapDir = ${ sampleUniformConeFunc }( ${ sunDirectionNode }, ${ sunOneMinusCosNode }, rr );

					// the sample came FROM the cone, so its density has the cone term by
					// construction. Asking the cone test instead fails at the rim: a dot
					// product against 1 - 1e-5 with rounding of 1e-7, and a sample that loses
					// the term weighs ~700 times too much - the streaks of fireflies measured
					let coneOneMinusCos = max( ${ sunOneMinusCosNode }, 1e-12 );
					pdf = w / ( 2.0 * PI * coneOneMinusCos ) + ( 1.0 - w ) * ${ envMapPdf }( mapDir );

				} else {

					rr.x = ( rr.x - w ) / max( 1.0 - w, 1e-6 );
					let drawn = ${ envMapSampleUv }( rr );
					// normalized: the cone test is a dot product against one minus a cosine
					// of 1e-5, and in single precision trigonometry this direction comes out
					// that far off unit length - the disc said inside, the pdf said outside
					mapDir = normalize( ${ equirectUvToDirectionFn }( drawn.xy ) );

					// the map density of the texel the sample came FROM, not of the
					// texel its direction reads back as
					let size = textureDimensions( ${ conditionalCdfNode } );
					let sinTheta = sin( drawn.y * PI );
					let mapPdf = drawn.z * f32( size.x * size.y ) / ( 2.0 * PI * PI * max( sinTheta, 1e-8 ) );
					let oneMinusCos = max( ${ sunOneMinusCosNode }, 1e-12 );
					let inCone = w > 0.0 && dot( mapDir, ${ sunDirectionNode } ) >= 1.0 - oneMinusCos;
					pdf = w * select( 0.0, 1.0 / ( 2.0 * PI * oneMinusCos ), inCone ) + ( 1.0 - w ) * select( 0.0, mapPdf, sinTheta > 0.0 );

				}

				result.direction = transpose( ${ rotationNode } ) * mapDir;
				result.pdf = select( 0.0, pdf, ${ totalSumNode } != 0.0 );

				return result;

			}
		`;

		this.getDirPdf = wgslTagFn/* wgsl */`
			${ [ envMixturePdf ] }
			fn getEnvDirPdf( direction: vec3f ) -> f32 {

				// normalized: the cone test is a dot product against one minus a cosine of
				// 1e-5, and a direction a hair off unit length would fall outside it
				let mapDir = normalize( ${ rotationNode } * direction );
				return select( 0.0, ${ envMixturePdf }( mapDir ), ${ totalSumNode } != 0.0 );

			}
		`;

	}

}
