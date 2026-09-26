import { StorageBufferAttribute, StorageTexture } from 'three/webgpu';
import { ComputeKernel } from '../ComputeKernel.js';
import { uniform, storage, textureStore, globalId } from 'three/tsl';
import { proxy, proxyFn, wgslTagFn } from 'three-mesh-bvh/webgpu';
import { misHeuristicFn, weightedAlphaBlendFn } from '../../nodes/sampling.wgsl.js';
import { clampPathContributionFunc, isTerminatingScatterFunc } from '../../nodes/utils.wgsl.js';
import { TRANSMISSIVE_BACKGROUND_ENVIRONMENT, TRANSMISSIVE_BACKGROUND_OVERLAY, TRANSMISSIVE_BACKGROUND_TRANSPARENT } from '../../constants.js';
import {
	rngInit, rand2, rand3,
	RNG_INDEX_BACKGROUND_SAMPLE,
	RNG_INDEX_DIRECT_LIGHT_SAMPLE,
} from '../../nodes/random.wgsl.js';
import { ENVIRONMENT_LIGHT_TYPE, LIGHT_FAR_DISTANCE, isMISWeightLightFn, neeSlotProbabilityFn } from '../../nodes/lights.wgsl.js';
import { lightRecordStruct, scatterRecordStruct } from '../../nodes/structs.wgsl.js';
import { rayDataStruct, intersectionResultStruct } from './structs.js';
import { SAMPLE_COUNT_MASK, SAMPLE_DISPATCHED_FLAG } from '../../constants.js';

// Path logic over the persistent slot pool: resolves the previous frame's shadow and bounce trace
// results, accumulates emission and NEE / forward MIS contributions, terminates finished paths into
// the output, and samples the next NEE light for MaterialKernel. Touches no BVH data.
export class LogicKernel extends ComputeKernel {

	constructor( ) {

		const params = {
			envInfo: { value: null },
			backgroundInfo: { value: null },
			lightsInfo: { value: null },

			// targets
			prevOutputTarget: textureStore( new StorageTexture( 1, 1 ) ).toReadOnly(),
			outputTarget: textureStore( new StorageTexture( 1, 1 ) ).toWriteOnly(),
			sampleCountTarget: textureStore( new StorageTexture( 1, 1 ) ).toReadWrite(),

			// settings
			misEnabled: uniform( 1, 'uint' ),
			rayCount: uniform( 0, 'uint' ),
			maxBounces: uniform( 5, 'uint' ),
			clampDirect: uniform( 0 ),
			clampIndirect: uniform( 10 ),
			transmissiveBackground: uniform( TRANSMISSIVE_BACKGROUND_OVERLAY ),

			rayDataStorage: storage( new StorageBufferAttribute( 1, 1 ), rayDataStruct ),
			rayIntersectionsStorage: storage( new StorageBufferAttribute( 1, 1 ), intersectionResultStruct ),
			shadowRayIntersectionsStorage: storage( new StorageBufferAttribute( 1, 1 ), intersectionResultStruct ),

			globalId: globalId,
		};

		// environment + background resources pulled off their providers (embedded functions)
		const envMeanRadianceNode = proxy( 'envInfo.value.meanRadianceNode', params );
		const envIntensityNode = proxy( 'envInfo.value.intensityNode', params );
		const sampleEnvColor = proxy( 'envInfo.value.sampleColor', params );
		const sampleEnvDir = proxy( 'envInfo.value.sampleDir', params );
		const getEnvDirPdf = proxy( 'envInfo.value.getDirPdf', params );
		const sampleBackground = proxy( 'backgroundInfo.value.sampleColor', params );

		// analytic scene lights pulled off the lightsInfo provider (LightsInfoNode)
		const lightsCountNode = proxy( 'lightsInfo.value.countNode', params );
		const randomLightSampleFn = proxyFn( 'lightsInfo.value.randomLightSample', params );
		const intersectLightAtIndexFn = proxyFn( 'lightsInfo.value.intersectLightAtIndex', params );
		const emitterCountNode = proxy( 'lightsInfo.value.emitterCountNode', params );
		const sampleEmitterFn = proxyFn( 'lightsInfo.value.sampleEmitter', params );
		const lightSlotWeightFn = proxyFn( 'lightsInfo.value.lightSlotWeight', params );
		const lightSlotActiveFn = proxyFn( 'lightsInfo.value.lightSlotActive', params );
		const emitterSlotWeightFn = proxyFn( 'lightsInfo.value.emitterSlotWeight', params );
		const neeTotalsFn = proxyFn( 'lightsInfo.value.neeTotals', params );

		const fn = wgslTagFn/* wgsl */`

			fn compute(
				// settings
				misEnabled: u32,
				rayCount: u32,
				maxBounces: u32,
				clampDirect: f32,
				clampIndirect: f32,
				transmissiveBackground: u32,

				globalId: vec3u
			) -> void {

				let rayDataStorage = &${ params.rayDataStorage };
				let rayIntersectionsStorage = &${ params.rayIntersectionsStorage };
				let shadowRayIntersectionsStorage = &${ params.shadowRayIntersectionsStorage };

				// bound by "rayCount" rather than the pool length. See MaterialKernel
				let index = globalId.x;
				if ( index >= rayCount ) {

					return;

				}

				// A whole element is read from the BUFFER, never through the pointer alias
				// above: WebKit packs every struct that holds a vec3 and does not unpack a
				// load made through a let pointer, so Safari refused this kernel with
				// "no viable conversion from __typeN_Packed". Field reads and writes
				// through the alias compile, and stay as they are.

				// skip slots that have never spawned a ray
				let input = ${ params.rayDataStorage }[ index ];
				if ( input.rayIntersectionIndex < 0 ) {

					return;

				}

				let indexUV = vec2u( input.pixelIndex >> 16, input.pixelIndex & 0xFFFF );
				${ rngInit }( indexUV, input.seed, input.currentBounce + input.alphaDepth );

				// the environment as a NEE slot: its weight is the irradiance a uniform world of its mean
				// radiance would give, pi times it, and its STRENGTH counts - a world at zero strength has
				// energy in its map and none in the scene, and took half the samples for nothing
				let envWeight = 3.14159265 * ${ envMeanRadianceNode } * ${ envIntensityNode };
				let envActive = envWeight > 0.0;
				let lightsCount = ${ lightsCountNode };
				let emitterCount = ${ emitterCountNode };

				var resultColor = input.resultColor;
				var throughputColor = input.throughputColor;

				// resolve the previous surface's NEE shadow ray (pre-scatter throughput). The index
				// is negative when no shadow ray was enqueued last frame
				if ( input.shadowRayIntersectionIndex >= 0 && input.lightPdf > 0.0 ) {

					let shadowHit = ${ params.shadowRayIntersectionsStorage }[ u32( input.shadowRayIntersectionIndex ) ];
					let occluded = shadowHit.objectIndex >= 0;
					if ( ! occluded ) {

						// env + area lights are also bsdf-sampled, so MIS-weight them; punctual take full weight
						let misWeight = select( 1.0, ${ misHeuristicFn }( input.lightPdf, input.lightBsdfPdf ), ${ isMISWeightLightFn }( input.lightType ) );
						let directLight = throughputColor * input.lightEmission * input.lightBsdf * misWeight / input.lightPdf;
						let contribution = ${ clampPathContributionFunc }( directLight, input.currentBounce, clampDirect, clampIndirect );
						resultColor += vec4f( contribution, 0.0 );

					}

				}

				// ── THE CLAMP DEPTH, as in Cycles (film/light_passes.h) ──
				//
				// A depth of one or less is DIRECT and is not clamped by default. Direct light is
				// clamped at the depth of the surface it lights; background, lights and emission
				// found by a ray at the depth of the path minus one, so light reaching the first
				// surface is direct whichever way it arrives. The escape, the light hits and the
				// gathered emission were all clamped one bounce deeper: under a bright source the
				// indirect limit cut them, and a glossy reflection of a bright area light came
				// out at 28% of its radiance.

				// emission gathered at the previous surface (pre-scatter throughput): that surface was
				// reached one bounce earlier than the one being lit now
				let emission = ${ clampPathContributionFunc }( throughputColor * input.emission, max( input.currentBounce, 1u ) - 1u, clampDirect, clampIndirect );
				resultColor += vec4f( emission, 0.0 );

				// reconstruct the scatter record staged by MaterialKernel
				var scatterRec: ${ scatterRecordStruct };
				scatterRec.color = input.scatterColor;
				scatterRec.pdf = input.scatterPdf;

				// the bounce limit, russian roulette, and terminating scatter checks all ran in
				// MaterialKernel, which stages a zeroed pdf and skips the bounce trace when they fire
				var isTerminated = all( throughputColor == vec3f( 0.0 ) ) || input.currentBounce >= maxBounces || ${ isTerminatingScatterFunc }( scatterRec );

				// the lights the traced segment passed through, added once at the end: the escape of a
				// camera segment ASSIGNS the pixel from the background, and would erase them
				var lightHits = vec3f( 0.0 );

				if ( ! isTerminated ) {

					// apply the scatter across the traced segment
					throughputColor *= scatterRec.color / scatterRec.pdf;

					let hitResult = ${ params.rayIntersectionsStorage }[ u32( input.rayIntersectionIndex ) ];
					let didHit = hitResult.objectIndex >= 0;
					let surfaceDist = select( ${ LIGHT_FAR_DISTANCE }, hitResult.dist, didHit );

					// the NEE choice as it was made at the vertex this segment left: what a light found by
					// the segment is weighed against
					let originTotals = ${ neeTotalsFn }( input.origin, envWeight );

					// forward hits: a segment that lands on an area light, on the sphere of a sized point
					// or spot, or - when it escapes - on the disc of a sun with an angle, which sits at
					// LIGHT_FAR_DISTANCE and so is never nearer than a surface.
					//
					// The CAMERA segment sees them too, as in Cycles: a camera ray skips a light only when
					// its object leaves camera visibility off (lights_intersect), and objects are born with
					// it on. There it takes full weight, since no NEE came before it. Past it the hit is
					// MIS-weighted, only when NEE is also sampling the lights. A light never stops the ray -
					// Cycles counts it as a transparent bounce - so it adds to what lies behind it
					for ( var li = 0u; li < lightsCount; li ++ ) {

						var lightRec: ${ lightRecordStruct };
						if ( ${ intersectLightAtIndexFn }( input.origin, input.direction, li, &lightRec ) && ( ! didHit || lightRec.dist < surfaceDist ) ) {

							var misWeight = 1.0;
							if ( misEnabled != 0u && input.currentBounce > 0u ) {

								let choice = ${ neeSlotProbabilityFn }( ${ lightSlotWeightFn }( li, input.origin ), ${ lightSlotActiveFn }( li ), originTotals );
								misWeight = ${ misHeuristicFn }( input.scatterPdf, lightRec.pdf * choice );

							}

							lightHits += ${ clampPathContributionFunc }( lightRec.emission * throughputColor * misWeight, input.currentBounce, clampDirect, clampIndirect );

						}

					}

					if ( didHit ) {

						// stage the hit for MaterialKernel
						rayDataStorage[ index ].barycoord = hitResult.barycoord;
						rayDataStorage[ index ].normal = hitResult.normal;
						rayDataStorage[ index ].side = hitResult.side;
						rayDataStorage[ index ].indices = hitResult.indices;
						// senza questa riga il colpo su un pelo arriva a chi ombreggia
						// travestito da triangolo, e "indices" diventa tre indici di vertice
						// che nessuno ha scritto
						rayDataStorage[ index ].isCurve = hitResult.isCurve;
						rayDataStorage[ index ].objectIndex = hitResult.objectIndex;
						rayDataStorage[ index ].dist = hitResult.dist;

						// the emitter slot's probability at the vertex this segment left, for MaterialKernel,
						// which weighs the emission of this surface and has no room for the lights buffer
						rayDataStorage[ index ].emitterSelectPdf = ${ neeSlotProbabilityFn }( ${ emitterSlotWeightFn }( input.origin ), emitterCount > 0u, originTotals );

						// next event estimation: pick one slot by importance (neeSlotProbability) with a single
						// sample. MaterialKernel evaluates the bsdf and enqueues the shadow ray.
						var lightPdf = 0.0;
						let totals = ${ neeTotalsFn }( hitResult.position, envWeight );
						if ( misEnabled != 0u && totals.y > 0.0 ) {

							let ruv = ${ rand3 }( ${ RNG_INDEX_DIRECT_LIGHT_SAMPLE } );

							// walk the slots in a fixed order - lights, environment, table - to the one whose
							// share of [0, 1) holds the sample. The last active slot catches what rounding
							// leaves past the end
							// "found" is its own flag: the environment and the table are the negative slots -2
							// and -3, so the sign of "chosen" cannot also say whether anything was picked
							var found = false;
							var chosen = - 1;
							var chosenProbability = 0.0;
							var remainder = 0.0;
							var cumulative = 0.0;
							var lastActive = - 1;
							var lastProbability = 0.0;
							for ( var li = 0u; li < lightsCount; li ++ ) {

								let probability = ${ neeSlotProbabilityFn }( ${ lightSlotWeightFn }( li, hitResult.position ), ${ lightSlotActiveFn }( li ), totals );
								if ( ! found && probability > 0.0 && ruv.x < cumulative + probability ) {

									found = true;
									chosen = i32( li );
									chosenProbability = probability;

								}
								if ( probability > 0.0 ) {

									lastActive = i32( li );
									lastProbability = probability;

								}
								cumulative += probability;

							}

							let envProbability = ${ neeSlotProbabilityFn }( envWeight, envActive, totals );
							if ( ! found && envProbability > 0.0 && ruv.x < cumulative + envProbability ) {

								found = true;
								chosen = - 2;
								chosenProbability = envProbability;

							}
							if ( envProbability > 0.0 ) {

								lastActive = - 2;
								lastProbability = envProbability;

							}
							cumulative += envProbability;

							let emitterProbability = ${ neeSlotProbabilityFn }( ${ emitterSlotWeightFn }( hitResult.position ), emitterCount > 0u, totals );
							if ( ! found && emitterProbability > 0.0 ) {

								found = true;
								chosen = - 3;
								chosenProbability = emitterProbability;
								// the fraction of the pick left over picks the triangle
								remainder = clamp( ( ruv.x - cumulative ) / emitterProbability, 0.0, 1.0 );

							}
							if ( ! found ) {

								chosen = lastActive;
								chosenProbability = lastProbability;

							}

							var lightRec: ${ lightRecordStruct };
							if ( chosen == - 2 ) {

								// the environment, sampled from its CDF, as a light of kind ENVIRONMENT
								let envSample = ${ sampleEnvDir }( ruv.yz );
								lightRec.direction = envSample.direction;
								// the radiance of the escape branch, not a copy of it
								lightRec.emission = ${ sampleEnvColor }( envSample.direction ).rgb;
								lightRec.pdf = envSample.pdf;
								lightRec.dist = ${ LIGHT_FAR_DISTANCE };
								lightRec.lightType = ${ ENVIRONMENT_LIGHT_TYPE };

							} else if ( chosen == - 3 ) {

								lightRec = ${ sampleEmitterFn }( hitResult.position, remainder, ruv.yz );

							} else {

								lightRec = ${ randomLightSampleFn }( u32( chosen ), hitResult.position, ruv.yz );

							}

							lightPdf = lightRec.pdf * chosenProbability;
							rayDataStorage[ index ].lightDirection = lightRec.direction;
							rayDataStorage[ index ].lightEmission = lightRec.emission;
							rayDataStorage[ index ].lightDist = lightRec.dist;
							rayDataStorage[ index ].lightType = lightRec.lightType;

						}

						rayDataStorage[ index ].lightPdf = lightPdf;

					} else {

						// the segment escaped the scene: gather the environment for opaque paths, or
						// the background for camera segments and fully transmissive paths
						if ( input.currentBounce > 0u && input.isFullyTransmissive == 0u ) {

							var misWeight = 1.0;
							if ( misEnabled != 0u && envActive ) {

								// the probability NEE gave the environment at the vertex the segment left, so the
								// two estimators balance
								let envPdf = ${ getEnvDirPdf }( input.direction ) * ${ neeSlotProbabilityFn }( envWeight, envActive, originTotals );
								misWeight = ${ misHeuristicFn }( input.scatterPdf, envPdf );

							}

							let environment = ${ sampleEnvColor }( input.direction ).rgb * throughputColor * misWeight;
							let contribution = ${ clampPathContributionFunc }( environment, input.currentBounce, clampDirect, clampIndirect );
							resultColor += vec4f( contribution, 0.0 );

						} else {

							// hit the background
							// support multiple transparent background blending techniques
							let rng = ${ rand2 }( ${ RNG_INDEX_BACKGROUND_SAMPLE } );
							let bg = ${ sampleBackground }( input.direction, rng );
							if ( input.currentBounce == 0u ) {

								// sample the background directly if this is the primary ray
								let background = ${ clampPathContributionFunc }( bg.a * bg.rgb, input.currentBounce, clampDirect, clampIndirect );
								resultColor = vec4f( background, bg.a );

							} else {

								// transmissive ray handling
								let env = ${ sampleEnvColor }( input.direction );
								let avg = saturate( dot( throughputColor, vec3f( 1.0 / 3.0 ) ) );
								let transparency = ( 1.0 - bg.a ) * avg;

								var misWeight = 1.0;
								if ( misEnabled != 0u && envActive ) {

									// with the probability of its slot, as in the opaque branch: without it the two
									// weights did not add up to one when other lights shared the choice
									let envPdf = ${ getEnvDirPdf }( input.direction ) * ${ neeSlotProbabilityFn }( envWeight, envActive, originTotals );
									misWeight = ${ misHeuristicFn }( input.scatterPdf, envPdf );

								}

								if ( transmissiveBackground == ${ TRANSMISSIVE_BACKGROUND_ENVIRONMENT }u ) {

									// display the env map through transmissive surfaces
									let background = ${ clampPathContributionFunc }( env.rgb * throughputColor * misWeight, input.currentBounce, clampDirect, clampIndirect );
									resultColor = vec4f(
										resultColor.rgb + background,
										1.0,
									);

								} else if ( transmissiveBackground == ${ TRANSMISSIVE_BACKGROUND_TRANSPARENT }u ) {

									// fade the background by the throughput color average
									let background = ${ clampPathContributionFunc }( bg.a * bg.rgb * throughputColor * misWeight, input.currentBounce, clampDirect, clampIndirect );
									resultColor = vec4f(
										resultColor.rgb + background,
										1.0 - transparency,
									);

								} else {

									// fade the background by the throughput color average, mixing in env lighting
									var light = mix( env.rgb, bg.rgb, bg.a ) * misWeight;
									let background = ${ clampPathContributionFunc }( light * throughputColor, input.currentBounce, clampDirect, clampIndirect );
									resultColor = vec4f(
										resultColor.rgb + background,
										1.0 - transparency,
									);

								}

							}

						}

						isTerminated = true;

					}

				}

				// the lights leave the alpha as it is, as in Cycles: they are emission, and the pixel is as
				// transparent as the background behind them
				resultColor += vec4f( lightHits, 0.0 );

				// the color rows are stored top down to match a rasterized render target
				let colorIndex = vec2u( indexUV.x, textureDimensions( ${ params.outputTarget } ).y - 1u - indexUV.y );
				let storedSamples = textureLoad( ${ params.sampleCountTarget }, indexUV ).r & ${ SAMPLE_COUNT_MASK }u;

				if ( isTerminated ) {

					// blend the finished sample into the output and free the slot for a new camera ray
					let sampleCount = storedSamples + 1;
					let prevColor = textureLoad( ${ params.prevOutputTarget }, colorIndex );
					let blendedColor = ${ weightedAlphaBlendFn }( prevColor, resultColor, 1.0 / f32( sampleCount ) );
					textureStore( ${ params.sampleCountTarget }, indexUV, vec4( ${ SAMPLE_DISPATCHED_FLAG }u | sampleCount ) );
					textureStore( ${ params.outputTarget }, colorIndex, blendedColor );

					rayDataStorage[ index ].objectIndex = - 1;

				} else {

					// show the path so far, which the first sample overwrites
					if ( storedSamples == 0u ) {

						textureStore( ${ params.outputTarget }, colorIndex, resultColor );

					}

					rayDataStorage[ index ].resultColor = resultColor;
					rayDataStorage[ index ].throughputColor = throughputColor;

				}

			}
		`;

		super( fn( params ) );

		this.defineUniformAccessors( params );

	}

}
