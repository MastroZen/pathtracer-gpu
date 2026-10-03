import { StorageBufferAttribute, StorageTexture } from 'three/webgpu';
import { ComputeKernel } from '../ComputeKernel.js';
import { uniform, storage, textureStore, globalId } from 'three/tsl';
import { proxy, proxyFn, wgslTagFn } from 'three-mesh-bvh/webgpu';
import { misHeuristicFn, weightedAlphaBlendFn } from '../../nodes/sampling.wgsl.js';
import { clampPathContributionFunc, isTerminatingScatterFunc } from '../../nodes/utils.wgsl.js';
import { TRANSMISSIVE_BACKGROUND_ENVIRONMENT, TRANSMISSIVE_BACKGROUND_OVERLAY, TRANSMISSIVE_BACKGROUND_TRANSPARENT } from '../../constants.js';
import {
	rngInit, rand2, rand3, rand4,
	RNG_INDEX_BACKGROUND_SAMPLE,
	RNG_INDEX_DIRECT_LIGHT_SAMPLE,
	RNG_INDEX_MEDIUM,
	RNG_INDEX_MEDIUM_ANCHOR,
	RNG_INDEX_MEDIUM_DIRECT,
	RNG_INDEX_MEDIUM_LIGHT,
} from '../../nodes/random.wgsl.js';
import { ENVIRONMENT_LIGHT_TYPE, LIGHT_EPSILON, LIGHT_FAR_DISTANCE, isMISWeightLightFn, neeSlotProbabilityFn } from '../../nodes/lights.wgsl.js';
import { lightRecordStruct, scatterRecordStruct } from '../../nodes/structs.wgsl.js';
import { henyeyGreensteinPhaseFunc } from '../../nodes/material.wgsl.js';
import { mediumStackCoefficientsFunc, mediumStackFlagsFunc, mediumStackPhaseFunc, mediumStackPhaseWeightsFunc } from '../../nodes/mediumStack.wgsl.js';
import { rayDataStruct, intersectionResultStruct, rayQueueAtomicStruct } from './structs.js';
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
			// the material records, for the medium a path travels inside: the sixth storage
			// buffer of this kernel, after its three and the two of the lights
			bvhData: { value: null },

			// targets
			prevOutputTarget: textureStore( new StorageTexture( 1, 1 ) ).toReadOnly(),
			outputTarget: textureStore( new StorageTexture( 1, 1 ) ).toWriteOnly(),
			sampleCountTarget: textureStore( new StorageTexture( 1, 1 ) ).toReadWrite(),

			// settings
			misEnabled: uniform( 1, 'uint' ),
			rayCount: uniform( 0, 'uint' ),
			maxBounces: uniform( 5, 'uint' ),
			// the volume bounce budget, as MaterialKernel has it: a segment whose scatter would
			// spend it does not sample one, and its direct light takes the whole weight
			maxVolumeBounces: uniform( 0, 'uint' ),
			clampDirect: uniform( 0 ),
			clampIndirect: uniform( 10 ),
			transmissiveBackground: uniform( TRANSMISSIVE_BACKGROUND_OVERLAY ),

			rayDataStorage: storage( new StorageBufferAttribute( 1, 1 ), rayDataStruct ),
			rayIntersectionsStorage: storage( new StorageBufferAttribute( 1, 1 ), intersectionResultStruct ),
			shadowRayIntersectionsStorage: storage( new StorageBufferAttribute( 1, 1 ), intersectionResultStruct ),
			// the seventh storage buffer: the direct light of a medium segment is queued from here
			shadowRayQueue: storage( new StorageBufferAttribute( 1, 1 ), rayQueueAtomicStruct ),

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
		const materialsBuffer = proxy( 'bvhData.value.storage.materials', params );
		// the stack of media a path is in, read from the records of its entries
		const stackFlagsFn = mediumStackFlagsFunc( materialsBuffer );
		const stackCoefficientsFn = mediumStackCoefficientsFunc( materialsBuffer );
		const stackPhaseWeightsFn = mediumStackPhaseWeightsFunc( materialsBuffer );
		const stackPhaseFn = mediumStackPhaseFunc( materialsBuffer, henyeyGreensteinPhaseFunc );

		// -- THE NEE CHOICE -- one slot by importance (neeSlotProbability) with a single number: walk
		// the slots in a fixed order - lights, environment, table - to the one whose share of [0, 1)
		// holds it, the last active slot catching what rounding leaves past the end. The answer is
		// ( slot, probability, remainder ): the slot is the light index, -2 the environment, -3 the
		// emitter table, -1 nothing that emits; the remainder is the fraction of the number left
		// over, which picks the triangle of the table. ONE choice for the vertex and for a medium
		// segment: two copies of a choice are two choices
		const neeChooseFn = wgslTagFn/* wgsl */`
			fn neeChoose( x: vec3f, u: f32, envWeight: f32, envActive: bool ) -> vec3f {

				let totals = ${ neeTotalsFn }( x, envWeight );
				if ( totals.y <= 0.0 ) {

					return vec3f( - 1.0, 0.0, 0.0 );

				}

				// "found" is its own flag: the environment and the table are the negative slots -2
				// and -3, so the sign of "chosen" cannot also say whether anything was picked
				var found = false;
				var chosen = - 1;
				var chosenProbability = 0.0;
				var remainder = 0.0;
				var cumulative = 0.0;
				var lastActive = - 1;
				var lastProbability = 0.0;
				for ( var li = 0u; li < ${ lightsCountNode }; li ++ ) {

					let probability = ${ neeSlotProbabilityFn }( ${ lightSlotWeightFn }( li, x ), ${ lightSlotActiveFn }( li ), totals );
					if ( ! found && probability > 0.0 && u < cumulative + probability ) {

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
				if ( ! found && envProbability > 0.0 && u < cumulative + envProbability ) {

					found = true;
					chosen = - 2;
					chosenProbability = envProbability;

				}
				if ( envProbability > 0.0 ) {

					lastActive = - 2;
					lastProbability = envProbability;

				}
				cumulative += envProbability;

				let emitterProbability = ${ neeSlotProbabilityFn }( ${ emitterSlotWeightFn }( x ), ${ emitterCountNode } > 0u, totals );
				if ( ! found && emitterProbability > 0.0 ) {

					found = true;
					chosen = - 3;
					chosenProbability = emitterProbability;
					remainder = clamp( ( u - cumulative ) / emitterProbability, 0.0, 1.0 );

				}
				if ( ! found ) {

					chosen = lastActive;
					chosenProbability = lastProbability;

				}
				return vec3f( f32( chosen ), chosenProbability, remainder );

			}
		`;

		// the probability the choice gives a slot at a point: what a MIS weight needs when the slot
		// was chosen somewhere else
		const neeSlotChoiceFn = wgslTagFn/* wgsl */`
			fn neeSlotChoice( slot: i32, x: vec3f, envWeight: f32, envActive: bool ) -> f32 {

				let totals = ${ neeTotalsFn }( x, envWeight );
				if ( slot == - 2 ) {

					return ${ neeSlotProbabilityFn }( envWeight, envActive, totals );

				}
				if ( slot == - 3 ) {

					return ${ neeSlotProbabilityFn }( ${ emitterSlotWeightFn }( x ), ${ emitterCountNode } > 0u, totals );

				}
				return ${ neeSlotProbabilityFn }( ${ lightSlotWeightFn }( u32( slot ), x ), ${ lightSlotActiveFn }( u32( slot ) ), totals );

			}
		`;

		// -- THE NEE SAMPLE -- a direction toward a point of the chosen slot, from "x". The record's
		// pdf is the light's alone, without the choice
		const neeSampleFn = wgslTagFn/* wgsl */`
			fn neeSample( slot: i32, x: vec3f, uv: vec2f, remainder: f32 ) -> ${ lightRecordStruct } {

				var lightRec: ${ lightRecordStruct };
				if ( slot == - 2 ) {

					// the environment, sampled from its CDF, as a light of kind ENVIRONMENT
					let envSample = ${ sampleEnvDir }( uv );
					lightRec.direction = envSample.direction;
					// the radiance of the escape branch, not a copy of it
					lightRec.emission = ${ sampleEnvColor }( envSample.direction ).rgb;
					lightRec.pdf = envSample.pdf;
					lightRec.dist = ${ LIGHT_FAR_DISTANCE };
					lightRec.lightType = ${ ENVIRONMENT_LIGHT_TYPE };

				} else if ( slot == - 3 ) {

					lightRec = ${ sampleEmitterFn }( x, remainder, uv );

				} else {

					lightRec = ${ randomLightSampleFn }( u32( slot ), x, uv );

				}
				return lightRec;

			}
		`;

		const fn = wgslTagFn/* wgsl */`

			fn compute(
				// settings
				misEnabled: u32,
				rayCount: u32,
				maxBounces: u32,
				maxVolumeBounces: u32,
				clampDirect: f32,
				clampIndirect: f32,
				transmissiveBackground: u32,

				globalId: vec3u
			) -> void {

				let rayDataStorage = &${ params.rayDataStorage };
				let rayIntersectionsStorage = &${ params.rayIntersectionsStorage };
				let shadowRayIntersectionsStorage = &${ params.shadowRayIntersectionsStorage };
				let shadowRayQueue = &${ params.shadowRayQueue };

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
						// times what the media it crossed let through, one without any
						let directLight = throughputColor * input.lightEmission * input.lightBsdf * misWeight / input.lightPdf * shadowHit.barycoord;
						let contribution = ${ clampPathContributionFunc }( directLight, input.currentBounce, clampDirect, clampIndirect );
						resultColor += vec4f( contribution, 0.0 );

					}

				}

				// and the direct light of the medium segment traced last frame, queued by this kernel
				// with its whole weight: the shadow brings what the media on its way let through
				if ( input.mediumShadowIndex >= 0 ) {

					let mediumHit = ${ params.shadowRayIntersectionsStorage }[ u32( input.mediumShadowIndex ) ];
					if ( mediumHit.objectIndex < 0 ) {

						resultColor += vec4f( input.mediumDirect * mediumHit.barycoord, 0.0 );

					}

				}
				rayDataStorage[ index ].mediumShadowIndex = - 1;

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

					// -- THE MEDIUM THE SEGMENT CROSSES (Cycles shade_volume.h, volume_integrate_homogeneous) --
					//
					// A path inside a participating medium does not reach the surface for sure: it
					// scatters on the way with an exponential law, and where it does that point is
					// the next vertex. As in Cycles the segment has TWO points: the one the path
					// continues from, and the one its direct light is estimated from - this happens
					// here and not in MaterialKernel, which has no room for the lights. A medium that
					// only absorbs draws nothing and takes its transmittance. The emission of the
					// whole segment is added either way, being an integral that does not depend on
					// the draw. A heterogeneous medium keeps the one shared point (docs/mezzi.md in
					// the app). The media are the path's STACK, summed as Cycles sums the closures of
					// every entry; a subsurface walk integrates its own volume, and the media around
					// it wait for the path to leave it, as there
					let stackInfo = select( vec2u( 0u ), ${ stackFlagsFn }( input.mediumStack ), input.insideMaterial < 0 );
					let mediumFlags = stackInfo.x;
					// a stack with a HETEROGENEOUS medium was walked by VolumeKernel before this
					// kernel: its weight is already in the throughput, and its answer is the scatter
					// flag and distance in the ray data. Lights seen through it take no attenuation -
					// the declared divergence (docs/mezzi.md in the app)
					let heterogeneous = ( mediumFlags & 4u ) != 0u;
					let inMedium = ( mediumFlags & 2u ) != 0u && ! heterogeneous;
					var sigmaT = vec3f( 0.0 );
					var channelP = vec3f( 0.0 );
					var mediumSampled = false;
					var mediumScatter = false;
					var mediumEnd = surfaceDist;
					var mediumWeight = vec3f( 1.0 );
					if ( heterogeneous && input.mediumScatter == 1u ) {

						mediumScatter = true;
						mediumEnd = input.dist;

					}
					if ( inMedium ) {

						let coefficients = ${ stackCoefficientsFn }( input.mediumStack );
						let sigmaS = coefficients[ 0 ];
						sigmaT = coefficients[ 1 ];
						let mediumEmission = coefficients[ 2 ];
						// the weights of the entries' phases, constant along a homogeneous segment
						let phaseWeights = ${ stackPhaseWeightsFn }( input.mediumStack );
						if ( any( mediumEmission > vec3f( 0.0 ) ) ) {

							// the integral of the emission times the transmittance, and its limit L
							// where the medium is clear (volume_emission_integrate)
							let opaque = sigmaT > vec3f( 1e-6 );
							let integral = select( mediumEmission * surfaceDist, mediumEmission * ( 1.0 - exp( - sigmaT * surfaceDist ) ) / max( sigmaT, vec3f( 1e-6 ) ), opaque );
							resultColor += vec4f( ${ clampPathContributionFunc }( throughputColor * integral, max( input.currentBounce, 1u ) - 1u, clampDirect, clampIndirect ), 0.0 );

						}

						if ( all( sigmaS <= vec3f( 0.0 ) ) ) {

							mediumWeight = exp( - sigmaT * surfaceDist );

						} else {

							let carried = max( throughputColor * sigmaS / max( sigmaT, vec3f( 1e-9 ) ), vec3f( 0.0 ) );
							let carriedSum = carried.r + carried.g + carried.b;
							channelP = select( vec3f( 1.0 / 3.0 ), carried / carriedSum, carriedSum > 1e-9 );
							// a scatter on this segment would spend the volume bounce budget: then none is
							// sampled, as Cycles past volume_bounces, and the direct light has no rival
							let budgetSpent = input.volumeBounce + 1u > maxVolumeBounces || input.currentBounce + 1u >= maxBounces;

							// -- THE DIRECT LIGHT OF THE SEGMENT (Cycles shade_volume.h, the direct point) --
							//
							// Estimated on EVERY segment, from a point of its own, whatever the path does
							// next: a thin fog transmits most paths, and a direct light tied to the scatter
							// left all of them dark (measured: no gain at all from equiangular sampling at
							// 0.15 per metre). In the order of Cycles: the LIGHT is chosen first, for the
							// segment, and a point sampled on it is the anchor of the equiangular strategy
							// (light_sample_from_volume_segment); then the point of the segment; then a
							// direction toward that same light from there. An anchor picked apart
							// from the light the point then saw left an emitting wall behind a fog with an
							// angle that pointed elsewhere (measured: 13.1% off, against 5.1% by distance).
							// The point is drawn by distance TRUNCATED to the segment, as Cycles does for a
							// homogeneous medium (volume_integrate_homogeneous, the direct scatter), or by
							// the angle, or half and half. The sun and the sky have no point: distance.
							// MIS on the direction against the phase-sampled hits of the continuation is
							// weighed with the choice as THEY see it, at the point: the weights only have
							// to add up to one there. Its shadow is a second ray of this slot, resolved
							// next frame - so only on a segment that ENDS somewhere: inside a medium a ray
							// always meets its boundary, and one that escapes is an open mesh. Bits 3 and 4
							// of the flags are Volume Sampling set to Distance, or to Equiangular alone
							if ( misEnabled != 0u && didHit ) {

								let segmentLen = surfaceDist;
								let ra = ${ rand4 }( ${ RNG_INDEX_MEDIUM_ANCHOR } );
								// the light first, when the angle needs its anchor: chosen where the segment
								// STARTS. Cycles weighs the whole segment; its middle was worse here, since it
								// falls beside the lamp and its 1 / r^2 crushes an emitting wall behind the fog
								// (measured: 5.6% off against 3.7% from the start, equiangular). By distance
								// there is no anchor, and the light is chosen at the point itself, which sees
								// it (18.9% off chosen at the start, 5.1% at the point)
								let distanceOnly = stackInfo.y == 0u;
								var choice = vec3f( - 1.0, 0.0, 0.0 );
								if ( ! distanceOnly ) {

									choice = ${ neeChooseFn }( input.origin, ra.x, envWeight, envActive );

								}

								var useEquiangular = false;
								var equiDelta = 0.0;
								var equiD = 0.0;
								var equiThetaA = 0.0;
								var equiThetaB = 0.0;
								if ( choice.y > 0.0 && i32( choice.x ) != - 2 ) {

									let anchorRec = ${ neeSampleFn }( i32( choice.x ), input.origin, ra.yz, choice.z );
									if ( anchorRec.pdf > 0.0 && anchorRec.dist < ${ LIGHT_FAR_DISTANCE } * 0.5 ) {

										let anchor = input.origin + anchorRec.direction * anchorRec.dist;
										equiDelta = dot( anchor - input.origin, input.direction );
										equiD = length( anchor - input.origin - input.direction * equiDelta );
										equiThetaA = atan2( - equiDelta, equiD );
										equiThetaB = atan2( segmentLen - equiDelta, equiD );
										useEquiangular = equiD > 1e-6 && equiThetaB - equiThetaA > 1e-6;

									}

								}

								// the truncated distance: a channel, then the inverse of its CDF on [0, L]
								let rd = ${ rand2 }( ${ RNG_INDEX_MEDIUM_DIRECT } );
								let reach = max( vec3f( 1.0 ) - exp( - sigmaT * segmentLen ), vec3f( 1e-9 ) );
								var directChannel = 2u;
								if ( rd.x < channelP.r ) { directChannel = 0u; }
								else if ( rd.x < channelP.r + channelP.g ) { directChannel = 1u; }
								var td = - log( max( 1.0 - rd.y * reach[ directChannel ], 1e-12 ) ) / max( sigmaT[ directChannel ], 1e-9 );
								// Equiangular alone draws every point by the angle, as Cycles does; without an
								// anchor it falls back on the distance
								let angleOnly = stackInfo.y == 1u;
								let byAngle = useEquiangular && ( angleOnly || ra.w < 0.5 );
								if ( byAngle ) {

									td = equiDelta + equiD * tan( mix( equiThetaA, equiThetaB, rd.y ) );

								}
								td = clamp( td, 0.0, segmentLen );
								let directT = exp( - sigmaT * td );
								let distancePdf = dot( channelP, sigmaT * directT / reach );
								let angle = td - equiDelta;
								let anglePdf = select( 0.0, equiD / ( ( equiThetaB - equiThetaA ) * ( equiD * equiD + angle * angle ) ), useEquiangular );
								// one strategy drawn of the two, weighed by the BALANCE heuristic, that is
								// by the density of the mixture: Veach (9.2.4) shows it optimal when one
								// sample is taken, and the power heuristic of volume_direct_scatter_mis is
								// not - its weight depends on which strategy came out, and near a light the
								// two branches give about 2C and 0 (measured, rendered at 64 samples against
								// 2048: 4.5% off with the power heuristic, 4.1% with this one). Equiangular
								// alone weighs by its own pdf
								let pointWeight = select( 1.0 / max( select( distancePdf, 0.5 * distancePdf + 0.5 * anglePdf, useEquiangular ), 1e-12 ),
									1.0 / max( anglePdf, 1e-12 ), byAngle && angleOnly );

								let directPoint = input.origin + input.direction * td;
								if ( distanceOnly ) {

									choice = ${ neeChooseFn }( directPoint, ra.x, envWeight, envActive );

								}
								let slot = i32( choice.x );
								var lightRec: ${ lightRecordStruct };
								lightRec.pdf = 0.0;
								if ( choice.y > 0.0 ) {

									lightRec = ${ neeSampleFn }( slot, directPoint, ${ rand2 }( ${ RNG_INDEX_MEDIUM_LIGHT } ), choice.z );

								}
								if ( lightRec.pdf > 0.0 && pointWeight > 0.0 ) {

									let phase = ${ stackPhaseFn }( input.mediumStack, phaseWeights, input.direction, lightRec.direction );
									let seenPdf = lightRec.pdf * ${ neeSlotChoiceFn }( slot, directPoint, envWeight, envActive );
									let misWeight = select( 1.0, ${ misHeuristicFn }( seenPdf, phase ), ${ isMISWeightLightFn }( lightRec.lightType ) && ! budgetSpent );
									let direct = throughputColor * sigmaS * directT * pointWeight * phase * lightRec.emission * misWeight / ( lightRec.pdf * choice.y );
									if ( any( direct > vec3f( 0.0 ) ) ) {

										let shadowIndex = atomicAdd( &shadowRayQueue.length, 1u );
										shadowRayQueue.elements[ shadowIndex ].origin = directPoint;
										shadowRayQueue.elements[ shadowIndex ].direction = lightRec.direction;
										shadowRayQueue.elements[ shadowIndex ].pixelIndex = input.pixelIndex;
										shadowRayQueue.elements[ shadowIndex ].currentBounce = input.currentBounce;
										shadowRayQueue.elements[ shadowIndex ].seed = input.seed;
										shadowRayQueue.elements[ shadowIndex ].alphaDepth = input.alphaDepth;
										shadowRayQueue.elements[ shadowIndex ].maxDist = lightRec.dist - ${ LIGHT_EPSILON };
										shadowRayQueue.elements[ shadowIndex ].mediumStack = input.mediumStack;
										rayDataStorage[ index ].mediumShadowIndex = i32( shadowIndex );
										// the light reaching a scatter is direct at the depth of the scatter
										rayDataStorage[ index ].mediumDirect = ${ clampPathContributionFunc }( direct, input.currentBounce + 1u, clampDirect, clampIndirect );

									}

								}

							}

							// -- THE CONTINUATION: distance sampling, untouched by the direct light --
							//
							// On a channel picked by albedo times throughput, as Cycles volume_sample_channel;
							// sigma_s T over the pdf on a scatter, T over the probability of getting this far
							// on a transmit. Past the budget nothing is drawn and the path goes on through,
							// weighed by the transmittance alone
							if ( budgetSpent ) {

								mediumWeight = exp( - sigmaT * surfaceDist );

							} else {

								mediumSampled = true;
								let u = ${ rand2 }( ${ RNG_INDEX_MEDIUM } );
								var channel = 2u;
								if ( u.x < channelP.r ) { channel = 0u; }
								else if ( u.x < channelP.r + channelP.g ) { channel = 1u; }
								let t = - log( max( 1.0 - u.y, 1e-9 ) ) / max( sigmaT[ channel ], 1e-9 );
								if ( t < surfaceDist ) {

									mediumScatter = true;
									mediumEnd = t;
									rayDataStorage[ index ].mediumPhaseWeights = phaseWeights;
									let transmittance = exp( - sigmaT * t );
									mediumWeight = sigmaS * transmittance / max( dot( channelP, sigmaT * transmittance ), 1e-9 );

								} else {

									let transmittance = exp( - sigmaT * surfaceDist );
									mediumWeight = transmittance / max( dot( channelP, transmittance ), 1e-9 );

								}

							}

						}

					}

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
						if ( ${ intersectLightAtIndexFn }( input.origin, input.direction, li, &lightRec ) && ( ( ! didHit && ! mediumScatter ) || lightRec.dist < mediumEnd ) ) {

							var misWeight = 1.0;
							if ( misEnabled != 0u && input.currentBounce > 0u ) {

								let choice = ${ neeSlotProbabilityFn }( ${ lightSlotWeightFn }( li, input.origin ), ${ lightSlotActiveFn }( li ), originTotals );
								misWeight = ${ misHeuristicFn }( input.scatterPdf, lightRec.pdf * choice );

							}

							// seen through the medium: its transmittance, over the probability of the
							// draw getting this far when there was one
							var mediumFactor = vec3f( 1.0 );
							if ( inMedium ) {

								let transmittance = exp( - sigmaT * lightRec.dist );
								mediumFactor = select( transmittance, transmittance / max( dot( channelP, transmittance ), 1e-9 ), mediumSampled );

							}
							lightHits += ${ clampPathContributionFunc }( lightRec.emission * throughputColor * misWeight * mediumFactor, input.currentBounce, clampDirect, clampIndirect );

						}

					}

					throughputColor *= mediumWeight;
					rayDataStorage[ index ].mediumScatter = select( 0u, 1u, mediumScatter );
					// the vertex the next light is chosen from: the surface, or the scatter point
					let vertexPosition = select( hitResult.position, input.origin + input.direction * mediumEnd, mediumScatter );

					if ( mediumScatter ) {

						// a live slot, whose object MaterialKernel does not read on a scatter
						rayDataStorage[ index ].objectIndex = max( hitResult.objectIndex, 0 );
						rayDataStorage[ index ].isCurve = 0u;
						rayDataStorage[ index ].dist = mediumEnd;

					}

					if ( didHit && ! mediumScatter ) {

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

					}

					if ( didHit || mediumScatter ) {

						// the emitter slot's probability at the vertex this segment left, for MaterialKernel,
						// which weighs the emission of this surface and has no room for the lights buffer
						rayDataStorage[ index ].emitterSelectPdf = ${ neeSlotProbabilityFn }( ${ emitterSlotWeightFn }( input.origin ), emitterCount > 0u, originTotals );

						// next event estimation: pick one slot by importance (neeSlotProbability) with a single
						// sample. MaterialKernel evaluates the bsdf and enqueues the shadow ray.
						var lightPdf = 0.0;
						let homogeneousScatter = mediumScatter && ! heterogeneous;
						if ( misEnabled != 0u && ! homogeneousScatter ) {

							let ruv = ${ rand3 }( ${ RNG_INDEX_DIRECT_LIGHT_SAMPLE } );
							let choice = ${ neeChooseFn }( vertexPosition, ruv.x, envWeight, envActive );
							var lightRec: ${ lightRecordStruct };
							lightRec.pdf = 0.0;
							if ( choice.y > 0.0 ) {

								lightRec = ${ neeSampleFn }( i32( choice.x ), vertexPosition, ruv.yz, choice.z );

							}
							lightPdf = lightRec.pdf * choice.y;
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

								// sample the background directly if this is the primary ray. ADDED to what the
								// camera segments gathered, not assigned over it: a medium crossed on the way out
								// - a fire in front of the sky - emits into the path before it escapes, and the
								// assignment erased it (measured: an emitting slab in front of the void read zero,
								// and 0.5000 with a black wall behind). The alpha stays the background's
								let background = ${ clampPathContributionFunc }( bg.a * bg.rgb, input.currentBounce, clampDirect, clampIndirect );
								resultColor = vec4f( resultColor.rgb + background, bg.a );

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
