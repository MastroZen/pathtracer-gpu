import { Vector2, Vector4 } from 'three';
import { StorageBufferAttribute, StorageTexture } from 'three/webgpu';
import { ComputeKernel } from '../ComputeKernel.js';
import { uniform, uniformArray, storage, textureStore, globalId } from 'three/tsl';
import { proxy, proxyFn, rayStruct, wgslTagFn } from 'three-mesh-bvh/webgpu';
import { mediumStackEnterFn, mediumStackExitFn, mediumStackPhaseFunc, mediumStackPickPhaseFunc } from '../../nodes/mediumStack.wgsl.js';
import { rngInit, rand1, rand2, rand3, RNG_INDEX_RAY_JITTER, RNG_INDEX_ALPHA_TEST, RNG_INDEX_RUSSIAN_ROULETTE, RNG_INDEX_DISPERSION_WAVELENGTH, RNG_INDEX_SUBSURFACE, RNG_INDEX_SUBSURFACE_WALK, RNG_INDEX_HAIR, RNG_INDEX_MEDIUM_PHASE } from '../../nodes/random.wgsl.js';
import { rayDataStruct, rayQueueAtomicStruct, pixelQueueStruct } from './structs.js';
import { SAMPLE_ACTIVE_FLAG, SAMPLE_COUNT_MASK, SAMPLE_DISPATCHED_FLAG } from '../../constants.js';
import { mixLeafFunc, applyDispersionFunc, dispersionColorWeightFunc, DISPERSION_MIN_WAVELENGTH, DISPERSION_MAX_WAVELENGTH, sampleHenyeyGreensteinFunc, henyeyGreensteinPhaseFunc, SUBSURFACE_MAX_STEPS, subsurfaceAlphaFunc, subsurfaceSigmaFunc } from '../../nodes/material.wgsl.js';
import { isTerminatingScatterFunc, offsetRayOriginFunc } from '../../nodes/utils.wgsl.js';
import { LIGHT_EPSILON } from '../../nodes/lights.wgsl.js';
import { misHeuristicFn } from '../../nodes/sampling.wgsl.js';
import { hairSetupFn, hairSigmaFn, hairMelaninFn, hairEvalFn, hairScatterFn } from '../../nodes/hairBsdf.wgsl.js';
import { huangBuildFrameFn, huangEvalFn, huangSampleFn, huangHairStruct } from '../../nodes/huangBsdf.wgsl.js';
import { GGX_GLASS_E, ggxGlassEFn } from '../../nodes/ggxGlassTable.wgsl.js';
import { svmRecordTexelFn, svmResultsTexture } from './SvmKernel.js';
import { PIXEL_FILTER_TABLE_SIZE, pixelFilterReadFunc } from '../../nodes/pixelFilter.wgsl.js';

// How many null boundaries of participating media one path may cross. They advance the alpha
// depth, so the next draws are not the ones the boundary hit already used, but they are not
// held to the transparent bounce limit: Cycles counts them apart too (volume_bounds_bounce)
const MEDIUM_BOUNDS_MAX = 128;

// Pure material evaluation and ray generation: terminated slots pull a recycled pixel and emit a
// fresh camera ray; live slots evaluate the surface staged by LogicKernel, sample the bsdf, and
// enqueue the next bounce ray plus the NEE shadow ray toward the light LogicKernel selected.
export class MaterialKernel extends ComputeKernel {

	constructor( ) {

		const params = {
			bvhData: { value: null },
			material: { value: null },
			misEnabled: uniform( 1, 'uint' ),

			seed: uniform( 0, 'uint' ),
			targetDimensions: uniform( new Vector2() ),
			maxSamples: uniform( 0, 'uint' ),
			rayCount: uniform( 0, 'uint' ),
			filterGlossy: uniform( 1 ),
			maxTransparentBounces: uniform( 5, 'uint' ),
			maxBounces: uniform( 5, 'uint' ),
			maxSubsurfaceSteps: uniform( SUBSURFACE_MAX_STEPS, 'uint' ),
			// Blender's Volume Bounces, whose default is zero: one scatter inside a medium, lit
			// by the direct light, and the path ends there (Cycles max_volume_bounce is it plus one)
			maxVolumeBounces: uniform( 0, 'uint' ),
			// the MEDIUM STACK the camera stands in, four words: decided on the host from the camera
			// position (cameraMedium.js), the volume stack init of Cycles
			cameraMedium0: uniform( 0xffffffff, 'uint' ),
			cameraMedium1: uniform( 0xffffffff, 'uint' ),
			cameraMedium2: uniform( 0xffffffff, 'uint' ),
			cameraMedium3: uniform( 0xffffffff, 'uint' ),

			sampleCountTarget: textureStore( new StorageTexture( 1, 1 ) ).toReadWrite(),

			rayDataStorage: storage( new StorageBufferAttribute( 1, 1 ), rayDataStruct ),
			rayQueue: storage( new StorageBufferAttribute( 1, 1 ), rayQueueAtomicStruct ),
			shadowRayQueue: storage( new StorageBufferAttribute( 1, 1 ), rayQueueAtomicStruct ),
			pixelQueue: storage( new StorageBufferAttribute( 1, 1 ), pixelQueueStruct ),

			// ── LA TABELLA DELL'ALBEDO DEL VETRO GGX, per il BSDF di Huang ──
			//
			// In un BUFFER e non dentro lo shader, ed e' misurato: come costante di
			// modulo da 4096 float dentro il codice di Huang, FXC (il compilatore HLSL
			// di D3D11) ci mette 21,8 secondi e poi fallisce con E_FAIL; con la stessa
			// tabella in un buffer lo shader compila in 555 ms.
			//
			// Porta questo kernel a OTTO storage buffer, che e' il minimo che WebGPU
			// garantisce: non ce ne sta un altro, e il prossimo dato di questa taglia
			// dovra' entrare da un'altra parte.
			ggxGlassTable: storage( new StorageBufferAttribute( GGX_GLASS_E, 1 ), 'float' ).toReadOnly(),

			// the sockets SvmKernel computed for this slot's hit: a storage TEXTURE,
			// because there is no ninth storage buffer to put them in
			svmResults: textureStore( svmResultsTexture( 1 ) ).toReadOnly(),

			// THE PIXEL FILTER OF CYCLES (scene/film.cpp filter_table): the inverted CDF of the
			// filter, read per camera ray. A uniform buffer and not a storage one: this kernel
			// already holds the eight storage buffers WebGPU guarantees. Off, the ray falls
			// uniformly inside the pixel, which is the box filter of width one
			pixelFilterOn: uniform( 0, 'uint' ),
			pixelFilterTable: uniformArray( Array.from( { length: PIXEL_FILTER_TABLE_SIZE / 4 }, () => new Vector4() ), 'vec4' ),

			globalId: globalId,
		};

		const materialsBuffer = proxy( 'bvhData.value.storage.materials', params );
		// the phase of a scatter in a stack of media: built ONCE, since two nodes of one function
		// would declare it twice in the shader
		const stackPhaseFn = mediumStackPhaseFunc( materialsBuffer, henyeyGreensteinPhaseFunc );
		const stackPickPhaseFn = mediumStackPickPhaseFunc( materialsBuffer );
		const transformsBuffer = proxy( 'bvhData.value.storage.transforms', params );
		const getCameraRayFn = proxyFn( 'bvhData.value.fns.getCameraRay', params );
		const sampleTrianglePointFn = proxyFn( 'bvhData.value.fns.sampleTrianglePoint', params );
		const getSurfaceRecordFn = proxyFn( 'bvhData.value.fns.getSurfaceRecord', params );
		const mixLeafFn = mixLeafFunc( materialsBuffer );
		const bsdfSampleFn = proxyFn( 'material.value.bsdfSample', params );
		const bsdfEvalPdfFn = proxyFn( 'material.value.bsdfEvalPdf', params );

		const pixelFilterReadFn = pixelFilterReadFunc( params.pixelFilterTable );

		const fn = wgslTagFn/* wgsl */`

			fn compute(
				seed: u32,
				targetDimensions: vec2u,
				maxSamples: u32,
				rayCount: u32,
				filterGlossy: f32,
				maxTransparentBounces: u32,
				maxBounces: u32,
				maxSubsurfaceSteps: u32,
				maxVolumeBounces: u32,
				cameraMedium0: u32,
				cameraMedium1: u32,
				cameraMedium2: u32,
				cameraMedium3: u32,
				misEnabled: u32,
				pixelFilterOn: u32,

				globalId: vec3u
			) -> void {

				let rayDataStorage = &${ params.rayDataStorage };
				let rayQueue = &${ params.rayQueue };
				let shadowRayQueue = &${ params.shadowRayQueue };
				let pixelQueue = &${ params.pixelQueue };

				let materials = &${ materialsBuffer };
				let transforms = &${ transformsBuffer };

				// A whole element is read from the BUFFER, never through the pointer alias
				// above: WebKit packs every struct that holds a vec3 and does not unpack a
				// load made through a let pointer, so Safari refused this kernel with
				// "no viable conversion from __typeN_Packed". Field reads and writes
				// through the alias compile, and stay as they are.

				// bound by "rayCount" rather than the pool length. The dispatch rounds up to the
				// workgroup size and those extra slots hold a zeroed pixel index
				let index = globalId.x;
				if ( index >= rayCount ) {

					return;

				}

				let input = ${ params.rayDataStorage }[ index ];
				if ( input.objectIndex < 0 ) {

					// the slot's path has terminated: recycle the pixel through the overflow queue and
					// generate a fresh camera ray
					var pixelIndex = input.pixelIndex;
					if ( pixelQueue.elementCount > 0u ) {

						// TODO: If we've pulled off a pixel that's already finished we currently just
						// write a no-op ray, wasting a frame. It may be better to iterate over a few
						// points in the queue to see if we can find one we can use.
						let queueIndex = atomicAdd( &pixelQueue.current, 1u ) % pixelQueue.elementCount;
						pixelIndex = atomicExchange( &pixelQueue.elements[ queueIndex ], pixelIndex );

					}

					let indexUV = vec2u( pixelIndex >> 16, pixelIndex & 0xFFFF );

					// skip the pixel if it has hit the sample limit
					let combinedField = textureLoad( ${ params.sampleCountTarget }, indexUV ).r;
					let samples = ( ${ SAMPLE_COUNT_MASK }u & combinedField );
					let isComplete = maxSamples != 0u && samples >= maxSamples;

					if ( isComplete ) {

						rayDataStorage[ index ].pixelIndex = pixelIndex;
						rayDataStorage[ index ].rayIntersectionIndex = - 1;
						rayDataStorage[ index ].shadowRayIntersectionIndex = - 1;
						return;

					}

					${ rngInit }( indexUV, seed + samples, 0 );

					let uv = vec2f( indexUV ) / vec2f( targetDimensions );
					// the pixel filter: a ray offset read from the table of the filter, and the first
					// sample at the pixel centre, as init_from_camera does in Cycles
					var pixelOffset = ${ rand2 }( ${ RNG_INDEX_RAY_JITTER } );
					if ( pixelFilterOn == 1u ) {

						let filterUv = select( pixelOffset, vec2f( 0.5 ), samples == 0u );
						pixelOffset = vec2f( ${ pixelFilterReadFn }( filterUv.x ), ${ pixelFilterReadFn }( filterUv.y ) );

					}
					let jitteredUv = uv + pixelOffset / vec2f( targetDimensions );
					var ray: ${ rayStruct };
					if ( ! ${ getCameraRayFn }( jitteredUv, vec2f( targetDimensions ), &ray ) ) {

						// the camera declined the pixel, so leave the slot dormant for this round
						// TODO: same as above - this work is a bit wasteful and leaves slots empty for
						// a frame. Is it possible to quickly skip rays outside the mask or are finished?
						rayDataStorage[ index ].pixelIndex = pixelIndex;
						rayDataStorage[ index ].rayIntersectionIndex = - 1;
						rayDataStorage[ index ].shadowRayIntersectionIndex = - 1;
						return;

					}

					ray.direction = normalize( ray.direction );

					let rayIndex = atomicAdd( &rayQueue.length, 1u );
					rayQueue.elements[ rayIndex ].origin = ray.origin;
					rayQueue.elements[ rayIndex ].direction = ray.direction;
					rayQueue.elements[ rayIndex ].pixelIndex = pixelIndex;
					rayQueue.elements[ rayIndex ].currentBounce = 0u;
					rayQueue.elements[ rayIndex ].seed = seed + samples;
					rayQueue.elements[ rayIndex ].alphaDepth = 0u;
					rayQueue.elements[ rayIndex ].maxDist = ray.maxDist;

					rayDataStorage[ index ].origin = ray.origin;
					rayDataStorage[ index ].direction = ray.direction;
					rayDataStorage[ index ].pixelIndex = pixelIndex;
					rayDataStorage[ index ].seed = seed + samples;
					rayDataStorage[ index ].currentBounce = 0u;
					rayDataStorage[ index ].throughputColor = vec3f( 1.0 );
					rayDataStorage[ index ].resultColor = vec4f( 0.0, 0.0, 0.0, 1.0 );
					rayDataStorage[ index ].scatterColor = vec3f( 1.0 );
					rayDataStorage[ index ].scatterPdf = 1.0;
					rayDataStorage[ index ].minPdf = 1.0;
					rayDataStorage[ index ].isFullyTransmissive = 1u;
					rayDataStorage[ index ].emission = vec3f( 0.0 );
					rayDataStorage[ index ].lightPdf = 0.0;
					rayDataStorage[ index ].alphaDepth = 0u;
					rayDataStorage[ index ].maxDist = ray.maxDist;
					rayDataStorage[ index ].rayIntersectionIndex = i32( rayIndex );
					rayDataStorage[ index ].shadowRayIntersectionIndex = - 1;
					rayDataStorage[ index ].mediumShadowIndex = - 1;
					// a camera ray starts in the media the camera stands in, which is none at all for a
					// camera in the open; and in no subsurface walk
					rayDataStorage[ index ].insideMaterial = - 1;
					rayDataStorage[ index ].mediumStack = vec4u( cameraMedium0, cameraMedium1, cameraMedium2, cameraMedium3 );
					rayDataStorage[ index ].subsurfaceSteps = 0u;
					rayDataStorage[ index ].mediumScatter = 0u;
					rayDataStorage[ index ].volumeBounce = 0u;
					rayDataStorage[ index ].dispersionWavelength = - mix( ${ DISPERSION_MIN_WAVELENGTH }.0, ${ DISPERSION_MAX_WAVELENGTH }.0, ${ rand1 }( ${ RNG_INDEX_DISPERSION_WAVELENGTH } ) );

					// write the active params & dispatched flag
					textureStore( ${ params.sampleCountTarget }, indexUV, vec4( ${ SAMPLE_ACTIVE_FLAG }u | ${ SAMPLE_DISPATCHED_FLAG }u | samples ) );

				} else {

					// evaluate the surface staged by LogicKernel
					let indexUV = vec2u( input.pixelIndex >> 16, input.pixelIndex & 0xFFFF );
					// the walk step counts like a bounce here: without it every step of a walk would
					// draw the same numbers, and the path would march in a straight line
					${ rngInit }( indexUV, input.seed, input.currentBounce + input.alphaDepth + input.subsurfaceSteps );

					// the path is walking a subsurface volume; the media it is inside of are its stack,
					// which every shadow ray from here starts with
					let inSubsurface = input.insideMaterial >= 0;

					// -- A SCATTER INSIDE THE MEDIUM, at the point LogicKernel chose --
					//
					// There is no surface here: the next direction is drawn from the phase function,
					// which weighs one against its own pdf, and the light LogicKernel picked from this
					// point is weighed by the same phase. When the volume bounce budget is spent the
					// path ends after the direct light, and that light takes the whole weight since no
					// bounce will find it from here - Cycles keeps the ray going through transparent
					// stuff only, which gathers the same light in expectation.
					if ( input.mediumScatter == 1u ) {

						// the phase of the point is the MIXTURE of the stack's, on the weights the scatter
						// wrote: an entry picked on them draws the direction, and the pdf is the mixture's
						let phaseDraw = ${ rand3 }( ${ RNG_INDEX_MEDIUM_PHASE } );
						let g = ${ stackPickPhaseFn }( input.mediumStack, input.mediumPhaseWeights, phaseDraw.z );
						let scatterPoint = input.origin + input.direction * input.dist;
						let newBounce = input.currentBounce + 1u;
						let newVolumeBounce = input.volumeBounce + 1u;
						let budgetSpent = newVolumeBounce > maxVolumeBounces || newBounce >= maxBounces;

						let nextDirection = ${ sampleHenyeyGreensteinFunc }( input.direction, g, phaseDraw.xy );
						let phasePdf = ${ stackPhaseFn }( input.mediumStack, input.mediumPhaseWeights, input.direction, nextDirection );
						var phaseColor = vec3f( phasePdf );
						var isTerminated = budgetSpent;
						if ( ! isTerminated && newBounce >= 3u ) {

							let rrProb = saturate( sqrt( max( max( input.throughputColor.r, input.throughputColor.g ), input.throughputColor.b ) ) );
							isTerminated = rrProb <= 0.0 || ${ rand1 }( ${ RNG_INDEX_RUSSIAN_ROULETTE } ) > rrProb;
							if ( ! isTerminated ) { phaseColor /= rrProb; }

						}

						rayDataStorage[ index ].currentBounce = newBounce;
						rayDataStorage[ index ].volumeBounce = newVolumeBounce;
						rayDataStorage[ index ].emission = vec3f( 0.0 );
						rayDataStorage[ index ].isFullyTransmissive = 0u;
						rayDataStorage[ index ].scatterColor = phaseColor;
						rayDataStorage[ index ].scatterPdf = select( phasePdf, 0.0, isTerminated );
						rayDataStorage[ index ].minPdf = min( input.minPdf, phasePdf );

						if ( ! isTerminated ) {

							let rayIndex = atomicAdd( &rayQueue.length, 1u );
							rayQueue.elements[ rayIndex ].origin = scatterPoint;
							rayQueue.elements[ rayIndex ].direction = nextDirection;
							rayQueue.elements[ rayIndex ].pixelIndex = input.pixelIndex;
							rayQueue.elements[ rayIndex ].currentBounce = newBounce;
							rayQueue.elements[ rayIndex ].seed = input.seed;
							rayQueue.elements[ rayIndex ].alphaDepth = input.alphaDepth;
							rayQueue.elements[ rayIndex ].maxDist = 0.0;
							rayDataStorage[ index ].rayIntersectionIndex = i32( rayIndex );
							rayDataStorage[ index ].origin = scatterPoint;
							rayDataStorage[ index ].direction = nextDirection;

						}

						if ( input.lightPdf > 0.0 ) {

							let phase = ${ stackPhaseFn }( input.mediumStack, input.mediumPhaseWeights, input.direction, input.lightDirection );
							rayDataStorage[ index ].lightBsdf = vec3f( phase );
							rayDataStorage[ index ].lightBsdfPdf = select( phase, 0.0, budgetSpent );

							let shadowIndex = atomicAdd( &shadowRayQueue.length, 1u );
							shadowRayQueue.elements[ shadowIndex ].origin = scatterPoint;
							shadowRayQueue.elements[ shadowIndex ].direction = input.lightDirection;
							shadowRayQueue.elements[ shadowIndex ].pixelIndex = input.pixelIndex;
							shadowRayQueue.elements[ shadowIndex ].currentBounce = input.currentBounce;
							shadowRayQueue.elements[ shadowIndex ].seed = input.seed;
							shadowRayQueue.elements[ shadowIndex ].alphaDepth = input.alphaDepth;
							shadowRayQueue.elements[ shadowIndex ].maxDist = input.lightDist - ${ LIGHT_EPSILON };
							shadowRayQueue.elements[ shadowIndex ].mediumStack = input.mediumStack;
							rayDataStorage[ index ].shadowRayIntersectionIndex = i32( shadowIndex );

						} else {

							rayDataStorage[ index ].lightPdf = 0.0;
							rayDataStorage[ index ].shadowRayIntersectionIndex = - 1;

						}
						return;

					}

					// ── SUBSURFACE: THE WALK INSIDE THE VOLUME ──
					//
					// Crossing in a straight line is not scattering: it makes a pane of frosted
					// glass, not skin. Inside the medium the path takes a step of exponentially
					// distributed length whose mean is the radius, and if that step ends BEFORE
					// the wall the path scatters right there and carries on in a new direction.
					// What leaves on the far side is soft and tinted because it was multiplied
					// by the albedo once per step - the deeper the path, the redder it comes out,
					// which is the whole look of skin.
					//
					// This is Cycles' random walk (subsurface_random_walk.h). Two divergences,
					// both deliberate: the step length is drawn from the MEAN of the three radii
					// instead of a channel picked at random - the colour still deepens with the
					// number of steps, only the per-channel depth is shared - and the walk has a
					// step budget instead of running to the end, which is how an open mesh is
					// truncated over there too.
					// ── SUBSURFACE: THE WALK INSIDE THE VOLUME ──
					//
					// Crossing in a straight line makes frosted glass, not skin. Inside the
					// medium the path takes a step of exponentially distributed length and, if
					// that step ends before the wall, scatters there and carries on. What leaves
					// is soft and tinted because it was multiplied by the albedo once per step,
					// and because each channel has its own mean free path: red goes about ten
					// times deeper through flesh than blue.
					//
					// This is Cycles' random walk (subsurface_random_walk.h) without its DWIVEDI
					// GUIDING, and the missing piece is a measured decision, not an omission. The
					// guide leans the direction toward the interface the path came in through, and
					// its strength comes from the diffusion length, which Cycles computes from the
					// HIGHEST of the three single scattering albedos - deliberately, to stay on
					// the safe side of fireflies. That leaves a vice: with a bright albedo the
					// diffusion length is huge and the guided distribution is uniform, so it
					// guides nothing; with a dark one the walk is absorbed in two or three
					// scatters and there is nothing to guide. Measured on a 2 m cube with a 1 cm
					// mean free path, noise between neighbouring pixels, denoiser off:
					//
					//   #fff1cc  10.28 guided against 9.19 - and the mean moves, 129.4 vs 131.4
					//   #cc6644   4.25 against 4.18, means equal
					//   #332211   1.30 against 1.27, means equal
					//
					// It was here for a while, forward half only, and it measured a 26% gain -
					// on a remap that was wrong. With the remap read from the source instead of
					// memory the gain evaporated, which is the whole lesson: an optimisation
					// measured on a broken foundation measures the foundation.
					var walkTransmittance = vec3f( 1.0 );
					if ( inSubsurface ) {

						let medium = ${ materialsBuffer }[ u32( input.insideMaterial ) ];

						// a radius near zero is a surface, not a medium: the path crosses straight
						// and the exit below handles it
						if ( max( medium.subsurfaceRadius.r, max( medium.subsurfaceRadius.g, medium.subsurfaceRadius.b ) ) > 1e-4 ) {

							// La conversione sta in subsurfaceAlpha / subsurfaceSigma, ed e' quella
							// di Cycles LETTA NEL SORGENTE: la prima versione era scritta a memoria
							// e sbagliava sia l'albedo (ignorava l'anisotropia, e aveva un pavimento
							// che di la' non c'e') sia l'estinzione (un fattore di accorciamento che
							// appartiene al profilo di diffusione, non al cammino).
							let alpha = ${ subsurfaceAlphaFunc }( medium.color, medium.subsurfaceAnisotropy );
							let sigma = ${ subsurfaceSigmaFunc }( medium.subsurfaceRadius );

							// ── IL CANALE SI SORTEGGIA SU alpha * throughput ──
							//
							// E' volume_sample_channel di Cycles: il canale che il cammino porta
							// gia' si pesca piu' spesso, e il peso resta vicino a uno. Si pesca QUI
							// e non un passo fa, sul throughput che c'e' davvero: portarlo costa il
							// 18% di rumore invece di risparmiarlo.
							var channelP = vec3f( 1.0 / 3.0 );
							let carried = max( input.throughputColor * alpha, vec3f( 0.0 ) );
							let carriedSum = carried.r + carried.g + carried.b;
							if ( carriedSum > 1e-9 ) { channelP = carried / carriedSum; }

							let pick = ${ rand1 }( ${ RNG_INDEX_SUBSURFACE_WALK } + 1 );
							var channel = 2u;
							if ( pick < channelP.r ) { channel = 0u; }
							else if ( pick < channelP.r + channelP.g ) { channel = 1u; }

							let stepDist = - log( max( 1.0 - ${ rand1 }( ${ RNG_INDEX_SUBSURFACE_WALK } ), 1e-9 ) )
								/ max( sigma[ channel ], 1e-6 );

							let reachedWall = stepDist >= input.dist;
							let travelled = min( stepDist, input.dist );
							let transmittance = exp( - sigma * travelled );

							// sigma_s * T / pdf diffondendo, T / pdf arrivando alla parete, con
							// l'euristica di bilancio sui tre canali
							let segmentPdf = select( sigma * transmittance, transmittance, reachedWall );
							let segmentWeight = select( alpha * sigma * transmittance, transmittance, reachedWall )
								/ max( dot( channelP, segmentPdf ), 1e-9 );

							if ( reachedWall ) {

								// il passo ha superato la parete: il cammino attraversa, attenuato da
								// quel che il mezzo ha assorbito. E' QUI che nasce il bordo rosso —
								// una traversata lunga tiene il rosso e perde il blu.
								walkTransmittance = segmentWeight;

							} else {

								// finiti i passi: il cammino si lascia cadere dov'e' invece di uscire
								// da qualche parte a caso. Perde la sua energia, ed e' il troncamento
								// che Cycles fa su una mesh non chiusa.
								if ( input.subsurfaceSteps >= maxSubsurfaceSteps ) {

									rayDataStorage[ index ].throughputColor = vec3f( 0.0 );
									rayDataStorage[ index ].emission = vec3f( 0.0 );
									rayDataStorage[ index ].lightPdf = 0.0;
									rayDataStorage[ index ].shadowRayIntersectionIndex = - 1;
									return;

								}

								let scatterPoint = input.origin + input.direction * travelled;
								let nextDirection = ${ sampleHenyeyGreensteinFunc }(
									input.direction, medium.subsurfaceAnisotropy,
									${ rand2 }( ${ RNG_INDEX_SUBSURFACE_WALK } + 2 ),
								);

								let walkIndex = atomicAdd( &rayQueue.length, 1u );
								rayQueue.elements[ walkIndex ].origin = scatterPoint;
								rayQueue.elements[ walkIndex ].direction = nextDirection;
								rayQueue.elements[ walkIndex ].pixelIndex = input.pixelIndex;
								rayQueue.elements[ walkIndex ].currentBounce = input.currentBounce;
								rayQueue.elements[ walkIndex ].seed = input.seed;
								rayQueue.elements[ walkIndex ].alphaDepth = input.alphaDepth;
								// la direzione e' cambiata, quindi il budget del segmento vecchio non
								// vale: zero traccia senza limite, come uno scatter qualsiasi
								rayQueue.elements[ walkIndex ].maxDist = 0.0;

								// la superficie non viene mai raggiunta, quindi non se ne mette in
								// scena niente: nessuna emissione e nessuna NEE. "scatterColor" porta
								// il peso moltiplicato per la pdf che LogicKernel dividera', che
								// lascia il throughput moltiplicato per il peso — lo stesso trucco
								// del passaggio attraverso l'alpha.
								rayDataStorage[ index ].subsurfaceSteps = input.subsurfaceSteps + 1u;
								rayDataStorage[ index ].emission = vec3f( 0.0 );
								rayDataStorage[ index ].scatterColor = segmentWeight * input.scatterPdf;
								rayDataStorage[ index ].lightPdf = 0.0;
								rayDataStorage[ index ].origin = scatterPoint;
								rayDataStorage[ index ].direction = nextDirection;
								rayDataStorage[ index ].rayIntersectionIndex = i32( walkIndex );
								rayDataStorage[ index ].shadowRayIntersectionIndex = - 1;
								return;

							}

						}

					}

					let objectInfo = ${ transformsBuffer }[ u32( input.objectIndex ) ];
					var materialInfo = ${ materialsBuffer }[ objectInfo.materialIndex ];

					// -- THE NULL BOUNDARY of a material that is only a medium (SD_HAS_ONLY_VOLUME) --
					//
					// Nothing to shade: the ray carries on straight, like an alpha pass through, and
					// what changes is the stack of the media it travels in - this one entered from
					// the front, left from the back (volume_stack_enter_exit).
					if ( ( materialInfo.mediumFlags & 1u ) != 0u && input.isCurve == 0u ) {

						if ( input.alphaDepth >= ${ MEDIUM_BOUNDS_MAX }u ) {

							rayDataStorage[ index ].throughputColor = vec3f( 0.0 );
							rayDataStorage[ index ].emission = vec3f( 0.0 );
							rayDataStorage[ index ].lightPdf = 0.0;
							rayDataStorage[ index ].shadowRayIntersectionIndex = - 1;
							return;

						}

						let word = ( objectInfo.materialIndex & 0xffffu ) | ( ( u32( input.objectIndex ) & 0xffffu ) << 16u );
						let stackNext = select( ${ mediumStackExitFn }( input.mediumStack, word ), ${ mediumStackEnterFn }( input.mediumStack, word ), input.side > 0.0 );

						let boundaryPoint = input.origin + input.direction * input.dist;
						let crossIndex = atomicAdd( &rayQueue.length, 1u );
						rayQueue.elements[ crossIndex ].origin = ${ offsetRayOriginFunc }( boundaryPoint, input.direction, input.normal );
						rayQueue.elements[ crossIndex ].direction = input.direction;
						rayQueue.elements[ crossIndex ].pixelIndex = input.pixelIndex;
						rayQueue.elements[ crossIndex ].currentBounce = input.currentBounce;
						rayQueue.elements[ crossIndex ].seed = input.seed;
						rayQueue.elements[ crossIndex ].alphaDepth = input.alphaDepth + 1u;
						rayQueue.elements[ crossIndex ].maxDist = max( input.maxDist - input.dist, 0.0 );

						// the same staging as the alpha pass through below: the throughput is left as it is
						rayDataStorage[ index ].alphaDepth = input.alphaDepth + 1u;
						rayDataStorage[ index ].emission = vec3f( 0.0 );
						rayDataStorage[ index ].scatterColor = vec3f( input.scatterPdf );
						rayDataStorage[ index ].lightPdf = 0.0;
						rayDataStorage[ index ].origin = rayQueue.elements[ crossIndex ].origin;
						rayDataStorage[ index ].rayIntersectionIndex = i32( crossIndex );
						rayDataStorage[ index ].shadowRayIntersectionIndex = - 1;
						rayDataStorage[ index ].mediumStack = stackNext;
						return;

					}
					// the emitter density of the TRIANGLE's material, read before a mix picks another
					// branch: the table knows only the material the mesh wears. Curves are never in it
					let emitterAreaPdf = select( materialInfo.emitterAreaPdf, 0.0, input.isCurve == 1u );

					// The surface point is sampled HERE and no longer below: the mix can read
					// a MASK, and a mask wants the uv of the hit.
					//
					// ── UN PELO NON HA VERTICI, e la superficie si costruisce a mano ──
					//
					// Un colpo su una curva porta una posizione, la normale del cono e la
					// tangente della fibra: non ci sono tre vertici da interpolare. La
					// chiamata si fa lo stesso, con baricentriche a zero, per avere una
					// struttura del TIPO giusto — quali attributi contenga lo decide la scena
					// (uv1, colori, tangenti: dipende dalle geometrie), quindi scriverla qui
					// a mano vorrebbe dire nominare campi che possono non esistere.
					//
					// La peluria vive in coordinate di MONDO: e' cotta dalla mesh gia'
					// trasformata, come un modificatore, quindi la posa dell'oggetto e' gia'
					// dentro i suoi punti e non va riapplicata. Il costo dichiarato e' che
					// muovere il volume vuole una ricottura.
					let isCurve = input.isCurve == 1u;
					var vertexData = ${ sampleTrianglePointFn }(
						select( input.barycoord, vec3f( 0.0 ), isCurve ),
						select( input.indices, vec3u( 0u ), isCurve ),
					);
					if ( isCurve ) {

						vertexData.position = vec4f( input.origin + input.direction * input.dist, 1.0 );
						vertexData.normal = vec4f( input.normal, 0.0 );
						// la tangente arriva nel campo delle baricentriche, e il "w" e' il verso
						// della bitangente: per una fibra ne vale uno qualsiasi purche' non zero,
						// che e' il modo in cui chi ombreggia riconosce "tangente assente"
						vertexData.tangent = vec4f( input.barycoord, 1.0 );
						vertexData.color = vec4f( 1.0 );

					} else {

						vertexData.normal = normalize( transpose( objectInfo.inverseMatrixWorld ) * vertexData.normal );
						vertexData.tangent = vec4f( ( objectInfo.matrixWorld * vec4f( vertexData.tangent.xyz, 0.0 ) ).xyz, vertexData.tangent.w );

					}

					// ── MIX SHADER: the leaf this hit shades (mixLeaf, material.wgsl.js) ──
					//
					// A chain that holds a program was walked by SvmKernel, with the same draws
					// this kernel would make, and its choice is READ: that kernel ran the
					// program of the leaf it picked. Every other chain is walked here. One
					// walker per hit, one function. The leaf also governs the NEE shadow ray
					// below - they share this record.
					//
					// The leaf's record is loaded even when the leaf is the root: skipping that
					// load measured 2.5% SLOWER, paired, on a room where every hit is a chain
					// with a program.
					let svmResolved = materialInfo.svmResolve != 0u;
					var svmLeafTexel = vec4u( 0u );
					if ( svmResolved ) {

						svmLeafTexel = textureLoad( ${ params.svmResults }, ${ svmRecordTexelFn }( index, 1u ) );
						materialInfo = ${ materialsBuffer }[ svmLeafTexel.x ];

					} else {

						_ = ${ mixLeafFn }( objectInfo.materialIndex, &materialInfo, false, vec4f( 0.0 ), vec4f( 0.0 ) );

					}

					// a matte surface hit by the camera ray renders as a fully transparent
					let isMatte = materialInfo.matte != 0 && input.currentBounce == 0u;
					if ( isMatte ) {

						rayDataStorage[ index ].resultColor = vec4f( 0.0 );
						rayDataStorage[ index ].throughputColor = vec3f( 0.0 );
						rayDataStorage[ index ].emission = vec3f( 0.0 );
						rayDataStorage[ index ].lightPdf = 0.0;
						rayDataStorage[ index ].shadowRayIntersectionIndex = - 1;
						return;

					}

					// apply per-object colors
					materialInfo.color *= objectInfo.color.rgb;
					materialInfo.opacity *= objectInfo.color.a;

					if ( ! isCurve ) { vertexData.position = objectInfo.matrixWorld * vertexData.position; }

					// ── SUBSURFACE: the exit is a NEW diffuse point, not the hit it arrived at ──
					//
					// A path travelling inside the volume meets the wall from behind, and that is
					// where the light leaves. Shading it as an ordinary hit does not work: the
					// normal faces inward, so the sampled lobe bounces back into the volume and
					// NEE toward a light that is OUTSIDE has a zero pdf. Measured: the shadowed
					// face stayed at 0.08 even with the weight forced to 1 in the kernel.
					//
					// Cycles solves the same thing in bssrdf_setup - the exit point does not
					// inherit the hit, it is rebuilt as if the light had arrived straight down
					// the OUTWARD normal (sd->wi = sd->N). Everything downstream - sampling, NEE,
					// shadow rays - is then the usual path.
					var hitNormal = input.normal;
					var hitSide = input.side;
					var view = - input.direction;
					if ( inSubsurface ) {

						// "input.normal" always faces the incoming ray, so on the way out it points
						// INTO the volume: the geometric one is input.normal * input.side, and it is
						// passed as though the hit came from outside (side = +1). Flipping only the
						// side looks like it works and does not: with smooth shading the vertex
						// normal puts it right by accident, with flat shading it does not.
						hitNormal = input.normal * input.side;
						hitSide = 1.0;
						view = hitNormal;

					}

					// blur glossy surfaces after low-probability bounces to suppress fireflies,
					// from the Cycles "filter glossy" approach in integrator/surface_shader.h
					let blurRoughness = sqrt( clamp( 1.0 - filterGlossy * input.minPdf, 0.0, 1.0 ) ) * 0.5;

					// the tangent normal of the leaf's program, when it drives the normal: the fourth
					// register of its outputs, x and y as two halves and z as a float in the second
					// texel of the record
					var liveNormal = vec3f( 0.0 );
					if ( svmResolved && ! isCurve && materialInfo.svmCount > 0u && ( materialInfo.svmOutputs >> 24u ) != 255u ) {

						liveNormal = vec3f( unpack2x16float( svmLeafTexel.y ), bitcast<f32>( svmLeafTexel.z ) );

					}

					var surface = ${ getSurfaceRecordFn }( materialInfo, vertexData, hitSide, hitNormal, view, blurRoughness, liveNormal );

					// ── THE NODE MACHINE: the sockets the leaf's program drives, run by SvmKernel ──
					//
					// They replace what getSurfaceRecord made of the flat values. The albedo is
					// linear as it stands, like the preview's. A fibre runs none: its colour
					// comes from the hair fields, and SvmKernel skips its program.
					if ( svmResolved && ! isCurve && materialInfo.svmCount > 0u ) {

						let svm = textureLoad( ${ params.svmResults }, ${ svmRecordTexelFn }( index, 0u ) );
						let svmRoughMetal = unpack2x16float( svm.w );
						let svmOutputs = materialInfo.svmOutputs;
						if ( ( svmOutputs & 0xffu ) != 255u ) { surface.color = bitcast<vec3f>( svm.xyz ); }
						if ( ( ( svmOutputs >> 8u ) & 0xffu ) != 255u ) { surface.roughness = clamp( svmRoughMetal.x, max( MIN_ROUGHNESS, blurRoughness ), 1.0 ); }
						if ( ( ( svmOutputs >> 16u ) & 0xffu ) != 255u ) { surface.metalness = svmRoughMetal.y; }

						// an Emission leaf: the colour the program made is the light the surface
						// gives off, and the record already scatters nothing (black, no specular)
						if ( materialInfo.svmEmission != 0u ) {

							surface.emission = surface.color;
							surface.color = vec3f( 0.0 );

						}

					}

					// Stochastically pass through partially transparent surfaces by re-enqueueing
					// the ray at the hit point, advancing the alpha depth but not the bounce count.
					let passesThrough = ${ rand1 }( ${ RNG_INDEX_ALPHA_TEST } ) > surface.opacity;
					if ( passesThrough ) {

						// out of transparent bounces, so stop rather than shading a surface that
						// should be invisible. A zeroed throughput terminates in LogicKernel.
						if ( input.alphaDepth >= maxTransparentBounces ) {

							rayDataStorage[ index ].throughputColor = vec3f( 0.0 );
							rayDataStorage[ index ].emission = vec3f( 0.0 );
							rayDataStorage[ index ].lightPdf = 0.0;
							rayDataStorage[ index ].shadowRayIntersectionIndex = - 1;
							return;

						}

						let alphaIndex = atomicAdd( &rayQueue.length, 1u );
						rayQueue.elements[ alphaIndex ].origin = ${ offsetRayOriginFunc }( vertexData.position.xyz, input.direction, input.normal );
						rayQueue.elements[ alphaIndex ].direction = input.direction;
						rayQueue.elements[ alphaIndex ].pixelIndex = input.pixelIndex;
						rayQueue.elements[ alphaIndex ].currentBounce = input.currentBounce;
						rayQueue.elements[ alphaIndex ].seed = input.seed;
						rayQueue.elements[ alphaIndex ].alphaDepth = input.alphaDepth + 1u;
						// the origin advanced to the hit point, so the remaining budget shrinks by the
						// distance already traced
						rayQueue.elements[ alphaIndex ].maxDist = max( input.maxDist - input.dist, 0.0 );

						// the surface is skipped, so no scatter or emission is staged for LogicKernel.
						// "pdf" is left alone so the previous scatter still weights the forward MIS,
						// and "bsdf" matches it so applying the scatter leaves the throughput as is.
						rayDataStorage[ index ].alphaDepth = input.alphaDepth + 1u;
						rayDataStorage[ index ].emission = vec3f( 0.0 );
						rayDataStorage[ index ].scatterColor = vec3f( input.scatterPdf );
						rayDataStorage[ index ].lightPdf = 0.0;
						rayDataStorage[ index ].origin = rayQueue.elements[ alphaIndex ].origin;
						rayDataStorage[ index ].rayIntersectionIndex = i32( alphaIndex );
						rayDataStorage[ index ].shadowRayIntersectionIndex = - 1;
						return;

					}

					// apply the hero wavelength to dispersive surfaces, folding the spectral weight
					// into the throughput at the path's first dispersive interaction
					var throughputColor = input.throughputColor;
					// what the medium absorbed on the way to this wall. One everywhere else, so
					// the multiplication costs nothing when there is no volume.
					if ( inSubsurface ) {

						throughputColor *= walkTransmittance;
						rayDataStorage[ index ].throughputColor = throughputColor;

					}
					let isDispersive = materialInfo.dispersion > 0.0 && surface.ior > 1.0 && surface.transmission > 0.0 && ! surface.thinWall;
					if ( isDispersive ) {

						let wavelength = abs( input.dispersionWavelength );
						${ applyDispersionFunc }( &surface, materialInfo.dispersion, wavelength );
						if ( input.dispersionWavelength < 0.0 ) {

							rayDataStorage[ index ].dispersionWavelength = wavelength;
							throughputColor *= ${ dispersionColorWeightFunc }( wavelength );
							rayDataStorage[ index ].throughputColor = throughputColor;

						}

					}

					// the absorption inside a transmissive volume is its MEDIUM, entered and left with the
					// stack and integrated by LogicKernel along the segment: the glTF attenuation that
					// was applied here on a back face exit is the second home of the same data, and left
					// (docs/mezzi.md §10 in the app)

					// ── IL PELO HA UN BSDF SUO: i tre lobi di Chiang ──
					//
					// Una fibra non e' una superficie. La luce ci entra, ci gira dentro e ne
					// esce, e i cammini che contano sono tre: R rimbalza sulla cuticola
					// (riflesso primario, BIANCO), TT attraversa (il controluce, e porta il
					// colore), TRT si riflette dentro e esce (il secondo riflesso, staccato
					// dal primo dall'inclinazione delle scaglie). Il modello sta in
					// nodes/hairBsdf.wgsl.js, col gemello provato in Node.
					//
					// I PARAMETRI VENGONO DAL MATERIALE, dove li ha messi il ponte
					// dell'applicazione leggendoli dal VOLUME: il pacchetto della peluria e'
					// legato ai kernel di tracciamento, non a questo, quindi di li' non si
					// puo' leggere niente. Il colore invece e' quello della superficie, che
					// per un pelo diventa assorbimento.
					//
					// ── E SI CALCOLANO SOLO SE SERVE ──
					//
					// Stavano fuori dal ramo, cioe' su OGNI colpo della scena: due logaritmi
					// e una catena di potenze per un triangolo che non e' un pelo. Dichiarati
					// fuori e assegnati dentro costano zero a chi non ha peluria — e servono
					// in due punti (lo scatter e la NEE), che e' il motivo per cui non stanno
					// semplicemente dentro un blocco.
					var hairTangent = vec3f( 1.0, 0.0, 0.0 );
					var hairFy = vec3f( 0.0, 1.0, 0.0 );
					var hairFz = vec3f( 0.0, 0.0, 1.0 );
					var hairH = 0.0;
					var hairVsm = vec3f( 0.1, 0.1, 0.1 );
					var hairAbsorption = vec3f( 0.0 );
					var hairAlpha = 0.0;
					var hairEta = 1.55;
					var huang: ${ huangHairStruct };
					var huangFrame = ${ huangBuildFrameFn }( vec3f( 1.0, 0.0, 0.0 ), vec3f( 0.0, 0.0, 1.0 ), vec3f( 0.0, 1.0, 0.0 ), 1.0 );
					var huangRng = 1u;
					let useHuang = isCurve && materialInfo.hairModel > 0.5;
					if ( isCurve ) {

						hairTangent = normalize( input.barycoord );
						hairFy = normalize( cross( hairTangent, view ) );
						hairFz = normalize( cross( hairTangent, hairFy ) );
						hairH = clamp( dot( cross( input.normal, hairTangent ), hairFz ), -1.0, 1.0 );

						// ── LA VARIAZIONE PER CIOCCA ──
						//
						// Il numero arriva nella terza parola degli indici, scritto dal kernel
						// di tracciamento: qui il pacchetto della peluria non e' legato, e
						// senza quel viaggio un manto sarebbe N copie dello stesso pelo — che
						// e' la differenza fra una pelliccia e una moquette.
						//
						// I due fattori sono quelli di Cycles: uno piu' due volte lo scarto da
						// mezzo, per quanto si vuole variare. E' una variazione CENTRATA —
						// meta' delle ciocche piu' chiare, meta' piu' scure — invece di uno
						// scarto che sposta la media.
						let hairRandom = bitcast<f32>( input.indices.z );
						let hairJitter = 2.0 * ( hairRandom - 0.5 );
						let hairRoughFactor = 1.0 + hairJitter * materialInfo.hairRandomRoughness;
						let hairRough = clamp( materialInfo.hairRoughness * hairRoughFactor, 0.02, 1.0 );
						let hairRadial = clamp( materialInfo.hairRadialRoughness * hairRoughFactor, 0.02, 1.0 );
						hairVsm = ${ hairSetupFn }( hairRough, hairRadial, materialInfo.hairCoat );

						// ── IL COLORE: TRE MODI, e si escludono ──
						//
						// E' il parametrization del nodo Principled Hair, letto nel sorgente
						// della 5.2: riflettanza (il colore e basta), pigmenti (melanina e
						// rossore, PIU' una tinta che si somma — la somma sta qui dentro,
						// dove Cycles la fa), o il coefficiente nudo.
						//
						// PRIMA SI SOMMAVANO SEMPRE, e la nota lo dichiarava una virtu':
						// melanina a zero lasciava il colore del materiale. Il conto era
						// giusto e la conclusione no — sommate, la melanina e il colore sono
						// due manopole sulla stessa cosa, e quale comandi si scopre provando.
						//
						// LA VARIAZIONE PER CIOCCA TOCCA TUTTO l'assorbimento, e di la' varia
						// solo la melanina. Divergenza dichiarata: cosi' la manopola fa
						// qualcosa in tutti e tre i modi invece che in uno solo.
						let hairColorFactor = max( 0.0, 1.0 + hairJitter * materialInfo.hairRandomColor );
						var hairSigmaBase: vec3f;
						if ( materialInfo.hairParametrization < 0.5 ) {

							hairSigmaBase = ${ hairSigmaFn }( materialInfo.color, hairRadial );

						} else if ( materialInfo.hairParametrization < 1.5 ) {

							hairSigmaBase = ${ hairMelaninFn }( materialInfo.hairMelanin, materialInfo.hairRedness )
								+ ${ hairSigmaFn }( vec3f( materialInfo.hairTintR, materialInfo.hairTintG, materialInfo.hairTintB ), hairRadial );

						} else {

							hairSigmaBase = vec3f( materialInfo.hairAbsorptionR, materialInfo.hairAbsorptionG, materialInfo.hairAbsorptionB );

						}
						hairAbsorption = hairSigmaBase * hairColorFactor;
						hairAlpha = - materialInfo.hairTilt;
						hairEta = max( materialInfo.hairIor, 1.001 );

						// ── E SE IL VOLUME CHIEDE HUANG, si prepara l'altro ──
						//
						// Huang non e' Chiang con piu' manopole: integra sulla sezione invece
						// di descrivere i lobi, e da li' vengono la sezione ELLITTICA e i
						// glint. Il record e' suo, e il frame anche — il suo asse X si allinea
						// all'asse maggiore dell'ellisse invece che alla vista.
						if ( materialInfo.hairModel > 0.5 ) {

							huang.sigma = hairAbsorption;
							huang.roughness = clamp( hairRough, 0.001, 1.0 );
							huang.tilt = hairAlpha;
							huang.eta = hairEta;
							huang.aspectRatio = clamp( materialInfo.hairAspect, 0.05, 1.0 );
							huang.r = 1.0;
							huang.tt = 1.0;
							huang.trt = 1.0;
							huangFrame = ${ huangBuildFrameFn }( hairTangent, view, input.normal, huang.aspectRatio );
							// il GENERATORE del cammino dentro la fibra: Huang ne consuma
							// sei-dieci per valutazione, e non ci sono dimensioni riservate per
							// tante. Si semina dal percorso, cosi' due pixel non camminano
							// uguale e lo stesso pixel non ripete se stesso fra un rimbalzo e
							// l'altro.
							huangRng = input.seed * 747796405u + input.pixelIndex * 2891336453u + input.currentBounce * 277803737u + 1u;

						}

					}

					// sample the next bounce direction and stage the scatter state for LogicKernel
					var scatterRec = ${ bsdfSampleFn }( view, surface );
					if ( isCurve ) {

						if ( useHuang && huangFrame.valid == 1u ) {

							let hu = ${ huangSampleFn }( &${ params.ggxGlassTable }, huang, huangFrame, &huangRng );
							// la PDF di Huang e' UNO da entrambe le parti: il peso del
							// campionamento e' gia' dentro il valore, e quel numero serve solo
							// al MIS — che confronta due stime della stessa cosa, quindi conta
							// che le due parti dicano lo stesso.
							scatterRec.color = select( vec3f( 0.0 ), hu.f, hu.valid == 1u );
							scatterRec.pdf = 1.0;
							scatterRec.direction = hu.direction;
							scatterRec.isTransmissive = false;

						} else {

						let hairRand = ${ rand3 }( ${ RNG_INDEX_HAIR } );
						let hs = ${ hairScatterFn }( hairAbsorption, hairVsm.x, hairVsm.y, hairVsm.z, hairAlpha, hairEta, hairTangent, hairFy, hairFz, hairH, view, hairRand );
						// NIENTE COSENO: la F di Chiang integra gia' all'albedo sulla sfera,
						// quindi il fattore di proiezione e' dentro. Moltiplicarlo un'altra
						// volta scurirebbe il manto di un coseno, e non lo direbbe nessuno.
						scatterRec.color = hs.f;
						scatterRec.pdf = hs.pdf;
						scatterRec.direction = hs.direction;
						// solo il lobo R e' una riflessione: gli altri tre passano DENTRO il
						// pelo, ed e' quel che i modi dello sfondo chiamano trasmissivo
						scatterRec.isTransmissive = hs.lobe != 0u;

						}

					}

					// ── SUBSURFACE: go IN, and come back OUT ──
					//
					// Entering is a closure pick, the same shape as the mix above: with
					// probability "subsurfaceWeight" the hit does not scatter off the surface,
					// it crosses it and the path continues INSIDE the volume. The next time
					// that path meets a surface it leaves, and that exit point is where the
					// light appears to come out — which is the whole phenomenon: light that
					// goes in here and leaves over there.
					//
					// The direction is MIRRORED through the surface plane rather than sampled
					// again: a cosine lobe around the normal becomes a cosine lobe around the
					// opposite normal, with the same pdf. One reflection instead of a second
					// basis and a second pair of random numbers.
					// the medium of this surface's material, when it has one, as the stack names it: the
					// record is the object's - the root of a Mix, whose leaves have no medium of their own
					let ownMediumFlags = ${ materialsBuffer }[ objectInfo.materialIndex ].mediumFlags;
					let ownMediumWord = ( objectInfo.materialIndex & 0xffffu ) | ( ( u32( input.objectIndex ) & 0xffffu ) << 16u );
					let ownMediumCrossable = ( ownMediumFlags & 2u ) != 0u && input.isCurve == 0u && ! surface.thinWall;

					var insideNext = input.insideMaterial;
					// a hit that goes in has PICKED the bssrdf closure, so the surface one is not
					// there to be lit: see the NEE block at the bottom
					var enteredSubsurface = false;
					if ( inSubsurface ) {

						// the path was travelling inside and has reached the surface: it leaves.
						// The sampled direction already points outward, because the surface above
						// was rebuilt with the OUTWARD normal.
						insideNext = - 1;
						rayDataStorage[ index ].subsurfaceSteps = 0u;

					} else if ( materialInfo.subsurfaceWeight > 0.0
						&& ${ rand1 }( ${ RNG_INDEX_SUBSURFACE } ) < materialInfo.subsurfaceWeight ) {

						let faceNormal = input.normal * input.side;
						scatterRec.direction = scatterRec.direction - 2.0 * dot( scatterRec.direction, faceNormal ) * faceNormal;
						insideNext = i32( objectInfo.materialIndex );
						enteredSubsurface = true;

						// ── E IL COLORE SI TOGLIE, perche' il cammino lo rimettera' ──
						//
						// Il peso della closure che il cammino porta dentro contiene GIA' l'albedo:
						// la BSDF appena campionata lo ha applicato. Dentro il volume lo rimette
						// una volta per diffusione, quindi lasciarlo qui vuol dire applicarlo due
						// volte e la materia esce troppo scura. Di la' e' una riga sola,
						// throughput = safe_divide_color(throughput, albedo), e qui si fa sul
						// colore dello scatter perche' e' lui che LogicKernel moltiplichera'.
						scatterRec.color = scatterRec.color / max( materialInfo.color, vec3f( 1e-4 ) );

						rayDataStorage[ index ].subsurfaceSteps = 0u;

					}
					rayDataStorage[ index ].insideMaterial = insideNext;

					// ── THE SURFACE WITH ITS VOLUME (shade_surface.h: volume_stack_enter_exit on a transmit) ──
					//
					// A path that CROSSES a surface whose material has a medium enters it from the front
					// and leaves it from the back; a reflection stays where it was, and so does a path
					// going into a subsurface walk, which integrates its own volume. A thin wall has no
					// inside to be in. CROSSING is read on the two directions, the arriving one and the
					// new one on the same side of the plane: the orientation of the hit normal does not
					// enter it (with normal times side the exit was never seen, and a glass with its
					// volume absorbed all the way to the wall behind it: 0.108 for an expected 0.607)
					if ( ownMediumCrossable && ! inSubsurface && ! enteredSubsurface
						&& dot( scatterRec.direction, input.normal ) * dot( input.direction, input.normal ) > 0.0 ) {

						rayDataStorage[ index ].mediumStack = select(
							${ mediumStackExitFn }( input.mediumStack, ownMediumWord ),
							${ mediumStackEnterFn }( input.mediumStack, ownMediumWord ),
							input.side > 0.0,
						);

					}

					let newBounce = input.currentBounce + 1u;

					// decide termination now so finished paths skip the bounce trace entirely - a
					// zeroed pdf reads as a terminating scatter in LogicKernel, which still resolves
					// the surface's emission and NEE before freeing the slot
					var isTerminated = newBounce >= maxBounces || all( scatterRec.color == vec3f( 0.0 ) ) || ${ isTerminatingScatterFunc }( scatterRec );

					// russian roulette early out:
					// Matches Cycles path_state_continuation_probability in integrator/path_state.h
					if ( ! isTerminated && newBounce >= 3u ) {

						let rrThroughput = throughputColor * scatterRec.color / scatterRec.pdf;
						let rrProb = saturate( sqrt( max( max( rrThroughput.r, rrThroughput.g ), rrThroughput.b ) ) );
						isTerminated = rrProb <= 0.0 || ${ rand1 }( ${ RNG_INDEX_RUSSIAN_ROULETTE } ) > rrProb;
						if ( ! isTerminated ) {

							// fold the survival boost into the scatter color so LogicKernel's
							// throughput update applies it without a separate division
							scatterRec.color /= rrProb;

						}

					}

					// Write the ray storage content here since things like the emissive value is read
					// in the logic kernel on the subsequent frame.
					rayDataStorage[ index ].scatterColor = scatterRec.color;
					rayDataStorage[ index ].scatterPdf = select( scatterRec.pdf, 0.0, isTerminated );
					rayDataStorage[ index ].minPdf = min( input.minPdf, scatterRec.pdf );
					rayDataStorage[ index ].isFullyTransmissive = input.isFullyTransmissive & select( 0u, 1u, scatterRec.isTransmissive );
					// -- EMISSION TAKES MIS -- a bsdf ray that lands on a triangle of the emitter table
					// found light NEE could have sampled too: weigh it with the pdf NEE would have given
					// this point from the previous vertex. "scatterPdf" and "dist" are still those of the
					// segment that got here, and LogicKernel left the probability the NEE choice gave
					// the table at that vertex: this kernel has no room for the lights buffer to work it
					// out. The camera segment and emitters outside the table keep full weight
					var weightedEmission = surface.emission;
					if ( misEnabled != 0u && input.currentBounce > 0u && emitterAreaPdf > 0.0 ) {

						let cosLight = abs( dot( input.normal, input.direction ) );
						let lightPdf = emitterAreaPdf * input.dist * input.dist / max( cosLight, 1e-6 ) * input.emitterSelectPdf;
						weightedEmission *= ${ misHeuristicFn }( input.scatterPdf, lightPdf );

					}
					rayDataStorage[ index ].emission = weightedEmission;
					rayDataStorage[ index ].currentBounce = newBounce;

					// the NEE shadow ray below still resolves the surface's direct light, so only
					// the bounce segment is skipped for finished paths
					if ( ! isTerminated ) {

						let rayIndex = atomicAdd( &rayQueue.length, 1u );
						rayQueue.elements[ rayIndex ].origin = ${ offsetRayOriginFunc }( vertexData.position.xyz, scatterRec.direction, input.normal );
						rayQueue.elements[ rayIndex ].direction = scatterRec.direction;
						rayQueue.elements[ rayIndex ].pixelIndex = input.pixelIndex;
						rayQueue.elements[ rayIndex ].currentBounce = newBounce;
						rayQueue.elements[ rayIndex ].seed = input.seed;
						rayQueue.elements[ rayIndex ].alphaDepth = input.alphaDepth;
						rayQueue.elements[ rayIndex ].maxDist = 0.0;
						rayDataStorage[ index ].rayIntersectionIndex = i32( rayIndex );

						rayDataStorage[ index ].origin = rayQueue.elements[ rayIndex ].origin;
						rayDataStorage[ index ].direction = scatterRec.direction;

					}

					// evaluate the bsdf toward the light LogicKernel selected and enqueue the shadow ray.
					// the light pdf will be 0 if NEE is disabled.
					//
					// A hit that entered the volume is skipped: the pick chose the bssrdf, and
					// lighting the surface closure as well would ADD the light instead of moving
					// it. Measured on a slab with the weight at 1 - the lit face lost 0.68 while
					// the shadowed one gained 19.18, when the lit face should go nearly dark. It
					// is the same rule the mix already follows: one closure per hit, and NEE is
					// for the one that was picked.
					var lightPdf = select( input.lightPdf, 0.0, enteredSubsurface );
					if ( lightPdf > 0.0 ) {

						var evalRec = ${ bsdfEvalPdfFn }( view, input.lightDirection, surface );
						if ( isCurve ) {

							if ( useHuang && huangFrame.valid == 1u ) {

								let hu = ${ huangEvalFn }( &${ params.ggxGlassTable }, huang, huangFrame, input.lightDirection, &huangRng );
								evalRec.color = hu;
								evalRec.pdf = 1.0;

							} else {

								let he = ${ hairEvalFn }( hairAbsorption, hairVsm.x, hairVsm.y, hairVsm.z, hairAlpha, hairEta, hairTangent, hairFy, hairFz, hairH, view, input.lightDirection );
								evalRec.color = he.f;
								evalRec.pdf = he.pdf;

							}

						}
						if ( evalRec.pdf > 0.0 ) {

							rayDataStorage[ index ].lightBsdf = evalRec.color;
							rayDataStorage[ index ].lightBsdfPdf = evalRec.pdf;

							let shadowIndex = atomicAdd( &shadowRayQueue.length, 1u );
							shadowRayQueue.elements[ shadowIndex ].origin = ${ offsetRayOriginFunc }( vertexData.position.xyz, input.lightDirection, input.normal );
							shadowRayQueue.elements[ shadowIndex ].direction = input.lightDirection;
							shadowRayQueue.elements[ shadowIndex ].pixelIndex = input.pixelIndex;
							shadowRayQueue.elements[ shadowIndex ].currentBounce = input.currentBounce;
							shadowRayQueue.elements[ shadowIndex ].seed = input.seed;
							shadowRayQueue.elements[ shadowIndex ].alphaDepth = input.alphaDepth;
							shadowRayQueue.elements[ shadowIndex ].maxDist = input.lightDist - ${ LIGHT_EPSILON };
							// a surface inside a medium sends its shadow ray from inside it, and one that
							// lights itself THROUGH its own medium (the light on the other side) starts the
							// shadow across it, as Cycles does for a transmission NEE
							var shadowStack = input.mediumStack;
							if ( ownMediumCrossable && dot( input.lightDirection, input.normal ) * dot( input.direction, input.normal ) > 0.0 ) {

								shadowStack = select(
									${ mediumStackExitFn }( input.mediumStack, ownMediumWord ),
									${ mediumStackEnterFn }( input.mediumStack, ownMediumWord ),
									input.side > 0.0,
								);

							}
							shadowRayQueue.elements[ shadowIndex ].mediumStack = shadowStack;
							rayDataStorage[ index ].shadowRayIntersectionIndex = i32( shadowIndex );

						} else {

							lightPdf = 0.0;

						}

					}

					if ( lightPdf <= 0.0 ) {

						rayDataStorage[ index ].lightPdf = 0.0;
						rayDataStorage[ index ].shadowRayIntersectionIndex = - 1;

					}

				}

			}
		`;

		super( fn( params ) );

		this.defineUniformAccessors( params );
		// the table node itself: the accessor above would swap its padded buffer, and the node
		// uploads from its array of vectors at every dispatch
		this.pixelFilterNode = params.pixelFilterTable;

	}

}
