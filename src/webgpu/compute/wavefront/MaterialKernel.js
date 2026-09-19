import { Vector2 } from 'three';
import { StorageBufferAttribute, StorageTexture } from 'three/webgpu';
import { ComputeKernel } from '../ComputeKernel.js';
import { uniform, storage, textureStore, globalId } from 'three/tsl';
import { proxy, proxyFn, rayStruct, wgslTagFn } from 'three-mesh-bvh/webgpu';
import { rngInit, rand1, rand2, RNG_INDEX_RAY_JITTER, RNG_INDEX_ALPHA_TEST, RNG_INDEX_RUSSIAN_ROULETTE, RNG_INDEX_DISPERSION_WAVELENGTH, RNG_INDEX_MIX_SHADER, RNG_INDEX_MIX_SHADER_COUNT, RNG_INDEX_SUBSURFACE, RNG_INDEX_SUBSURFACE_WALK } from '../../nodes/random.wgsl.js';
import { rayDataStruct, rayQueueAtomicStruct, pixelQueueStruct } from './structs.js';
import { SAMPLE_ACTIVE_FLAG, SAMPLE_COUNT_MASK, SAMPLE_DISPATCHED_FLAG } from '../../constants.js';
import { applyDispersionFunc, dispersionColorWeightFunc, DISPERSION_MIN_WAVELENGTH, DISPERSION_MAX_WAVELENGTH, transmissionAttenuationFunc, sampleHenyeyGreensteinFunc, SUBSURFACE_MAX_STEPS, subsurfaceAlphaFunc, subsurfaceSigmaFunc, henyeyGreensteinPdfFunc, directionFromCosineFunc, diffusionLengthDwivediFunc, samplePhaseDwivediFunc, evalPhaseDwivediFunc } from '../../nodes/material.wgsl.js';
import { isTerminatingScatterFunc, offsetRayOriginFunc } from '../../nodes/utils.wgsl.js';
import { LIGHT_EPSILON } from '../../nodes/lights.wgsl.js';

// Pure material evaluation and ray generation: terminated slots pull a recycled pixel and emit a
// fresh camera ray; live slots evaluate the surface staged by LogicKernel, sample the bsdf, and
// enqueue the next bounce ray plus the NEE shadow ray toward the light LogicKernel selected.
export class MaterialKernel extends ComputeKernel {

	constructor( ) {

		const params = {
			bvhData: { value: null },
			material: { value: null },

			seed: uniform( 0, 'uint' ),
			targetDimensions: uniform( new Vector2() ),
			maxSamples: uniform( 0, 'uint' ),
			rayCount: uniform( 0, 'uint' ),
			filterGlossy: uniform( 1 ),
			maxTransparentBounces: uniform( 5, 'uint' ),
			maxBounces: uniform( 5, 'uint' ),
			maxSubsurfaceSteps: uniform( SUBSURFACE_MAX_STEPS, 'uint' ),

			sampleCountTarget: textureStore( new StorageTexture( 1, 1 ) ).toReadWrite(),

			rayDataStorage: storage( new StorageBufferAttribute( 1, 1 ), rayDataStruct ),
			rayQueue: storage( new StorageBufferAttribute( 1, 1 ), rayQueueAtomicStruct ),
			shadowRayQueue: storage( new StorageBufferAttribute( 1, 1 ), rayQueueAtomicStruct ),
			pixelQueue: storage( new StorageBufferAttribute( 1, 1 ), pixelQueueStruct ),

			globalId: globalId,
		};

		const getCameraRayFn = proxyFn( 'bvhData.value.fns.getCameraRay', params );
		const sampleTrianglePointFn = proxyFn( 'bvhData.value.fns.sampleTrianglePoint', params );
		const getSurfaceRecordFn = proxyFn( 'bvhData.value.fns.getSurfaceRecord', params );
		const sampleMixFactorFn = proxyFn( 'bvhData.value.fns.sampleMixFactor', params );
		const bsdfSampleFn = proxyFn( 'material.value.bsdfSample', params );
		const bsdfEvalPdfFn = proxyFn( 'material.value.bsdfEvalPdf', params );

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

				globalId: vec3u
			) -> void {

				let rayDataStorage = &${ params.rayDataStorage };
				let rayQueue = &${ params.rayQueue };
				let shadowRayQueue = &${ params.shadowRayQueue };
				let pixelQueue = &${ params.pixelQueue };

				let materials = &${ proxy( 'bvhData.value.storage.materials', params ) };
				let transforms = &${ proxy( 'bvhData.value.storage.transforms', params ) };

				// bound by "rayCount" rather than the pool length. The dispatch rounds up to the
				// workgroup size and those extra slots hold a zeroed pixel index
				let index = globalId.x;
				if ( index >= rayCount ) {

					return;

				}

				let input = rayDataStorage[ index ];
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
					let jitteredUv = uv + ${ rand2 }( ${ RNG_INDEX_RAY_JITTER } ) / vec2f( targetDimensions );
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
					// a camera ray starts OUTSIDE every medium
					rayDataStorage[ index ].insideMaterial = - 1;
					rayDataStorage[ index ].subsurfaceSteps = 0u;
					rayDataStorage[ index ].dispersionWavelength = - mix( ${ DISPERSION_MIN_WAVELENGTH }.0, ${ DISPERSION_MAX_WAVELENGTH }.0, ${ rand1 }( ${ RNG_INDEX_DISPERSION_WAVELENGTH } ) );

					// write the active params & dispatched flag
					textureStore( ${ params.sampleCountTarget }, indexUV, vec4( ${ SAMPLE_ACTIVE_FLAG }u | ${ SAMPLE_DISPATCHED_FLAG }u | samples ) );

				} else {

					// evaluate the surface staged by LogicKernel
					let indexUV = vec2u( input.pixelIndex >> 16, input.pixelIndex & 0xFFFF );
					// the walk step counts like a bounce here: without it every step of a walk would
					// draw the same numbers, and the path would march in a straight line
					${ rngInit }( indexUV, input.seed, input.currentBounce + input.alphaDepth + input.subsurfaceSteps );

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
					var walkTransmittance = vec3f( 1.0 );
					if ( input.insideMaterial >= 0 ) {

						let medium = materials[ u32( input.insideMaterial ) ];

						// a radius near zero is a surface, not a medium: the path crosses straight
						// and the exit below handles it
						if ( max( medium.subsurfaceRadius.r, max( medium.subsurfaceRadius.g, medium.subsurfaceRadius.b ) ) > 1e-4 ) {

							// ── ONE MEAN FREE PATH PER CHANNEL, which is what makes skin skin ──
							//
							// Red travels about ten times deeper than blue through flesh, so a
							// shallow crossing comes out pale and a deep one comes out red: that
							// bleed is the whole reason subsurface exists, and a single averaged
							// radius cannot produce it.
							//
							// A step can only be drawn for ONE channel, so one is picked at random
							// and the result is weighted by the mixture pdf - the standard spectral
							// estimator, and Cycles' bssrdf_channel_pdf is the same idea. The
							// weight is a vec3: the channel that was picked is not the only one
							// that gets carried, it is only the one that chose the distance.
							// ── E IL RAGGIO NON E' IL CAMMINO LIBERO: c'e' una CONVERSIONE ──
							//
							// Quel che l'utente scrive e' il raggio di DIFFUSIONE - quanto lontano
							// la luce riemerge - e il colore e' l'albedo di SUPERFICIE. Il cammino
							// vuole altre due cose: il libero cammino medio e l'albedo di singolo
							// scattering, che sono molto piu' corti e molto piu' alti.
							//
							// La conversione sta in subsurfaceAlpha / subsurfaceSigma, ed e' quella
							// di Cycles LETTA NEL SORGENTE: la prima versione era scritta a memoria
							// e sbagliava sia l'albedo (ignorava l'anisotropia, e aveva un pavimento
							// che di la' non c'e') sia l'estinzione (un fattore di accorciamento che
							// appartiene al profilo di diffusione, non al cammino).
							let alpha = ${ subsurfaceAlphaFunc }( medium.color, medium.subsurfaceAnisotropy );
							let sigma = ${ subsurfaceSigmaFunc }( medium.subsurfaceRadius );
							// ── IL CANALE SI SORTEGGIA SU alpha * throughput ──
							//
							// E' volume_sample_channel di Cycles, letto nel sorgente: il canale che
							// il cammino porta gia' si pesca piu' spesso, e il peso resta vicino a
							// uno. Provata anche la strada opposta — una distribuzione sola col
							// sigma medio pesato — ed e' PEGGIO, misurato sulla scena di un utente:
							// piu' polvere, non meno.
							var channelP = vec3f( 1.0 / 3.0 );
							let carried = max( input.throughputColor * alpha, vec3f( 0.0 ) );
							let carriedSum = carried.r + carried.g + carried.b;
							if ( carriedSum > 1e-9 ) { channelP = carried / carriedSum; }

							let pick = ${ rand1 }( ${ RNG_INDEX_SUBSURFACE_WALK } + 1 );
							var channel = 2u;
							if ( pick < channelP.r ) { channel = 0u; }
							else if ( pick < channelP.r + channelP.g ) { channel = 1u; }

							// ── LA DISTANZA SI STIRA SE LA DIREZIONE ERA GUIDATA ──
							//
							// Lo stiramento e' la meta' della guida di Dwivedi, e le due sono UN
							// meccanismo solo: pendere verso l'uscita senza allungare il passo che
							// ci va peggiora invece di migliorare (misurato, 19,98 -> 23,43). Lo
							// stiramento arriva col segmento perche' dipende dalla sua direzione,
							// scelta un passo fa.
							let guidedStretch = select( 1.0, input.subsurfaceStretch, input.subsurfaceGuided == 1u );
							let stepDist = - log( max( 1.0 - ${ rand1 }( ${ RNG_INDEX_SUBSURFACE_WALK } ), 1e-9 ) )
								/ max( sigma[ channel ] * guidedStretch, 1e-6 );

							// ── E LA PDF DEL SEGMENTO E' LA MISCELA DELLE DUE STRATEGIE ──
							//
							// Questa direzione poteva venire da entrambe, quindi entrambe le pdf
							// contano — la euristica di bilancio, come per i canali. Sul PRIMO
							// segmento no: li' la direzione e' quella che la superficie ha
							// rifratto, non una che la fase ha estratto, quindi e' classica.
							// ── QUANTO SI GUIDA, E PERCHE' NON COME CYCLES ──
							//
							// Di la' la frazione e' 1 - max(0.5, |g|^0.125), cioe' meta' dei
							// campioni al piu'. Quel tetto serve a loro perche' guidano anche
							// ALL'INDIETRO, verso l'interfaccia opposta, e le due meta' si dividono
							// il budget; noi abbiamo solo quella in avanti, che vuole un raggio
							// tracciato per sapere dov'e' l'altra parete.
							//
							// Con la sola meta' in avanti la forma giusta e' un'altra, e sta in una
							// tabella misurata su un cubo di 2 m con cammino libero di 1 cm (rumore
							// fra pixel vicini, denoise spento):
							//
							//   g = 0     7,68 guidando  contro  10,43  -> il 26% in meno
							//   g = 0,4  13,10           contro  12,99  -> pari
							//   g = 0,8  23,95           contro  21,65  -> peggio, E la media
							//                                              scende da 134 a 131
							//
							// Dwivedi e' derivato per mezzi ISOTROPI: piu' il mezzo diffonde in
							// avanti, meno la guida somiglia alla fase, e le due strategie finiscono
							// per discordare invece che aiutarsi. Quindi si spegne dove smette di
							// pagare, e non si accende «un po'» dove non serve. Il nostro default e'
							// zero, quindi cera, marmo e latte prendono il guadagno; la pelle a 0,8
							// resta esattamente com'era.
							// ── E SOLO SE IL MEZZO E' OTTICAMENTE SPESSO ──
							//
							// La guida in avanti riporta i cammini verso la parete da cui sono
							// entrati. In un mezzo profondo e' quel che serve — escono prima invece
							// di vagare. In un mezzo SOTTILE toglie la traversata: misurato su un
							// pannello di 12 cm con cammino libero di 2 cm, cioe' sei cammini liberi
							// di spessore, la faccia in ombra passa da 3,7 a 2,7. Il 27% di luce che
							// non arriva piu' dall'altra parte.
							//
							// Di la' il rimedio e' la guida ALL'INDIETRO, verso l'interfaccia
							// opposta, che noi non abbiamo. Quel che abbiamo e' la sua distanza, e
							// gratis: il primo segmento attraversa l'oggetto per intero. Venti
							// cammini liberi e' la soglia — sotto, il pannello del banco (6) resta
							// com'era; sopra, il cubo dell'utente (200) prende il guadagno.
							let opticalDepth = input.subsurfaceOpposite * ( sigma.r + sigma.g + sigma.b ) / 3.0;
							let anisotropy = medium.subsurfaceAnisotropy;
							let guidedFraction = select( 0.0, max( 0.0, 1.0 - abs( anisotropy ) / 0.4 ), opticalDepth > 20.0 );
							let reachedWall = stepDist >= input.dist;
							let travelled = min( stepDist, input.dist );
							let transmittance = exp( - sigma * travelled );

							var segmentPdf = select( sigma * transmittance, transmittance, reachedWall );
							if ( input.subsurfaceSteps > 0u ) {

								let stretched = sigma * input.subsurfaceStretch;
								let stretchedT = exp( - stretched * travelled );
								let guidedPdf = input.subsurfacePdfFactor * select( stretched * stretchedT, stretchedT, reachedWall );
								segmentPdf = mix( segmentPdf, guidedPdf, guidedFraction );

							}

							// sigma_s * T / pdf diffondendo, T / pdf arrivando alla parete. Il
							// valore della fase non e' al numeratore perche' e' gia' stato
							// semplificato dentro il fattore di pdf.
							let segmentWeight = select( alpha * sigma * transmittance, transmittance, reachedWall )
								/ max( dot( channelP, segmentPdf ), 1e-9 );

							if ( ! reachedWall ) {

								// out of steps: the path is dropped where it stands rather than let out
								// somewhere arbitrary. It loses its energy, and that is the truncation.
								if ( input.subsurfaceSteps >= maxSubsurfaceSteps ) {

									rayDataStorage[ index ].throughputColor = vec3f( 0.0 );
									rayDataStorage[ index ].emission = vec3f( 0.0 );
									rayDataStorage[ index ].lightPdf = 0.0;
									rayDataStorage[ index ].shadowRayIntersectionIndex = - 1;
									return;

								}

								let scatterPoint = input.origin + input.direction * travelled;

								// ── DWIVEDI: la direzione si GUIDA verso l'uscita ──
								//
								// Un cammino in un mezzo denso vaga, e quasi tutti i suoi passi non
								// vanno verso l'uscita: e' li' che nasce il rumore. La distribuzione
								// di Dwivedi pende verso l'interfaccia da cui il cammino e' entrato.
								// Con un mezzo che diffonde in avanti il cammino ci va gia' da se' e
								// la guida si spegne quasi del tutto: la pelle a 0,8 la usa il 3%
								// delle volte.
								//
								// La frazione e' quella di Cycles SENZA il suo tetto di 0,5, e il
								// tetto mancante e' una divergenza MISURATA: di la' meta' dei
								// campioni al massimo sono guidati perche' guidano anche
								// ALL'INDIETRO, verso l'interfaccia opposta, e le due si dividono il
								// budget. Noi abbiamo solo la meta' in avanti, e mescolarla a meta'
								// col classico non rende niente — le due strategie discordano, e la
								// discordanza costa quanto la guida guadagna. Misurato su un cubo di
								// 2 m con cammino libero di 1 cm, isotropo: 10,43 senza guida,
								// 10,52 alla meta' di Cycles, 7,95 guidando sempre.
								let albedoMax = max( alpha.r, max( alpha.g, alpha.b ) );
								let diffusionLength = ${ diffusionLengthDwivediFunc }( albedoMax );
								let phaseLog = log( ( diffusionLength + 1.0 ) / max( diffusionLength - 1.0, 1e-6 ) );
								let guideNormal = input.subsurfaceNormal;
								let useGuided = ${ rand1 }( ${ RNG_INDEX_SUBSURFACE_WALK } + 4 ) < guidedFraction;

								var nextDirection: vec3f;
								if ( useGuided ) {

									let guidedCos = ${ samplePhaseDwivediFunc }(
										diffusionLength, phaseLog, ${ rand1 }( ${ RNG_INDEX_SUBSURFACE_WALK } + 5 ),
									);
									nextDirection = ${ directionFromCosineFunc }(
										guideNormal, guidedCos, ${ rand1 }( ${ RNG_INDEX_SUBSURFACE_WALK } + 6 ),
									);

								} else {

									nextDirection = ${ sampleHenyeyGreensteinFunc }(
										input.direction, anisotropy,
										${ rand2 }( ${ RNG_INDEX_SUBSURFACE_WALK } + 2 ),
									);

								}

								// lo stiramento si tiene per TUTTI i segmenti, perche' la pdf guidata
								// e' quella che la guida AVREBBE avuto, chiunque l'abbia poi estratta;
								// si applica al campionamento solo se l'ha estratta lei. Applicarlo
								// sempre rende la miscela una bugia e l'immagine esce il 5% piu'
								// scura — misurato, ed e' il difetto che questa riga nomina.
								let cosGuide = dot( nextDirection, guideNormal );
								let phaseValue = max( ${ henyeyGreensteinPdfFunc }( dot( input.direction, nextDirection ), anisotropy ), 1e-9 );
								let nextPdfFactor = ${ evalPhaseDwivediFunc }( diffusionLength, phaseLog, cosGuide ) / phaseValue;
								let nextStretch = max( 1.0 - cosGuide / diffusionLength, 1e-3 );

								let walkIndex = atomicAdd( &rayQueue.length, 1u );
								rayQueue.elements[ walkIndex ].origin = scatterPoint;
								rayQueue.elements[ walkIndex ].direction = nextDirection;
								rayQueue.elements[ walkIndex ].pixelIndex = input.pixelIndex;
								rayQueue.elements[ walkIndex ].currentBounce = input.currentBounce;
								rayQueue.elements[ walkIndex ].seed = input.seed;
								rayQueue.elements[ walkIndex ].alphaDepth = input.alphaDepth;
								// the direction changed, so whatever budget the old segment had does not
								// apply: zero traces unbounded, the way an ordinary scatter does
								rayQueue.elements[ walkIndex ].maxDist = 0.0;

								// the surface is never reached, so nothing of it is staged: no emission
								// and no NEE. "scatterColor" carries the albedo scaled by the pdf that
								// LogicKernel will divide out, which leaves throughput *= albedo - the
								// same trick the alpha pass through uses to leave it untouched.
								rayDataStorage[ index ].subsurfaceSteps = input.subsurfaceSteps + 1u;
								rayDataStorage[ index ].subsurfaceStretch = nextStretch;
								rayDataStorage[ index ].subsurfacePdfFactor = nextPdfFactor;
								rayDataStorage[ index ].subsurfaceGuided = select( 0u, 1u, useGuided );
								// il primo segmento ha appena misurato lo spessore: la sua distanza,
								// proiettata sulla normale d'ingresso. Dopo si tiene quella.
								rayDataStorage[ index ].subsurfaceOpposite = select(
									input.subsurfaceOpposite,
									input.dist * max( dot( input.direction, - guideNormal ), 0.0 ),
									input.subsurfaceSteps == 0u,
								);
								rayDataStorage[ index ].emission = vec3f( 0.0 );
								rayDataStorage[ index ].scatterColor = segmentWeight * input.scatterPdf;
								rayDataStorage[ index ].lightPdf = 0.0;
								rayDataStorage[ index ].origin = scatterPoint;
								rayDataStorage[ index ].direction = nextDirection;
								rayDataStorage[ index ].rayIntersectionIndex = i32( walkIndex );
								rayDataStorage[ index ].shadowRayIntersectionIndex = - 1;
								return;

							}

							// il passo ha superato la parete: il cammino attraversa, attenuato da
							// quel che il mezzo ha assorbito per strada. E' QUI che nasce il bordo
							// rosso — una traversata lunga tiene il rosso e perde il blu.
							walkTransmittance = segmentWeight;

						}

					}

					let objectInfo = transforms[ u32( input.objectIndex ) ];
					var materialInfo = materials[ objectInfo.materialIndex ];

					// The surface point is sampled HERE and no longer below: the mix can read
					// a MASK, and a mask wants the uv of the hit.
					var vertexData = ${ sampleTrianglePointFn }( input.barycoord, input.indices );
					vertexData.normal = normalize( transpose( objectInfo.inverseMatrixWorld ) * vertexData.normal );
					vertexData.tangent = vec4f( ( objectInfo.matrixWorld * vec4f( vertexData.tangent.xyz, 0.0 ) ).xyz, vertexData.tangent.w );

					// ── MIX SHADER ──
					//
					// One branch is taken at random, in proportion to its weight, instead of
					// evaluating both and blending: it is what Cycles does
					// (surface_shader_bsdf_bssrdf_pick). The estimator stays unbiased because
					// the branch is chosen with exactly its weight, a hit still costs a single
					// BSDF, and the choice made here also governs the NEE shadow ray below —
					// they share this record, which is the property a per-lobe blend would lose.
					//
					// A mix of N shaders arrives FLATTENED into a chain: at each link the
					// record is kept with probability 1 - mixWeight, or the chain moves on to
					// the next leaf. The weights are conditional, so the product telescopes
					// back to the probability each leaf had in the tree.
					//
					// Every link draws a DIFFERENT dimension: these are dimensions of one
					// sequence, so reusing an index would hand the chain the same number twice
					// and pile the probability onto the first leaves. The bound is the number
					// of reserved dimensions, and it also keeps wavefront lanes from diverging
					// on depth.
					for ( var mixStep = 0u; mixStep < ${ RNG_INDEX_MIX_SHADER_COUNT }u; mixStep ++ ) {

						// the factor is per hit: a wired Fac is a mask, and then it is the
						// texture that decides the branch, pixel by pixel
						let mixFac = ${ sampleMixFactorFn }( materialInfo, vertexData );
						if ( mixFac <= 0.0 ) { break; }
						if ( ${ rand1 }( ${ RNG_INDEX_MIX_SHADER } + mixStep ) >= mixFac ) { break; }
						materialInfo = materials[ u32( materialInfo.mixIndex ) ];

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

					vertexData.position = objectInfo.matrixWorld * vertexData.position;

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
					if ( input.insideMaterial >= 0 ) {

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

					var surface = ${ getSurfaceRecordFn }( materialInfo, vertexData, hitSide, hitNormal, view, blurRoughness );

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
					if ( input.insideMaterial >= 0 ) {

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

					// attenuate the light transmitted through the volume when exiting a backface so
					// the surface's emission and NEE resolve against the attenuated throughput
					if ( input.side < 0.0 && materialInfo.transmission > 0.0 ) {

						throughputColor *= ${ transmissionAttenuationFunc }( input.dist, materialInfo.attenuationColor, materialInfo.attenuationDistance );
						rayDataStorage[ index ].throughputColor = throughputColor;

					}

					// sample the next bounce direction and stage the scatter state for LogicKernel
					var scatterRec = ${ bsdfSampleFn }( view, surface );

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
					var insideNext = input.insideMaterial;
					// a hit that goes in has PICKED the bssrdf closure, so the surface one is not
					// there to be lit: see the NEE block at the bottom
					var enteredSubsurface = false;
					if ( input.insideMaterial >= 0 ) {

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

						// il cammino si ricorda da dove e' entrato: e' il verso in cui la guida
						// pende. Il PRIMO segmento pero' e' classico — la sua direzione e' quella
						// specchiata qui sopra, non una che la fase ha estratto.
						rayDataStorage[ index ].subsurfaceNormal = faceNormal;
						rayDataStorage[ index ].subsurfaceSteps = 0u;
						rayDataStorage[ index ].subsurfaceStretch = 1.0;
						rayDataStorage[ index ].subsurfacePdfFactor = 0.0;
						rayDataStorage[ index ].subsurfaceGuided = 0u;
						rayDataStorage[ index ].subsurfaceOpposite = 0.0;

					}
					rayDataStorage[ index ].insideMaterial = insideNext;

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
					rayDataStorage[ index ].emission = surface.emission;
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

						let evalRec = ${ bsdfEvalPdfFn }( view, input.lightDirection, surface );
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

	}

}
