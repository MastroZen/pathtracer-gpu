import { StorageBufferAttribute } from 'three/webgpu';
import { ComputeKernel } from '../ComputeKernel.js';
import { storage, globalId, uniform } from 'three/tsl';
import { proxy, proxyFn, wgslTagFn } from 'three-mesh-bvh/webgpu';
import { rngInit } from '../../nodes/random.wgsl.js';
import { rayQueueStruct, intersectionResultStruct } from './structs.js';
import { hairQueryFn, hairSegmentObjectFn, EMPTY_HAIR_DATA } from '../../nodes/hair.wgsl.js';
import { offsetRayOriginFunc } from '../../nodes/utils.wgsl.js';
import { chainPassThroughFunc } from '../../nodes/material.wgsl.js';
import { svmSurfacePointFunc, svmWeightFunc, svmLinkWeightsFunc } from '../../nodes/svmSurface.wgsl.js';
import { mediumPointFunc } from '../../nodes/medium.wgsl.js';
import {
	mediumStackCoefficientsFunc, mediumStackEnterFn, mediumStackExitFn, mediumStackMajorantFunc, mediumStackPointFunc,
	mediumStackTransmittanceFunc,
} from '../../nodes/mediumStack.wgsl.js';

// How many null boundaries of participating media a shadow ray crosses before it gives up and
// counts as unoccluded with the transmittance gathered so far: entering and leaving a fog box is
// two. Cycles bounds the same walk by its transparent bounce limit.
//
// A UNIFORM and not a constant in the loop: FXC, the HLSL compiler under D3D, may unroll a loop
// with a constant count, and this one holds the whole BVH traversal. With the constant, the second
// tracer of the Render view once failed with E_OUTOFMEMORY creating its pipelines - but on a VM
// whose GPU also loses the device with the unchanged kernel, so the cause is likely, not proven.
const MEDIUM_SHADOW_STEPS = 16;

// Pure BVH traversal over the queued shadow rays. Uses the same first-hit traversal as the bounce
// rays ( no dedicated any-hit traversal exists yet ); LogicKernel decides occlusion by comparing the
// hit distance against the light distance.
export class TraceShadowRayKernel extends ComputeKernel {

	// "registers" is the register file of the programs a shadow ray runs, and zero when the scene has
	// none: then the interpreter is not compiled in at all, since every scene would pay its compile
	// time for a feature it does not use. "shadowMixes" says whether one of them is the weights of a
	// chain that lets light through - a wired Mix Shader Fac over a Transparent BSDF, the clouds
	constructor( registers = 0, shadowMixes = false ) {

		const params = {
			bvhData: { value: null },

			shadowRayQueue: storage( new StorageBufferAttribute( 1, 1 ), rayQueueStruct ),
			shadowRayIntersectionsStorage: storage( new StorageBufferAttribute( 1, 1 ), intersectionResultStruct ),

			// lo stesso pacchetto del kernel di tracciamento: senza, i peli non si fanno
			// ombra fra loro e una pelliccia diventa una nuvola uniforme — l'ombra
			// portata e' meta' di cio' che la fa leggere come volume
			hairData: storage( new StorageBufferAttribute( EMPTY_HAIR_DATA, 1 ), 'uint' ).toReadOnly(),

			maxMediumCrossings: uniform( MEDIUM_SHADOW_STEPS, 'uint' ),

			globalId: globalId,
		};

		const raycastOutput = proxy( 'bvhData.value.fns.raycastFirstHit.outputType', params );
		const raycastFirstHitFn = proxy( 'bvhData.value.fns.raycastFirstHit', params );
		// already bound by the raycast, which reads them for the alpha test: no new binding
		const materialsBuffer = proxy( 'bvhData.value.storage.materials', params );
		const transformsBuffer = proxy( 'bvhData.value.storage.transforms', params );
		// the transmittance of a stretch through the stack: without a heterogeneous medium in the
		// scene only the closed form, and the interpreter stays out of this kernel
		const coefficientsFn = mediumStackCoefficientsFunc( materialsBuffer );
		const chainPassThroughFn = chainPassThroughFunc( materialsBuffer );
		// ── THE WEIGHTS OF A CHAIN, run where the shadow ray crosses it ──
		//
		// A wired Fac decides per point how much light passes: run at the crossing, as SvmKernel runs it for a
		// path, on the same surface point (nodes/svmSurface.wgsl.js). Without it the chain weighed a half
		// everywhere, and a sphere of clouds dimmed the whole planet evenly instead of casting its shadows.
		// It is a second call site of the interpreter beside the density's, and FXC pays per call site: only
		// a scene that has such a chain compiles it
		let crossingWeights = '';
		if ( shadowMixes && registers > 0 ) {

			const svmRunFn = proxyFn( `bvhData.value.fns.svmRun${ registers }`, params );
			const surfacePointFn = svmSurfacePointFunc(
				proxy( 'bvhData.value.storage.attributes', params ), transformsBuffer,
				proxyFn( 'bvhData.value.fns.sampleTrianglePoint', params ),
				proxyFn( 'bvhData.value.fns.getUvFromChannel', params ),
				proxyFn( 'bvhData.value.fns.getGenerated', params ),
			);
			const linkWeightsFn = svmLinkWeightsFunc( svmWeightFunc( registers ), registers );
			crossingWeights = wgslTagFn/* wgsl */`
				fn crossingWeights( materialIndex: u32, hit: ${ raycastOutput }, direction: vec3f, first: ptr<function, vec4f>, second: ptr<function, vec4f> ) -> bool {

					let material = ${ materialsBuffer }[ materialIndex ];
					if ( ( material.svmResolve & 2u ) == 0u ) { return false; }
					var generated: mat3x3f;
					var objectPos: mat3x3f;
					var world: mat3x3f;
					var normal: vec3f;
					let uv = ${ surfacePointFn }( hit.indices.xyz, hit.barycoord.xyz, u32( hit.objectIndex ), direction, material.flatShading != 0,
						&generated, &objectPos, &world, &normal );
					var regs: array<vec4f, ${ registers }>;
					_ = ${ svmRunFn }( material.svmMixCode, material.svmMixCount, material.svmMixConsts, uv, generated, objectPos, normal, world, hit.indices.xyz, hit.barycoord.xyz, &regs );
					*first = ${ linkWeightsFn }( &regs, material.svmMixOutputs0, material.svmMixOutputs1, second );
					return true;

				}
			`;

		}
		let transmittanceFn = wgslTagFn/* wgsl */`
			fn mediumStackTransmittance( stack: vec4u, origin: vec3f, direction: vec3f, len: f32, rng: ptr<function, u32> ) -> vec3f {
				if ( stack.x == 0xffffffffu ) { return vec3f( 1.0 ); }
				return exp( - ${ coefficientsFn }( stack )[ 1 ] * len );
			}
		`;
		if ( registers > 0 ) {

			const svmRunFn = proxyFn( `bvhData.value.fns.svmRun${ registers }`, params );
			const gridFn = proxyFn( 'bvhData.value.fns.mediumGrid', params );
			const pointFn = mediumPointFunc( materialsBuffer, transformsBuffer, svmRunFn, registers, gridFn );
			transmittanceFn = mediumStackTransmittanceFunc( materialsBuffer, coefficientsFn,
				mediumStackPointFunc( materialsBuffer, pointFn ), mediumStackMajorantFunc( materialsBuffer ) );

		}

		const fn = wgslTagFn /* wgsl */`

			fn compute( maxMediumCrossings: u32, globalId: vec3u ) -> void {

				let shadowRayQueue = &${ params.shadowRayQueue };
				let shadowRayIntersectionsStorage = &${ params.shadowRayIntersectionsStorage };

				let index = globalId.x;
				if ( index >= shadowRayQueue.length ) {

					return;

				}

				// A whole element is read from the BUFFER, never through the pointer alias
				// above: WebKit packs every struct that holds a vec3 and does not unpack a
				// load made through a let pointer, so Safari refused this kernel with
				// "no viable conversion from __typeN_Packed". Field reads and writes
				// through the alias compile, and stay as they are.
				let queuedRay = ${ params.shadowRayQueue }.elements[ index ];
				let indexUV = vec2u( queuedRay.pixelIndex >> 16, queuedRay.pixelIndex & 0xFFFF );
				${ rngInit }( indexUV, queuedRay.seed, queuedRay.currentBounce + queuedRay.alphaDepth );

				// ── THE SHADOW RAY CROSSES PARTICIPATING MEDIA, as in Cycles (shade_shadow.h) ──
				//
				// A surface that is only a medium does not occlude: the ray goes through it and
				// carries on, and the medium it is inside attenuates every stretch it crosses by
				// exp( - sigma_t * length ). What reaches LogicKernel is the TRANSMITTANCE in the
				// barycoord field, one when there is no medium, and an occluder as before.
				var origin = queuedRay.origin;
				let direction = queuedRay.direction;
				let bounded = queuedRay.maxDist > 0.0;
				var remaining = queuedRay.maxDist;
				var stack = queuedRay.mediumStack;
				var transmittance = vec3f( 1.0 );
				var occluder = - 1;
				var occluderDist = 0.0;
				var travelled = 0.0;
				var rng = queuedRay.seed * 747796405u + queuedRay.pixelIndex * 2891336453u
					+ ( queuedRay.currentBounce * 128u + queuedRay.alphaDepth ) * 277803737u + 0x68e31da4u;
				for ( var crossing = 0u; crossing < maxMediumCrossings; crossing ++ ) {

					let ray = Ray( origin, direction, select( 0.0, remaining, bounded ) );
					var hitResult: ${ raycastOutput };
					let hitTriangle = ${ raycastFirstHitFn }( ray, &hitResult );

					// qui basta il PIU' VICINO: chi decide l'occlusione confronta la distanza
					// con quella della luce, e non guarda cosa c'era
					var limit = 1e30;
					if ( bounded ) { limit = remaining; }
					if ( hitTriangle ) { limit = min( limit, hitResult.dist ); }
					let hair = ${ hairQueryFn }( &${ params.hairData }, origin, direction, limit );
					if ( hair.didHit ) {

						occluder = i32( ${ hairSegmentObjectFn }( &${ params.hairData }, hair.segment ) );
						occluderDist = travelled + hair.dist;
						break;

					}

					// the stretch up to the next boundary, or to the light, through every medium of
					// the stack: in closed form when they are all homogeneous, by ratio tracking when not
					if ( stack.x != 0xffffffffu ) {

						let stretch = select( select( 1e30, remaining, bounded ), hitResult.dist, hitTriangle );
						transmittance *= ${ transmittanceFn }( stack, origin, direction, stretch, &rng );

					}

					if ( ! hitTriangle ) { break; }

					let materialIndex = ${ transformsBuffer }[ u32( hitResult.objectIndex ) ].materialIndex;
					if ( ( ${ materialsBuffer }[ materialIndex ].mediumFlags & 1u ) == 0u ) {

						// ── A SURFACE THAT LETS LIGHT THROUGH: a Transparent BSDF, or an alpha ──
						//
						// Cycles carries a shadow ray through every transparent closure, weighted by its share
						// of the surface (shade_shadow.h, the transparent shadow). Here the share is the
						// expected one of the chain (chainPassThrough), and the crossing counts against the
						// same limit as a medium's
						var linkWeights0 = vec4f( 0.0 );
						var linkWeights1 = vec4f( 0.0 );
						let live = ${ crossingWeights || 'false' }${ crossingWeights ? '( materialIndex, hitResult, direction, &linkWeights0, &linkWeights1 )' : '' };
						let through = ${ chainPassThroughFn }( materialIndex, live, linkWeights0, linkWeights1 );
						if ( all( through <= vec3f( 0.0 ) ) ) {

							occluder = i32( hitResult.objectIndex );
							occluderDist = travelled + hitResult.dist;
							break;

						}
						transmittance *= through;

					} else {

						// a null boundary: its medium enters the stack from the front, leaves it from the back
						let word = ( materialIndex & 0xffffu ) | ( ( u32( hitResult.objectIndex ) & 0xffffu ) << 16u );
						if ( hitResult.side > 0.0 ) {

							stack = ${ mediumStackEnterFn }( stack, word );

						} else {

							stack = ${ mediumStackExitFn }( stack, word );

						}

					}

					origin = ${ offsetRayOriginFunc }( origin + direction * hitResult.dist, direction, hitResult.normal.xyz );
					travelled += hitResult.dist;
					if ( bounded ) { remaining = max( remaining - hitResult.dist, 1e-6 ); }

				}

				shadowRayIntersectionsStorage[ index ].objectIndex = occluder;
				shadowRayIntersectionsStorage[ index ].dist = occluderDist;
				shadowRayIntersectionsStorage[ index ].barycoord = transmittance;

			}
		`;

		super( fn( params ) );

		this.registers = registers;
		this.shadowMixes = shadowMixes;
		this.defineUniformAccessors( params );

	}

}
