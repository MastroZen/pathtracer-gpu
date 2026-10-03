import { StorageBufferAttribute } from 'three/webgpu';
import { ComputeKernel } from '../ComputeKernel.js';
import { storage, globalId, uniform } from 'three/tsl';
import { proxy, proxyFn, wgslTagFn } from 'three-mesh-bvh/webgpu';
import { rngInit } from '../../nodes/random.wgsl.js';
import { rayQueueStruct, intersectionResultStruct } from './structs.js';
import { hairQueryFn, hairSegmentObjectFn, EMPTY_HAIR_DATA } from '../../nodes/hair.wgsl.js';
import { offsetRayOriginFunc } from '../../nodes/utils.wgsl.js';
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

	// "registers" is the register file of the density programs of the scene, and zero when it has
	// no heterogeneous medium: then the interpreter is not compiled in at all, since every scene
	// would pay its compile time for a feature it does not use
	constructor( registers = 0 ) {

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
		let transmittanceFn = wgslTagFn/* wgsl */`
			fn mediumStackTransmittance( stack: vec4u, origin: vec3f, direction: vec3f, len: f32, rng: ptr<function, u32> ) -> vec3f {
				if ( stack.x == 0xffffffffu ) { return vec3f( 1.0 ); }
				return exp( - ${ coefficientsFn }( stack )[ 1 ] * len );
			}
		`;
		if ( registers > 0 ) {

			const svmRunFn = proxyFn( `bvhData.value.fns.svmRun${ registers }`, params );
			const pointFn = mediumPointFunc( materialsBuffer, transformsBuffer, svmRunFn, registers );
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

						occluder = i32( hitResult.objectIndex );
						occluderDist = travelled + hitResult.dist;
						break;

					}

					// a null boundary: its medium enters the stack from the front, leaves it from the back
					let word = ( materialIndex & 0xffffu ) | ( ( u32( hitResult.objectIndex ) & 0xffffu ) << 16u );
					if ( hitResult.side > 0.0 ) {

						stack = ${ mediumStackEnterFn }( stack, word );

					} else {

						stack = ${ mediumStackExitFn }( stack, word );

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
		this.defineUniformAccessors( params );

	}

}
