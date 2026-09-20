import { StorageBufferAttribute } from 'three/webgpu';
import { ComputeKernel } from '../ComputeKernel.js';
import { storage, globalId } from 'three/tsl';
import { proxy, wgslTagFn } from 'three-mesh-bvh/webgpu';
import { rngInit } from '../../nodes/random.wgsl.js';
import { rayQueueStruct, intersectionResultStruct } from './structs.js';
import { hairQueryFn, hairSegmentObjectFn, EMPTY_HAIR_DATA } from '../../nodes/hair.wgsl.js';

// Pure BVH traversal over the queued shadow rays. Uses the same first-hit traversal as the bounce
// rays ( no dedicated any-hit traversal exists yet ); LogicKernel decides occlusion by comparing the
// hit distance against the light distance.
export class TraceShadowRayKernel extends ComputeKernel {

	constructor( ) {

		const params = {
			bvhData: { value: null },

			shadowRayQueue: storage( new StorageBufferAttribute( 1, 1 ), rayQueueStruct ),
			shadowRayIntersectionsStorage: storage( new StorageBufferAttribute( 1, 1 ), intersectionResultStruct ),

			// lo stesso pacchetto del kernel di tracciamento: senza, i peli non si fanno
			// ombra fra loro e una pelliccia diventa una nuvola uniforme — l'ombra
			// portata e' meta' di cio' che la fa leggere come volume
			hairData: storage( new StorageBufferAttribute( EMPTY_HAIR_DATA, 1 ), 'uint' ).toReadOnly(),

			globalId: globalId,
		};

		const raycastOutput = proxy( 'bvhData.value.fns.raycastFirstHit.outputType', params );
		const raycastFirstHitFn = proxy( 'bvhData.value.fns.raycastFirstHit', params );

		const fn = wgslTagFn /* wgsl */`

			fn compute( globalId: vec3u ) -> void {

				let shadowRayQueue = &${ params.shadowRayQueue };
				let shadowRayIntersectionsStorage = &${ params.shadowRayIntersectionsStorage };

				let index = globalId.x;
				if ( index >= shadowRayQueue.length ) {

					return;

				}

				let queuedRay = shadowRayQueue.elements[ index ];
				let indexUV = vec2u( queuedRay.pixelIndex >> 16, queuedRay.pixelIndex & 0xFFFF );
				${ rngInit }( indexUV, queuedRay.seed, queuedRay.currentBounce + queuedRay.alphaDepth );

				let ray = Ray( queuedRay.origin, queuedRay.direction, queuedRay.maxDist );
				var hitResult: ${ raycastOutput };
				let hitTriangle = ${ raycastFirstHitFn }( ray, &hitResult );

				// qui basta il PIU' VICINO: chi decide l'occlusione confronta la distanza
				// con quella della luce, e non guarda cosa c'era
				var limit = 1e30;
				if ( queuedRay.maxDist > 0.0 ) { limit = queuedRay.maxDist; }
				if ( hitTriangle ) { limit = min( limit, hitResult.dist ); }
				let hair = ${ hairQueryFn }( &${ params.hairData }, ray.origin, ray.direction, limit );

				if ( hair.didHit ) {

					shadowRayIntersectionsStorage[ index ].objectIndex = i32( ${ hairSegmentObjectFn }( &${ params.hairData }, hair.segment ) );
					shadowRayIntersectionsStorage[ index ].dist = hair.dist;

				} else if ( hitTriangle ) {

					shadowRayIntersectionsStorage[ index ].objectIndex = i32( hitResult.objectIndex );
					shadowRayIntersectionsStorage[ index ].dist = hitResult.dist;

				} else {

					shadowRayIntersectionsStorage[ index ].objectIndex = - 1;

				}

			}
		`;

		super( fn( params ) );

		this.defineUniformAccessors( params );

	}

}
