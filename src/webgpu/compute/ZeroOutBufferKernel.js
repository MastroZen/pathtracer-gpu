import { IndirectStorageBufferAttribute } from 'three/webgpu';
import { ComputeKernel } from './ComputeKernel.js';
import { storage, globalId, uniform } from 'three/tsl';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';

export class ZeroOutBufferKernel extends ComputeKernel {

	constructor( options = {} ) {

		const {
			type = 'u32',
		} = options;

		const params = {
			globalId: globalId,
			outputTarget: storage( new IndirectStorageBufferAttribute( 1, 1 ), type ),
			// the first element zeroed: the queue counters share one buffer and reset at different steps
			offset: uniform( 0, 'uint' ),
		};

		const fn = wgslTagFn/* wgsl */`
			fn compute( globalId: vec3u ) -> void {

				${ params.outputTarget }[ ${ params.offset } + globalId.x ] = 0;

			}
		`;

		super( fn( params ) );

		this.defineUniformAccessors( {
			target: params.outputTarget,
			offset: params.offset,
		} );

	}

}
