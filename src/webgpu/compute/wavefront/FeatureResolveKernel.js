import { Matrix4, StorageBufferAttribute, StorageTexture } from 'three/webgpu';
import { ComputeKernel } from '../ComputeKernel.js';
import { storage, textureStore, uniform, globalId } from 'three/tsl';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';

// The guiding passes of the denoiser out of the features the paths gathered (LogicKernel), in the form
// the OIDN filter reads (OIDNDenoiser.denoise): the albedo clamped to [0, 1], and the normal in VIEW
// space mapped to [0, 1] so a normal facing the camera is (0.5, 0.5, 1). The view space is the one of
// the rasterized buffers this replaces; a pixel no path wrote a normal into stays at the zero vector,
// which is what OIDN expects of the background.
export class FeatureResolveKernel extends ComputeKernel {

	constructor() {

		const params = {
			globalId: globalId,
			featureStorage: storage( new StorageBufferAttribute( 2, 4 ), 'vec4f' ).toReadOnly(),
			albedoTarget: textureStore( new StorageTexture( 1, 1 ) ).toWriteOnly(),
			normalTarget: textureStore( new StorageTexture( 1, 1 ) ).toWriteOnly(),
			viewMatrix: uniform( new Matrix4() ),
		};

		const fn = wgslTagFn/* wgsl */`
			fn compute( viewMatrix: mat4x4f, globalId: vec3u ) -> void {

				let size = textureDimensions( ${ params.albedoTarget } );
				if ( globalId.x >= size.x || globalId.y >= size.y ) {

					return;

				}

				let featureIndex = ( globalId.y * size.x + globalId.x ) * 2u;
				let albedo = ${ params.featureStorage }[ featureIndex ].xyz;
				let normal = ( viewMatrix * vec4f( ${ params.featureStorage }[ featureIndex + 1u ].xyz, 0.0 ) ).xyz;

				textureStore( ${ params.albedoTarget }, globalId.xy, vec4f( clamp( albedo, vec3f( 0.0 ), vec3f( 1.0 ) ), 1.0 ) );
				textureStore( ${ params.normalTarget }, globalId.xy, vec4f( clamp( normal * 0.5 + 0.5, vec3f( 0.0 ), vec3f( 1.0 ) ), 1.0 ) );

			}
		`;

		super( fn( params ) );

		this.defineUniformAccessors( params );

	}

}
