import { StorageBufferAttribute, StorageTexture, RGBAIntegerFormat, UnsignedIntType, NearestFilter } from 'three/webgpu';
import { ComputeKernel } from '../ComputeKernel.js';
import { uniform, storage, textureStore, globalId, wgslFn } from 'three/tsl';
import { proxy, proxyFn, wgslTagFn } from 'three-mesh-bvh/webgpu';
import { rayDataStruct } from './structs.js';
import { SVM_REGISTERS } from '../../nodes/svm.wgsl.js';
import { rngInit } from '../../nodes/random.wgsl.js';
import { mixLeafFunc } from '../../nodes/material.wgsl.js';

// ── THE NODE TREE of a surface is resolved in a kernel of its own ──
//
// Between LogicKernel, which stages the hit in the ray data, and MaterialKernel, which
// shades it. For every slot whose hit wears a material flagged "svmResolve" - it has a
// program, or it heads a Mix Shader chain that holds one - it picks the leaf, runs that
// leaf's program at the uv of the hit, and leaves both in a record of two texels per
// slot: which material to shade, and the sockets its program drives.
//
// Not inside getSurfaceRecord, and it is measured: there the interpreter took the
// material kernel from 40 s to 78 s of compilation (FXC is superlinear in the size of a
// shader) and cost 18% of the iterations of a scene with four image materials out of
// twenty-five - it was paid by every hit, program or not. Alone it compiles in 2.7 s.
//
// The leaf of such a chain is picked HERE and not in MaterialKernel because the program
// to run is the leaf's: before the choice nobody knows which one. MaterialKernel reads
// the choice and does not make it again - two walks would be two answers the day they
// differ. Chains without a program it still walks itself (mixLeaf, material.wgsl.js).
//
// Its results do NOT grow the ray data, and that is also measured: the path pool is the
// ray data buffer divided by the struct, 128 MB / 240 bytes = 559k paths, below the
// pixels of a 1280 x 800 view - two more rows would take 12% of the paths in flight
// from every scene, with a program or without. A texture costs only the scenes that
// have one.
//
// Its register file is the one the scene needs (SVM_REGISTER_BUCKETS in svm.wgsl.js),
// so the tracer rebuilds it when a scene crosses a size: 2.7 s, this kernel alone.
//
// The megakernel does not run it, and has no Mix Shader either.

// slots per row of the record texture: the readers divide by it, so it is fixed
export const SVM_SLOTS_PER_ROW = 1024;
// texel 0: the albedo as three float32 words, bit for bit, and roughness and metalness
// as two halves of the fourth. Texel 1: the index of the material to shade.
// Integer and not float because a float texel may flush a denormal - a metalness of
// zero in the high half of a bitcast word is one
export const SVM_RECORD_TEXELS = 2;

export function svmResultsTexture( count ) {

	const rows = Math.max( 1, Math.ceil( count / SVM_SLOTS_PER_ROW ) );
	const tex = new StorageTexture( SVM_SLOTS_PER_ROW * SVM_RECORD_TEXELS, rows );
	tex.format = RGBAIntegerFormat;
	tex.type = UnsignedIntType;
	tex.minFilter = NearestFilter;
	tex.magFilter = NearestFilter;
	tex.generateMipmaps = false;
	tex.name = 'SVM Results';
	return tex;

}

/** How many slots the record texture holds. */
export function svmResultsCapacity( tex ) {

	return ( tex.image.width / SVM_RECORD_TEXELS ) * tex.image.height;

}

/** Where texel "texel" of slot "index" lives: the one layout both kernels read. */
export const svmRecordTexelFn = wgslFn( /* wgsl */ `
	fn svmRecordTexel( index: u32, texel: u32 ) -> vec2i {

		return vec2i( i32( ( index % ${ SVM_SLOTS_PER_ROW }u ) * ${ SVM_RECORD_TEXELS }u + texel ), i32( index / ${ SVM_SLOTS_PER_ROW }u ) );

	}
` );

export class SvmKernel extends ComputeKernel {

	constructor( registers = SVM_REGISTERS ) {

		const params = {
			bvhData: { value: null },
			rayCount: uniform( 0, 'uint' ),

			rayDataStorage: storage( new StorageBufferAttribute( 1, 1 ), rayDataStruct ),
			svmResults: textureStore( svmResultsTexture( 1 ) ).toWriteOnly(),

			globalId: globalId,
		};

		const materialsBuffer = proxy( 'bvhData.value.storage.materials', params );
		const transformsBuffer = proxy( 'bvhData.value.storage.transforms', params );
		const sampleTrianglePointFn = proxyFn( 'bvhData.value.fns.sampleTrianglePoint', params );
		const getUvFromChannelFn = proxyFn( 'bvhData.value.fns.getUvFromChannel', params );
		const sampleMixFactorFn = proxyFn( 'bvhData.value.fns.sampleMixFactor', params );
		const svmRunFn = proxyFn( `bvhData.value.fns.svmRun${ registers }`, params );
		const mixLeafFn = mixLeafFunc( materialsBuffer, sampleMixFactorFn );

		const fn = wgslTagFn/* wgsl */`

			fn compute( rayCount: u32, globalId: vec3u ) -> void {

				let index = globalId.x;
				if ( index >= rayCount ) {

					return;

				}

				// fields of the ray and not the whole element: this kernel wants a few words
				// of 240 bytes. Measured, it makes no difference - the compiler drops the
				// loads nobody uses - and it says what is read. Field reads through the
				// alias compile on WebKit too
				let rayDataStorage = &${ params.rayDataStorage };
				let objectIndex = rayDataStorage[ index ].objectIndex;
				if ( objectIndex < 0 ) {

					return;

				}

				// the material the MESH wears, as MaterialKernel reads it before the mix
				var materialIndex = ${ transformsBuffer }[ u32( objectIndex ) ].materialIndex;
				if ( ${ materialsBuffer }[ materialIndex ].svmResolve == 0u ) {

					return;

				}
				var materialInfo = ${ materialsBuffer }[ materialIndex ];

				// the same surface point MaterialKernel samples: a curve has no vertices, and
				// its call is made with zeros for a struct of the right type
				let isCurve = rayDataStorage[ index ].isCurve == 1u;
				let vertexData = ${ sampleTrianglePointFn }(
					select( rayDataStorage[ index ].barycoord, vec3f( 0.0 ), isCurve ),
					select( rayDataStorage[ index ].indices, vec3u( 0u ), isCurve ),
				);

				// THE SAME SEQUENCE as MaterialKernel at this vertex, so the chain draws the
				// dimensions reserved for it and not numbers from somewhere else. A test holds
				// the two lines equal (tests/tracerPatch.test.mjs)
				let indexUV = vec2u( rayDataStorage[ index ].pixelIndex >> 16, rayDataStorage[ index ].pixelIndex & 0xFFFF );
				${ rngInit }( indexUV, rayDataStorage[ index ].seed, rayDataStorage[ index ].currentBounce + rayDataStorage[ index ].alphaDepth + rayDataStorage[ index ].subsurfaceSteps );

				// the leaf of the chain (nodes/material.wgsl.js, mixLeaf): this kernel walks
				// only the chains that hold a program, and MaterialKernel reads its choice
				materialIndex = ${ mixLeafFn }( materialIndex, &materialInfo, vertexData );

				// the leaf's program, on a surface: a fibre takes its colour from the hair
				// fields of the record, and has no uv of its own to run one at
				var albedo = vec3f( 0.0 );
				var roughness = 0.0;
				var metalness = 0.0;
				if ( ! isCurve && materialInfo.svmCount > 0u ) {

					var regs: array<vec4f, ${ registers }>;
					_ = ${ svmRunFn }(
						materialInfo.svmCode,
						materialInfo.svmCount,
						materialInfo.svmConsts,
						${ getUvFromChannelFn }( vertexData, 0 ),
						&regs,
					);

					// a socket the program does not drive reads register zero, and
					// MaterialKernel does not use it: the mask is the leaf's svmOutputs
					let outputs = materialInfo.svmOutputs;
					albedo = regs[ min( outputs & 0xffu, ${ registers - 1 }u ) ].xyz;
					roughness = regs[ min( ( outputs >> 8u ) & 0xffu, ${ registers - 1 }u ) ].x;
					metalness = regs[ min( ( outputs >> 16u ) & 0xffu, ${ registers - 1 }u ) ].x;

				}

				textureStore(
					${ params.svmResults },
					${ svmRecordTexelFn }( index, 0u ),
					vec4u( bitcast<vec3u>( albedo ), pack2x16float( vec2f( roughness, metalness ) ) ),
				);
				textureStore( ${ params.svmResults }, ${ svmRecordTexelFn }( index, 1u ), vec4u( materialIndex, 0u, 0u, 0u ) );

			}
		`;

		super( fn( params ) );

		this.registers = registers;
		this.defineUniformAccessors( params );

	}

}
