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
// so the tracer rebuilds it when a scene crosses a size: this kernel alone.
//
// The weights of a wired Fac are always compiled in, and it is measured: a variant
// without them compiled in 7.4 s against 7.8 with, inside the noise of the measure.
// What costs is the loop of rounds around the interpreter - 4.7 s before it, once
// per page against the 35 s of the material kernel.
//
// The megakernel does not run it, and has no Mix Shader either.

// slots per row of the record texture: the readers divide by it, so it is fixed
export const SVM_SLOTS_PER_ROW = 1024;
// texel 0: the albedo as three float32 words, bit for bit, and roughness and metalness
// as two halves of the fourth. Texel 1: the index of the material to shade, and the
// x and y of the relief's tangent normal as two halves.
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
		const getGeneratedFn = proxyFn( 'bvhData.value.fns.getGenerated', params );
		const attributesBuffer = proxy( 'bvhData.value.storage.attributes', params );
		const sampleMixFactorFn = proxyFn( 'bvhData.value.fns.sampleMixFactor', params );
		const svmRunFn = proxyFn( `bvhData.value.fns.svmRun${ registers }`, params );
		const mixLeafFn = mixLeafFunc( materialsBuffer, sampleMixFactorFn );

		// a register read as a link weight: 255 is no link, and weighs zero
		const svmWeightFn = wgslFn( /* wgsl */ `
			fn svmWeight( regs: ptr<function, array<vec4f, ${ registers }>>, r: u32 ) -> f32 {

				return select( 0.0, ( *regs )[ min( r, ${ registers - 1 }u ) ].x, r != 255u );

			}
		` );

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

				// ONE CALL SITE of the interpreter, in one round or two: the weights of the
				// chain when its Fac is wired, then the program of the leaf they pick. Two call
				// sites were two copies of the interpreter in this kernel, and FXC paid for
				// both - measured per pipeline, 15.3 s to compile against 9.3 with one.
				// ("pass" would be the name of the counter, and it is a WGSL reserved word.)
				let uv = ${ getUvFromChannelFn }( vertexData, 0 );

				// ── THE SURFACE the Texture Coordinate node reads, in OBJECT space ──
				//
				// vertexData is local: MaterialKernel takes it to the world with the object's
				// matrices. Generated and Object travel with their derivatives along the uv of
				// this triangle - the columns of a mat3x3f are the value, d/du and d/dv - so a
				// relief that reads a point beside the hit moves them as far as the uv moved.
				// The normal is the shading normal on the hit side, as sd->N in Cycles: the
				// face's where the material is flat, and flipped when the ray meets the back.
				// The names carry a prefix: "normal" is the relief further down
				let surfTri = select( rayDataStorage[ index ].indices, vec3u( 0u ), isCurve );
				let surfA0 = ${ attributesBuffer }[ surfTri.x ];
				let surfA1 = ${ attributesBuffer }[ surfTri.y ];
				let surfA2 = ${ attributesBuffer }[ surfTri.z ];
				let surfE1 = surfA1.position.xyz - surfA0.position.xyz;
				let surfE2 = surfA2.position.xyz - surfA0.position.xyz;
				let surfT0 = ${ getUvFromChannelFn }( surfA0, 0 );
				let surfT1 = ${ getUvFromChannelFn }( surfA1, 0 ) - surfT0;
				let surfT2 = ${ getUvFromChannelFn }( surfA2, 0 ) - surfT0;
				let surfDet = surfT1.x * surfT2.y - surfT2.x * surfT1.y;
				let surfInv = select( 0.0, 1.0 / surfDet, abs( surfDet ) > 1e-20 );
				let surfG0 = ${ getGeneratedFn }( surfA0 );
				let surfG1 = ${ getGeneratedFn }( surfA1 ) - surfG0;
				let surfG2 = ${ getGeneratedFn }( surfA2 ) - surfG0;
				let surfGenerated = mat3x3f(
					${ getGeneratedFn }( vertexData ),
					( surfG1 * surfT2.y - surfG2 * surfT1.y ) * surfInv,
					( surfG2 * surfT1.x - surfG1 * surfT2.x ) * surfInv,
				);
				let surfObject = mat3x3f(
					vertexData.position.xyz,
					( surfE1 * surfT2.y - surfE2 * surfT1.y ) * surfInv,
					( surfE2 * surfT1.x - surfE1 * surfT2.x ) * surfInv,
				);
				let surfFace = cross( surfE1, surfE2 );
				var surfNormal = select( normalize( vertexData.normal.xyz ), normalize( surfFace ), materialInfo.flatShading != 0 );
				let surfDirection = ( ${ transformsBuffer }[ u32( objectIndex ) ].inverseMatrixWorld * vec4f( rayDataStorage[ index ].direction, 0.0 ) ).xyz;
				if ( dot( surfFace, surfDirection ) > 0.0 ) {

					surfNormal = - surfNormal;

				}

				var regs: array<vec4f, ${ registers }>;
				let liveWeights = materialInfo.svmMixCount > 0u;
				let rounds = select( 1u, 2u, liveWeights );
				var linkWeights0 = vec4f( 0.0 );
				var linkWeights1 = vec4f( 0.0 );
				var code = materialInfo.svmMixCode;
				var count = materialInfo.svmMixCount;
				var consts = materialInfo.svmMixConsts;
				for ( var round = 0u; round < rounds; round ++ ) {

					// the last round runs the leaf: the chain is walked first, with the weights
					// the round before left in the registers
					if ( round + 1u == rounds ) {

						if ( liveWeights ) {

							let w0 = materialInfo.svmMixOutputs0;
							let w1 = materialInfo.svmMixOutputs1;
							linkWeights0 = vec4f(
								${ svmWeightFn }( &regs, w0 & 0xffu ), ${ svmWeightFn }( &regs, ( w0 >> 8u ) & 0xffu ),
								${ svmWeightFn }( &regs, ( w0 >> 16u ) & 0xffu ), ${ svmWeightFn }( &regs, ( w0 >> 24u ) & 0xffu ),
							);
							linkWeights1 = vec4f(
								${ svmWeightFn }( &regs, w1 & 0xffu ), ${ svmWeightFn }( &regs, ( w1 >> 8u ) & 0xffu ),
								${ svmWeightFn }( &regs, ( w1 >> 16u ) & 0xffu ), 0.0,
							);

						}

						// the leaf of the chain (nodes/material.wgsl.js, mixLeaf): this kernel walks
						// only the chains that hold a program, and MaterialKernel reads its choice
						materialIndex = ${ mixLeafFn }( materialIndex, &materialInfo, vertexData, liveWeights, linkWeights0, linkWeights1 );

						// a fibre takes its colour from the hair fields of the record, and has no
						// uv of its own to run a program at
						code = materialInfo.svmCode;
						count = select( 0u, materialInfo.svmCount, ! isCurve );
						consts = materialInfo.svmConsts;

					}

					if ( count > 0u ) {

						_ = ${ svmRunFn }( code, count, consts, uv, surfGenerated, surfObject, surfNormal, &regs );

					}

				}

				// a socket the program does not drive reads register zero, and MaterialKernel
				// does not use it: the mask is the leaf's svmOutputs
				var albedo = vec3f( 0.0 );
				var roughness = 0.0;
				var metalness = 0.0;
				var normal = vec2f( 0.0 );
				if ( count > 0u ) {

					let outputs = materialInfo.svmOutputs;
					albedo = regs[ min( outputs & 0xffu, ${ registers - 1 }u ) ].xyz;
					roughness = regs[ min( ( outputs >> 8u ) & 0xffu, ${ registers - 1 }u ) ].x;
					metalness = regs[ min( ( outputs >> 16u ) & 0xffu, ${ registers - 1 }u ) ].x;
					// the relief: the x and y of a tangent normal whose z is one
					normal = regs[ min( outputs >> 24u, ${ registers - 1 }u ) ].xy;

				}

				textureStore(
					${ params.svmResults },
					${ svmRecordTexelFn }( index, 0u ),
					vec4u( bitcast<vec3u>( albedo ), pack2x16float( vec2f( roughness, metalness ) ) ),
				);
				textureStore( ${ params.svmResults }, ${ svmRecordTexelFn }( index, 1u ), vec4u( materialIndex, pack2x16float( normal ), 0u, 0u ) );

			}
		`;

		super( fn( params ) );

		this.registers = registers;
		this.defineUniformAccessors( params );

	}

}
