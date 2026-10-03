import { BufferAttribute, BufferGeometry, StorageBufferAttribute, StructTypeNode, Vector4, SkinnedMesh, RepeatWrapping, ClampToEdgeWrapping, MirroredRepeatWrapping, NearestFilter, DataTexture, RedIntegerFormat, UnsignedIntType } from 'three/webgpu';
import { BVHComputeData, intersectRayTriangle, bvhNodeBoundsStruct, bvhNodeStruct, rayStruct, rayIntersectionResultStruct as intersectionResultStruct, wgslTagFn } from 'three-mesh-bvh/webgpu';
import { storage, float, texture, uniformArray, uint } from 'three/tsl';
import { SkinnedMeshBVH, MeshBVH, SAH } from 'three-mesh-bvh';
import { materialStruct } from './structs.wgsl.js';
import { getTextureHash } from '../../core/utils/sceneUpdateUtils.js';
import { sampleTexelFunc } from './utils.wgsl.js';
import { getSurfaceRecordFunc } from './material.wgsl.js';
import { AtlasTexture } from '../AtlasTexture.js';
import { materialSideValue } from '../emitters.js';
import { packSvmPrograms, svmRunFn, SVM_REGISTER_BUCKETS } from './svm.wgsl.js';
import { RNG_INDEX_MIX_SHADER_COUNT } from './random.wgsl.js';

// ── THE WORDS OF THE NODE MACHINE live in a TEXTURE ──
//
// The material kernel already binds eight storage buffers, the minimum a device
// guarantees, so a ninth would break it where no more are granted. An r32uint
// texture read with textureLoad is not a storage buffer. Its width is fixed because
// the reader divides by it.
const SVM_DATA_WIDTH = 1024;

function svmDataTexture( words ) {

	const height = Math.max( 1, Math.ceil( words.length / SVM_DATA_WIDTH ) );
	const data = new Uint32Array( SVM_DATA_WIDTH * height );
	data.set( words );
	const tex = new DataTexture( data, SVM_DATA_WIDTH, height, RedIntegerFormat, UnsignedIntType );
	tex.minFilter = NearestFilter;
	tex.magFilter = NearestFilter;
	tex.generateMipmaps = false;
	tex.needsUpdate = true;
	return tex;

}

const _colorVec = new Vector4();
const transformStruct = new StructTypeNode( {
	matrixWorld: 'mat4x4f',
	inverseMatrixWorld: 'mat4x4f',
	visible: 'uint',
	materialIndex: 'uint',
	_alignment0: 'uint',
	_alignment1: 'uint',
	color: 'vec4f',
}, 'TransformStruct' );

// Pathtracer-specific version of the BVHComputeData tht includes material mapping, property structs
export class PathtracerBVHComputeData extends BVHComputeData {

	constructor( bvh, options = {} ) {

		// TODO: once supported we should use the appropriately-sized member sizes
		super( bvh, {
			attributes: {
				position: 'vec4f',
				normal: 'vec4f',
				tangent: 'vec4f',
				color: 'vec4f',
				uv: 'vec4f',
			},
			...options,
		} );

		this.structs.transform = transformStruct;
		this.structs.material = materialStruct;
		this.storage.materials = null;
		this.materialsMap = new Map();
		// material -> area density in the emitter table, written into each material record;
		// set by the path tracer from collectEmitters before it rewrites the materials
		this.emitterAreaPdf = new Map();
		this.materials = [];
		this.bvhMap = new Map();
		this.textureAtlas = new AtlasTexture();
		// ONE node for the program words, made here and never again: a new node per
		// scene would be a new binding name, and a new name is a new shader - the
		// recompilation the stable names below exist to avoid. A bigger program set
		// swaps the node's value, not the node.
		this.svmTexture = svmDataTexture( new Uint32Array( 4 ) );
		this.svmDataNode = texture( this.svmTexture ).setName( 'bvh_svmData' );
		// the register file the widest program needs, and how many materials SvmKernel
		// resolves: at zero the tracer does not dispatch it
		this.svmRegisters = 0;
		this.svmResolved = 0;
		// how many materials are a HETEROGENEOUS medium: zero, and VolumeKernel is not dispatched
		this.svmMediums = 0;

	}

	updateUvAttributesFromScene() {

		const { attributes } = this;
		// "generated" is the Generated texture coordinate of the Texture Coordinate node,
		// the undeformed mesh normalised on its texture space: a geometry carries it only
		// when one of its materials reads it, so it joins the struct like a second uv
		delete attributes.generated;
		const keys = new Set( [ 'color', 'generated' ] );
		for ( let i = 0; i < 8; i ++ ) {

			const key = i === 0 ? 'uv' : 'uv' + i;
			delete attributes[ key ];
			keys.add( key );

		}

		this.getRootObject().traverse( c => {

			if ( c.geometry ) {

				for ( const key in c.geometry.attributes ) {

					if ( keys.has( key ) ) {

						attributes[ key ] = 'vec4f';

					}

				}

			}

		} );

	}

	updateUvSampleFunction() {

		// TODO: eventually it may be best to pack the uvs into an array in the
		// attribute struct so they can be sampled via array index but TSL structs
		// make this difficult at the moment. It may break down when a uv channel
		// unused, as well.
		const { structs, fns } = this;

		// generate the switch cases for the uv channels
		const cases = [];
		let fallback = null;
		structs.attributes.membersLayout.forEach( ( { name } ) => {

			if ( /^uv/.test( name ) ) {

				const channel = name === 'uv' ? 0 : Number( name.replace( /^uv/, '' ) );
				cases.push( /* wgsl */`
					case ${ channel }u: {

						return vertexData.${ name }.xy;

					}
				` );

				if ( fallback === null ) {

					fallback = `vertexData.${ name }.xy`;

				}

			}

		} );

		fns.getUvFromChannel = wgslTagFn/* wgsl */`
			fn getUvFromChannel( vertexData: ${ structs.attributes }, packed: i32 ) -> vec2f {

				// the uv channel is packed into bits 23-25 of the "*Map" descriptor
				let channel = u32( ( packed >> 23 ) & 0x7 );
				switch ( channel ) {

					${ cases.join( '\n' ) }
					default: {

						return ${ fallback ?? 'vec2f( 0.0 )' };

					}

				}

			}
		`;

	}

	updateColorSampleFn() {

		// create a function for retrieving the color from the vertex data instance. If the struct does not include
		// color then return white.
		const { structs, fns } = this;
		const hasColor = Boolean( structs.attributes.membersLayout.find( ( { name } ) => name === 'color' ) );
		if ( hasColor ) {

			fns.getColor = wgslTagFn/* wgsl */`
				fn getColor( vertexData: ${ structs.attributes } ) -> vec4f {

					return vertexData.color;

				}
			`;

		} else {

			fns.getColor = wgslTagFn/* wgsl */`
				fn getColor( vertexData: ${ structs.attributes } ) -> vec4f {

					return vec4f( 1.0 );

				}
			`;

		}

	}

	updateGeneratedSampleFn() {

		// the Generated coordinate of a vertex, or zero where no geometry of the scene
		// carries the attribute: the same zero a geometry without it reads from the
		// default of its struct member
		const { structs, fns } = this;
		const hasGenerated = Boolean( structs.attributes.membersLayout.find( ( { name } ) => name === 'generated' ) );
		fns.getGenerated = wgslTagFn/* wgsl */`
			fn getGenerated( vertexData: ${ structs.attributes } ) -> vec3f {

				return ${ hasGenerated ? 'vertexData.generated.xyz' : 'vec3f( 0.0 )' };

			}
		`;

	}

	useTransparencyRaycastFn() {

		const { textureAtlas, storage, fns } = this;
		const textures = textureAtlas.texture;
		// A STABLE NAME: without one three names the buffer after the node id, the node is
		// new with every scene, and the kernels that read it came out as a different
		// shader each time - the same scene opened twice recompiled for 37 seconds. With
		// the name the source is identical, and the device reuses the compiled pipeline.
		const textureInfo = uniformArray( textureAtlas.textureInfo, 'uvec4' ).setName( 'bvh_textureInfo' );

		// build the single sampleTexel bound to this instance's textureInfo node
		const sampleTexel = sampleTexelFunc( textureInfo, texture( textures ) );

		// THE NODE MACHINE reads its words from the data texture and its images from the
		// atlas. A clipped image (extension 2) is transparent outside [0, 1]: the atlas
		// clamps, and clip has no wrap mode of its own
		const svmWord = wgslTagFn/* wgsl */`
			fn svmWord( i: u32 ) -> u32 {

				return textureLoad( ${ this.svmDataNode }, vec2i( i32( i % ${ SVM_DATA_WIDTH }u ), i32( i / ${ SVM_DATA_WIDTH }u ) ), 0 ).r;

			}
		`;
		const svmSample = wgslTagFn/* wgsl */`
			fn svmSample( texRef: u32, uv: vec2f, extension: u32, interpolation: u32 ) -> vec4f {

				if ( extension == 2u && ( any( uv < vec2f( 0.0 ) ) || any( uv > vec2f( 1.0 ) ) ) ) { return vec4f( 0.0 ); }
				return ${ sampleTexel }( uv, bitcast<i32>( texRef ), 0.0 );

			}
		`;
		// one interpreter per register file: SvmKernel includes the one its scene needs
		for ( const registers of SVM_REGISTER_BUCKETS ) {

			fns[ `svmRun${ registers }` ] = svmRunFn( svmSample, svmWord, registers );

		}

		// getSurfaceRecord shares the same sampleTexel, so the surface shading and
		// the transparency raycast resolve to one textureInfo binding per pipeline
		fns.getSurfaceRecord = getSurfaceRecordFunc( sampleTexel, fns.getUvFromChannel, fns.getColor );

		// raycast first hit, bounded by the ray's "maxDist" - 0 means unbounded
		const currentMaterialIndex = uint().toVar( 'bvh_materialIndex' );
		const scratchRayScalar = float( 1.0 ).toVar( 'bvh_rayScalar' );
		const baseOpacityScalar = float( 1.0 ).toVar( 'bvh_baseOpacity' );

		fns.raycastFirstHit = this.getShapecastFn( {
			name: 'raycastFirstHit',
			shapeStruct: rayStruct,
			resultStruct: intersectionResultStruct,

			boundsOrderFn: wgslTagFn/* wgsl */`
				fn getBoundsOrder( ray: ${ rayStruct }, splitAxis: u32, node: ${ bvhNodeStruct } ) -> bool {

					return ray.direction[ splitAxis ] >= 0.0;

				}
			`,
			intersectsBoundsFn: wgslTagFn/* wgsl */`
				fn rayIntersectsBounds( ray: ${ rayStruct }, bounds: ${ bvhNodeBoundsStruct }, result: ptr<function, ${ intersectionResultStruct }> ) -> u32 {

					// early-out if our object is completely transparent
					if ( ${ baseOpacityScalar } == 0.0 ) {

						return 0u;

					}

					let boundsMin = vec3( bounds.min[0], bounds.min[1], bounds.min[2] );
					let boundsMax = vec3( bounds.max[0], bounds.max[1], bounds.max[2] );

					let invDir = 1.0 / ray.direction;
					let tMinPlane = ( boundsMin - ray.origin ) * invDir;
					let tMaxPlane = ( boundsMax - ray.origin ) * invDir;

					let tMinHit = min( tMinPlane, tMaxPlane );
					let tMaxHit = max( tMinPlane, tMaxPlane );

					let t0 = max( max( tMinHit.x, tMinHit.y ), tMinHit.z );
					let t1 = min( min( tMaxHit.x, tMaxHit.y ), tMaxHit.z );

					let dist = max( t0, 0.0 );
					if ( t1 < dist ) {

						return 0u;

					} else if ( ray.maxDist > 0.0 && dist * ${ scratchRayScalar } >= ray.maxDist ) {

						return 0u;

					} else if ( result.didHit && dist * ${ scratchRayScalar } >= result.dist ) {

						return 0u;

					} else {

						return 1u;

					}

				}

			`,
			intersectRangeFn: wgslTagFn/* wgsl */`
				fn intersectRange( ray: ${ rayStruct }, offset: u32, count: u32, result: ptr<function, ${ intersectionResultStruct }> ) -> bool {

					var didHit = false;
					for ( var ti = offset; ti < offset + count; ti = ti + 1u ) {

						let i0 = ${ storage.index }[ ti * 3u ];
						let i1 = ${ storage.index }[ ti * 3u + 1u ];
						let i2 = ${ storage.index }[ ti * 3u + 2u ];

						let a = ${ storage.attributes }[ i0 ].position.xyz;
						let b = ${ storage.attributes }[ i1 ].position.xyz;
						let c = ${ storage.attributes }[ i2 ].position.xyz;

						var triResult = ${ intersectRayTriangle }( ray, a, b, c, 0.0 );
						triResult.dist *= ${ scratchRayScalar };
						if ( triResult.didHit && ( ray.maxDist <= 0.0 || triResult.dist < ray.maxDist ) && ( ! result.didHit || triResult.dist < result.dist ) ) {

							let material = ${ storage.materials }[ ${ currentMaterialIndex } ];

							// TODO: if material is a transmissive volume we may need to assume double-sidedness
							if ( material.side != 0 && triResult.side != material.side ) {

								continue;

							}

							if ( material.transparent != 0 || material.alphaTest > 0.0 ) {

								var opacity = ${ baseOpacityScalar };

								// add support for vertex color opacity
								if ( material.vertexColors == 1 ) {

									let barycoord = triResult.barycoord;
									let a = ${ fns.getColor }( ${ storage.attributes }[ i0 ] );
									let b = ${ fns.getColor }( ${ storage.attributes }[ i1 ] );
									let c = ${ fns.getColor }( ${ storage.attributes }[ i2 ] );
									let col = barycoord.x * a + barycoord.y * b + barycoord.z * c;

									opacity *= col.a;

								}

								// account for alpha component of albedo map
								if ( material.map != - 1 ) {

									let barycoord = triResult.barycoord;
									let a = ${ fns.getUvFromChannel }( ${ storage.attributes }[ i0 ], material.map );
									let b = ${ fns.getUvFromChannel }( ${ storage.attributes }[ i1 ], material.map );
									let c = ${ fns.getUvFromChannel }( ${ storage.attributes }[ i2 ], material.map );
									let uv = barycoord.x * a + barycoord.y * b + barycoord.z * c;
									let uvPrime = material.mapTransform * vec3f( uv, 1 );

									opacity *= ${ sampleTexel }( uvPrime.xy, material.map, 0 ).a;

								}

								// account for green component of alpha map
								if ( material.alphaMap != - 1 ) {

									let barycoord = triResult.barycoord;
									let a = ${ fns.getUvFromChannel }( ${ storage.attributes }[ i0 ], material.alphaMap );
									let b = ${ fns.getUvFromChannel }( ${ storage.attributes }[ i1 ], material.alphaMap );
									let c = ${ fns.getUvFromChannel }( ${ storage.attributes }[ i2 ], material.alphaMap );
									let uv = barycoord.x * a + barycoord.y * b + barycoord.z * c;
									let uvPrime = material.alphaMapTransform * vec3f( uv, 1 );

									opacity *= ${ sampleTexel }( uvPrime.xy, material.alphaMap, 0 ).g;

								}

								// Opacity is resolved at surface processing but skip full transparent hits here.
								if ( material.transparent != 0 && opacity <= 0.0 ) {

									continue;

								}

								if ( opacity < material.alphaTest ) {

									continue;

								}

							}

							result.didHit = true;
							result.dist = triResult.dist;
							result.normal = triResult.normal;
							result.side = triResult.side;
							result.barycoord = triResult.barycoord;
							result.indices = vec4u( i0, i1, i2, ti );

							didHit = true;

						}

					}

					return didHit;

				}
			`,
			transformShapeFn: wgslTagFn/* wgsl */`
				fn transformRay( ray: ptr<function, ${ rayStruct }>, objectIndex: u32 ) -> void {

					let toLocal = ${ storage.transforms }[ objectIndex ].inverseMatrixWorld;
					ray.origin = ( toLocal * vec4f( ray.origin, 1.0 ) ).xyz;
					ray.direction = ( toLocal * vec4f( ray.direction, 0.0 ) ).xyz;

					let len = length( ray.direction );
					ray.direction /= len;
					${ scratchRayScalar } = 1.0 / len;

					let object = ${ storage.transforms }[ objectIndex ];
					${ currentMaterialIndex } = object.materialIndex;

					let material = ${ storage.materials }[ object.materialIndex ];
					if ( material.transparent == 1 ) {

						${ baseOpacityScalar } = material.opacity * object.color.a;

					} else {

						${ baseOpacityScalar } = 1.0;

					}

				}
			`,
			transformResultFn: wgslTagFn/* wgsl */`
				fn transformResult( hit: ptr<function, ${ intersectionResultStruct }>, objectIndex: u32 ) -> void {

					let toLocal = ${ storage.transforms }[ objectIndex ].inverseMatrixWorld;
					hit.normal = normalize( ( transpose( toLocal ) * vec4f( hit.normal, 0.0 ) ).xyz );
					hit.objectIndex = objectIndex;

				}
			`,
			resetShapeFn: wgslTagFn/* wgsl */`
				fn resetRayScalar( objectIndex: u32 ) -> void {

					${ scratchRayScalar } = 1.0;
					${ baseOpacityScalar } = 1.0;

				}
			`,
		} );

	}

	update() {

		const { structs } = this;
		const attr = new StorageBufferAttribute( new Uint8Array(), structs.material.getLength() );
		this.storage.materials = storage( attr, structs.material ).toReadOnly().setName( 'bvh_materials' );

		this.updateUvAttributesFromScene();

		super.update();

		// build the channel -> uv lookup now that the geometry struct (and its uv members) exist
		this.updateUvSampleFunction();
		this.updateColorSampleFn();
		this.updateGeneratedSampleFn();

		// build material storage
		this.updateMaterials();

		this.bvhMap.clear();
		this.useTransparencyRaycastFn();

	}

	updateMaterialsMap() {

		const { materials, materialsMap } = this;
		materialsMap.clear();
		materials.length = 0;
		this.getRootObject().traverse( o => {

			if ( o.material ) {

				if ( Array.isArray( o.material ) ) {

					o.material.forEach( m => add( m ) );

				} else {

					add( o.material );

				}

			}

		} );

		materials
			.sort( ( a, b ) => {

				return a.uuid < b.uuid ? 1 : - 1;

			} )
			.forEach( ( m, i ) => {

				materialsMap.set( m, i );

			} );

		function add( mat ) {

			if ( ! materialsMap.has( mat ) ) {

				materials.push( mat );
				materialsMap.set( mat, - 1 );

				// Mix Shader: the other branch usually belongs to no mesh of its own, so
				// walking the scene would never reach it. Following the reference here is
				// what lets a mix be expressed without a decoy object for the second
				// shader. Recursing covers nested mixes, and the guard above stops cycles.
				if ( mat.mixMaterial ) add( mat.mixMaterial );

			}

		}

	}


	/**
	 * Lo slot che un oggetto occupa nell'elenco delle trasformate, o -1 se non c'e'.
	 *
	 * Serve a chi deve NOMINARE un oggetto dall'esterno — la peluria gli prende il
	 * materiale — e non puo' indovinare l'indice: lo assegna la deduplicazione del
	 * BVH, che segue l'ordine dei suoi primitivi e non quello della scena.
	 *
	 * Si ricalcola a ogni chiamata (la mappa non si conserva) quindi non e' per il
	 * ciclo di disegno: si chiede quando la scena cambia.
	 *
	 * @param {import('three').Object3D} object
	 * @returns {number}
	 */
	getObjectSlot( object ) {

		if ( ! this.bvh ) return - 1;

		for ( const info of this._getTransformMap( this.bvh ).values() ) {

			if ( info.object === object ) return info.slot;

		}

		return - 1;

	}

	updateMaterials() {

		this.updateMaterialsMap();

		const { materials, storage, structs, bvh } = this;
		const { materialData, textures, svmWords, svmRegisters, svmResolved, svmMediums } = this.writeMaterialsBuffer( materials );
		this.svmMediums = svmMediums;
		this.svmRegisters = svmRegisters;
		this.svmResolved = svmResolved;

		// the program words: in place when they fit, a new texture - on the SAME node -
		// when they do not
		const capacity = this.svmTexture.image.data.length;
		if ( svmWords.length > capacity ) {

			this.svmTexture.dispose();
			this.svmTexture = svmDataTexture( svmWords );
			this.svmDataNode.value = this.svmTexture;

		} else {

			this.svmTexture.image.data.fill( 0 );
			this.svmTexture.image.data.set( svmWords );
			this.svmTexture.needsUpdate = true;

		}

		const materialsStorage = storage.materials.proxyNode;
		const transformsStorage = storage.transforms.proxyNode;
		const count = materialData.length / structs.material.getLength();
		if ( materialsStorage.value.count < count ) {

			materialsStorage.value.dispose();
			materialsStorage.value = new StorageBufferAttribute( materialData, structs.material.getLength() );

		}

		// copy the material buffer content
		materialsStorage.value.array.set( materialData );
		materialsStorage.value.needsUpdate = true;

		// update the transform content
		this._getTransformMap( bvh ).forEach( info => {

			this.writeMaterialData( info, info.slot, transformsStorage.value.array.buffer );

		} );
		transformsStorage.value.needsUpdate = true;

		// save the textures
		this.textures = textures;

	}

	writeMaterialsBuffer( materials ) {

		function encodeTextureWrap( wrap ) {

			switch ( wrap ) {

				case RepeatWrapping:
					return 0;
				case ClampToEdgeWrapping:
					return 1;
				case MirroredRepeatWrapping:
					return 2;
				default:
					return 0;

			}

		}

		// the packed reference of one texture, registering it in the atlas list: shared
		// by the maps and by the images of the node machine
		function textureWord( texture ) {

			const hash = getTextureHash( texture );

			if ( ! textureLookUp.has( hash ) ) {

				textureLookUp.set( hash, textureLookUp.size );
				textures.push( texture );

			}

			const idx = textureLookUp.get( hash );							// 23 bits
			const channel = texture.channel & 7;							// 3 bits
			const wrapS = encodeTextureWrap( texture.wrapS );				// 2 bits
			const wrapT = encodeTextureWrap( texture.wrapT );				// 2 bits
			const nearest = texture.magFilter === NearestFilter ? 1 : 0;	// 1 bit
			return ( nearest << 30 ) | ( wrapT << 28 ) | ( wrapS << 26 ) | ( channel << 23 ) | ( idx & 0x7fffff );

		}

		function getTexture( material, key ) {

			if ( key in material && material[ key ] ) {

				return textureWord( material[ key ] );

			} else {

				return - 1;

			}

		}

		function getField( material, key, def ) {

			return key in material ? material[ key ] : def;

		}

		function writeTextureMatrixToArray( material, textureKey, array, offset ) {

			const texture = material[ textureKey ] && material[ textureKey ].isTexture ? material[ textureKey ] : null;

			// check if texture exists
			if ( texture ) {

				if ( texture.matrixAutoUpdate ) {

					texture.updateMatrix();

				}

				const elements = texture.matrix.elements;

				// Both wgsl struct and elements should be in column-major format
				for ( let i = 0; i < 3; i ++ ) {

					array[ offset + 4 * i + 0 ] = elements[ 3 * i + 0 ];
					array[ offset + 4 * i + 1 ] = elements[ 3 * i + 1 ];
					array[ offset + 4 * i + 2 ] = elements[ 3 * i + 2 ];
					array[ offset + 4 * i + 3 ] = 0; // padding float

				}

			}

			return 12;

		}

		let index = 0;

		// Collect and index the list of textures based on shareable source
		const textureLookUp = new Map();
		const textures = [];

		// THE NODE MACHINE: every material's program in one array of words, its image
		// slots resolved to atlas references by the same registry the maps use. A
		// program whose image is missing does not run: its sockets keep the maps
		const svm = packSvmPrograms( materials.map( m => {

			// a program whose image is missing does not run, and the same for the weights
			const usable = ( program, images ) => program && images.every( t => t && t.isTexture );
			const images = m.svmImages ?? [];
			const mixImages = m.svmMixImages ?? [];
			const mediumImages = m.svmMediumImages ?? [];
			return {
				program: usable( m.svmProgram, images ) ? m.svmProgram : null,
				textureRef: slot => textureWord( images[ slot ] ),
				mixProgram: usable( m.svmMixProgram, mixImages ) ? m.svmMixProgram : null,
				mixTextureRef: slot => textureWord( mixImages[ slot ] ),
				mediumProgram: usable( m.svmMediumProgram, mediumImages ) ? m.svmMediumProgram : null,
				mediumTextureRef: slot => textureWord( mediumImages[ slot ] ),
			};

		} ) );

		// NOTE: make the minimum material buffer length 2 in order to avoid TSL converting it to a scalar
		// TODO: remove this when fixed in three
		const materialBufferLength = Math.max( materials.length, 2 );
		const floatArray = new Float32Array( materialBufferLength * this.structs.material.getLength() );
		const intArray = new Int32Array( floatArray.buffer );

		// ── CHI SCRIVE DEVE SCRIVERE ESATTAMENTE UN RECORD ──
		//
		// La struct ha dei `vec3`, quindi il suo passo e' allineato a quattro float e
		// `getLength()` puo' rispondere piu' dei campi dichiarati. Un record scritto
		// corto non da' errore: sposta OGNI materiale dopo il primo, e la scena esce
		// coi materiali di qualcun altro.
		//
		// Misurato una volta: nove campi aggiunti, record a 293 contro un passo di
		// 296, e una scena senza peli passata da 77 a 31 di luminanza media. Da qui
		// in poi si conta.
		const recordLength = this.structs.material.getLength();
		let svmResolved = 0;
		let svmMediums = 0;

		// TODO: make features work
		// features.reset();
		for ( let i = 0, l = materials.length; i < l; i ++ ) {

			const recordStart = index;

			const m = materials[ i ];

			// if ( m.isFogVolumeMaterial ) {
			//
			// 	// features.setUsed( 'FOG' );
			//
			// 	for ( let j = 0; j < MATERIAL_STRIDE; j ++ ) {
			//
			// 		floatArray[ index + j ] = 0;
			//
			// 	}
			//
			// 	// sample 0 .rgb
			// 	floatArray[ index + 0 * 4 + 0 ] = m.color.r;
			// 	floatArray[ index + 0 * 4 + 1 ] = m.color.g;
			// 	floatArray[ index + 0 * 4 + 2 ] = m.color.b;
			//
			// 	// sample 2 .a
			// 	floatArray[ index + 2 * 4 + 3 ] = getField( m, 'emissiveIntensity', 0.0 );
			//
			// 	// sample 3 .rgb
			// 	floatArray[ index + 3 * 4 + 0 ] = m.emissive.r;
			// 	floatArray[ index + 3 * 4 + 1 ] = m.emissive.g;
			// 	floatArray[ index + 3 * 4 + 2 ] = m.emissive.b;
			//
			// 	// sample 13 .g
			// 	// reusing opacity field
			// 	floatArray[ index + 13 * 4 + 1 ] = m.density;
			//
			// 	// side
			// 	floatArray[ index + 13 * 4 + 3 ] = 0.0;
			//
			// 	// sample 14 .b
			// 	floatArray[ index + 14 * 4 + 2 ] = 1 << 2;
			//
			// 	index += MATERIAL_STRIDE;
			// 	continue;
			//
			// }

			// color - offset 0
			floatArray[ index ++ ] = m.color.r;
			floatArray[ index ++ ] = m.color.g;
			floatArray[ index ++ ] = m.color.b;
			intArray[ index ++ ] = getTexture( m, 'map' );

			// metalness & roughness - offset 4
			floatArray[ index ++ ] = getField( m, 'metalness', 0.0 );
			intArray[ index ++ ] = getTexture( m, 'metalnessMap' );
			floatArray[ index ++ ] = getField( m, 'roughness', 0.0 );
			intArray[ index ++ ] = getTexture( m, 'roughnessMap' );

			// transmission & emissiveIntensity - offset 8
			// three.js assumes a default f0 of 0.04 if no ior is provided which equates to an ior of 1.5
			floatArray[ index ++ ] = getField( m, 'ior', 1.5 );
			floatArray[ index ++ ] = getField( m, 'transmission', 0.0 );
			intArray[ index ++ ] = getTexture( m, 'transmissionMap' );
			floatArray[ index ++ ] = getField( m, 'emissiveIntensity', 0.0 );

			// emission - offset 12
			if ( 'emissive' in m ) {

				floatArray[ index ++ ] = m.emissive.r;
				floatArray[ index ++ ] = m.emissive.g;
				floatArray[ index ++ ] = m.emissive.b;

			} else {

				floatArray[ index ++ ] = 0.0;
				floatArray[ index ++ ] = 0.0;
				floatArray[ index ++ ] = 0.0;

			}

			intArray[ index ++ ] = getTexture( m, 'emissiveMap' );

			// normals - offset 16
			intArray[ index ++ ] = getTexture( m, 'normalMap' );
			index ++; // because of vec2 alignment
			if ( 'normalScale' in m ) {

				floatArray[ index ++ ] = m.normalScale.x;
				floatArray[ index ++ ] = m.normalScale.y;

 			} else {

 				floatArray[ index ++ ] = 1;
 				floatArray[ index ++ ] = 1;

 			}

			// clearcoat - offset 20
			floatArray[ index ++ ] = getField( m, 'clearcoat', 0.0 );
			intArray[ index ++ ] = getTexture( m, 'clearcoatMap' );
			intArray[ index ++ ] = getTexture( m, 'clearcoatNormalMap' );
			index ++; // because of vec2 alignment

			// offset 24
			if ( 'clearcoatNormalScale' in m ) {

				floatArray[ index ++ ] = m.clearcoatNormalScale.x;
				floatArray[ index ++ ] = m.clearcoatNormalScale.y;

			} else {

				floatArray[ index ++ ] = 1;
				floatArray[ index ++ ] = 1;

			}

			floatArray[ index ++ ] = getField( m, 'clearcoatRoughness', 0.0 );
			intArray[ index ++ ] = getTexture( m, 'clearcoatRoughnessMap' );

			// iridescence - offset 28
			intArray[ index ++ ] = getTexture( m, 'iridescenceMap' );
			intArray[ index ++ ] = getTexture( m, 'iridescenceThicknessMap' );

			floatArray[ index ++ ] = getField( m, 'iridescence', 0.0 );
			floatArray[ index ++ ] = getField( m, 'iridescenceIOR', 1.3 );

			// offset 32
			const iridescenceThicknessRange = getField( m, 'iridescenceThicknessRange', [ 100, 400 ] );
			floatArray[ index ++ ] = iridescenceThicknessRange[ 0 ];
			floatArray[ index ++ ] = iridescenceThicknessRange[ 1 ];
			floatArray[ index ++ ] = getField( m, 'diffuseRoughness', 0.0 );
			floatArray[ index ++ ] = getField( m, 'dispersion', 0.0 );

			// specular color - offset 36
			if ( 'specularColor' in m ) {

				floatArray[ index ++ ] = m.specularColor.r;
				floatArray[ index ++ ] = m.specularColor.g;
				floatArray[ index ++ ] = m.specularColor.b;

			} else {

				floatArray[ index ++ ] = 1.0;
				floatArray[ index ++ ] = 1.0;
				floatArray[ index ++ ] = 1.0;

			}

			intArray[ index ++ ] = getTexture( m, 'specularColorMap' );

			// specular intensity - offset 40
			floatArray[ index ++ ] = getField( m, 'specularIntensity', 1.0 );
			intArray[ index ++ ] = getTexture( m, 'specularIntensityMap' );

			// isThinWall
			const isThinWall = getField( m, 'thickness', 0.0 ) === 0.0 && getField( m, 'attenuationDistance', Infinity ) === Infinity;
			intArray[ index ++ ] = Number( isThinWall );
			index ++;

			// attenuation - offset 44
			if ( 'attenuationColor' in m ) {

				floatArray[ index ++ ] = m.attenuationColor.r;
				floatArray[ index ++ ] = m.attenuationColor.g;
				floatArray[ index ++ ] = m.attenuationColor.b;

			} else {

				floatArray[ index ++ ] = 1.0;
				floatArray[ index ++ ] = 1.0;
				floatArray[ index ++ ] = 1.0;

			}

			floatArray[ index ++ ] = getField( m, 'attenuationDistance', Infinity );

			// alphaMap - offset 48
			intArray[ index ++ ] = getTexture( m, 'alphaMap' );
			intArray[ index ++ ] = Number( getField( m, 'castShadow', true ) ); // shadow
			floatArray[ index ++ ] = m.opacity;
			floatArray[ index ++ ] = m.alphaTest;

			// side & matte - offset 52: one definition shared with the emitter table, so the side
			// NEE samples is the side the raycast accepts
			floatArray[ index ++ ] = materialSideValue( m );

			intArray[ index ++ ] = Number( getField( m, 'matte', false ) ); // matte
			floatArray[ index ++ ] = getField( m, 'sheen', 0.0 );
			index ++; // vec3 alignment requirements

			// sheenColor - offset 56
			if ( 'sheenColor' in m ) {

				floatArray[ index ++ ] = m.sheenColor.r;
				floatArray[ index ++ ] = m.sheenColor.g;
				floatArray[ index ++ ] = m.sheenColor.b;

			} else {

				floatArray[ index ++ ] = 0.0;
				floatArray[ index ++ ] = 0.0;
				floatArray[ index ++ ] = 0.0;

			}

			intArray[ index ++ ] = getTexture( m, 'sheenColorMap' );

			// sheenRoughness, flags - offset 60
			floatArray[ index ++ ] = getField( m, 'sheenRoughness', 0.0 );
			intArray[ index ++ ] = getTexture( m, 'sheenRoughnessMap' );

			intArray[ index ++ ] = Number( m.vertexColors );
			intArray[ index ++ ] = Number( m.flatShading );

			// transparent, fogVolume - offset 64
			intArray[ index ++ ] = Number( m.transparent );
			intArray[ index ++ ] = 0;

			// anisotropy, anisotropyRotation, anisotropyMap - offset 66
			floatArray[ index ++ ] = getField( m, 'anisotropy', 0.0 );
			floatArray[ index ++ ] = getField( m, 'anisotropyRotation', 0.0 );
			intArray[ index ++ ] = getTexture( m, 'anisotropyMap' );
			index ++; // padding for mat3 alignment
			index ++;
			index ++;

			// map transform - offset 72
			index += writeTextureMatrixToArray( m, 'map', floatArray, index );

			// metalnessMap transform - offset 84
			index += writeTextureMatrixToArray( m, 'metalnessMap', floatArray, index );

			// roughnessMap transform - offset 96
			index += writeTextureMatrixToArray( m, 'roughnessMap', floatArray, index );

			// transmissionMap transform - offset 108
			index += writeTextureMatrixToArray( m, 'transmissionMap', floatArray, index );

			// emissiveMap transform - offset 120
			index += writeTextureMatrixToArray( m, 'emissiveMap', floatArray, index );

			// normalMap transform - offset 132
			index += writeTextureMatrixToArray( m, 'normalMap', floatArray, index );

			// clearcoatMap transform - offset 144
			index += writeTextureMatrixToArray( m, 'clearcoatMap', floatArray, index );

			// clearcoatNormalMap transform - offset 156
			index += writeTextureMatrixToArray( m, 'clearcoatNormalMap', floatArray, index );

			// clearcoatRoughnessMap transform - offset 168
			index += writeTextureMatrixToArray( m, 'clearcoatRoughnessMap', floatArray, index );

			// sheenColorMap transform - offset 180
			index += writeTextureMatrixToArray( m, 'sheenColorMap', floatArray, index );

			// sheenRoughnessMap transform - offset 192
			index += writeTextureMatrixToArray( m, 'sheenRoughnessMap', floatArray, index );

			// iridescenceMap transform - offset 204
			index += writeTextureMatrixToArray( m, 'iridescenceMap', floatArray, index );

			// iridescenceThicknessMap transform - offset 216
			index += writeTextureMatrixToArray( m, 'iridescenceThicknessMap', floatArray, index );

			// specularColorMap transform - offset 228
			index += writeTextureMatrixToArray( m, 'specularColorMap', floatArray, index );

			// specularIntensityMap transform - offset 240
			index += writeTextureMatrixToArray( m, 'specularIntensityMap', floatArray, index );

			// alphaMap transform - offset 252
			index += writeTextureMatrixToArray( m, 'alphaMap', floatArray, index );

			// anisotropyMap transform - offset 264
			index += writeTextureMatrixToArray( m, 'anisotropyMap', floatArray, index );

			// Mix Shader - offset 276
			// A branch that never made it into the table would leave the kernel indexing
			// out of range, so a missing reference zeroes the weight instead: the record
			// then reads as "no mix", which is the one safe answer.
			const mixIndex = m.mixMaterial ? materials.indexOf( m.mixMaterial ) : - 1;
			const mixWeight = mixIndex < 0 ? 0.0 : getField( m, 'mixWeight', 0.0 );
			floatArray[ index ++ ] = mixWeight;
			intArray[ index ++ ] = Math.max( 0, mixIndex );
			// mixPad: see structs.wgsl.js
			intArray[ index ++ ] = 0;

			// Subsurface - offset 279
			floatArray[ index ++ ] = getField( m, 'subsurfaceWeight', 0.0 );

			// offset 280: the radius is a Color so the three channels can differ, and it
			// arrives already multiplied by the scale - the kernel wants a distance
			const subsurfaceRadius = m.subsurfaceRadius;
			floatArray[ index ++ ] = subsurfaceRadius ? subsurfaceRadius.r : 0.0;
			floatArray[ index ++ ] = subsurfaceRadius ? subsurfaceRadius.g : 0.0;
			floatArray[ index ++ ] = subsurfaceRadius ? subsurfaceRadius.b : 0.0;
			floatArray[ index ++ ] = getField( m, 'subsurfaceAnisotropy', 0.0 );

			// offset 284: i peli. I default sono quelli del Principled Hair di
			// Blender — ruvidita' 0,3 e 0,3, inclinazione 2 gradi — cosi' un materiale
			// che non li dichiara veste comunque un pelo sensato.
			floatArray[ index ++ ] = getField( m, 'hairRoughness', 0.3 );
			floatArray[ index ++ ] = getField( m, 'hairRadialRoughness', 0.3 );
			floatArray[ index ++ ] = getField( m, 'hairTilt', 0.034906585 );
			floatArray[ index ++ ] = getField( m, 'hairRandomColor', 0.0 );
			floatArray[ index ++ ] = getField( m, 'hairRandomRoughness', 0.0 );
			floatArray[ index ++ ] = getField( m, 'hairMelanin', 0.0 );
			floatArray[ index ++ ] = getField( m, 'hairRedness', 1.0 );
			floatArray[ index ++ ] = getField( m, 'hairCoat', 0.0 );
			floatArray[ index ++ ] = getField( m, 'hairIor', 1.55 );

			floatArray[ index ++ ] = getField( m, 'hairModel', 0.0 );
			floatArray[ index ++ ] = getField( m, 'hairAspect', 0.85 );

			// COME SI DA' IL COLORE: 0 riflettanza (il default, cioe' il comportamento
			// di sempre), 1 melanina, 2 assorbimento
			floatArray[ index ++ ] = getField( m, 'hairParametrization', 0.0 );

			// la TINTA: bianca di fabbrica come di la', e bianco vuol dire
			// nessuna tinta perche' la riflettanza piena da' assorbimento zero.
			// TRE FLOAT e non un vec3: il perche' sta in structs.wgsl.js
			const hairTint = m.hairTint;
			floatArray[ index ++ ] = hairTint ? hairTint.r : 1.0;
			floatArray[ index ++ ] = hairTint ? hairTint.g : 1.0;
			floatArray[ index ++ ] = hairTint ? hairTint.b : 1.0;

			// il COEFFICIENTE DI ASSORBIMENTO nudo, col default di Blender
			const hairAbsorption = m.hairAbsorption;
			floatArray[ index ++ ] = hairAbsorption ? hairAbsorption.x : 0.245531;
			floatArray[ index ++ ] = hairAbsorption ? hairAbsorption.y : 0.52;
			floatArray[ index ++ ] = hairAbsorption ? hairAbsorption.z : 1.365;

			// the area density of this material in the emitter table, zero when it is not a light
			floatArray[ index ++ ] = this.emitterAreaPdf.get( m ) ?? 0.0;

			// the node machine, see structs.wgsl.js: no program is a count of zero
			const svmPlace = svm.placements[ i ];
			const svmOutputs = m.svmOutputs ?? {};
			const svmReg = r => ( Number.isInteger( r ) && r >= 0 && r < 255 ? r : 255 );
			intArray[ index ++ ] = svmPlace ? svmPlace.codeWord : 0;
			intArray[ index ++ ] = svmPlace ? svmPlace.count : 0;
			intArray[ index ++ ] = svmPlace ? svmPlace.constWord : 0;
			intArray[ index ++ ] = svmReg( svmOutputs.albedo ) | ( svmReg( svmOutputs.roughness ) << 8 )
				| ( svmReg( svmOutputs.metalness ) << 16 ) | ( svmReg( svmOutputs.normal ) << 24 );

			// SvmKernel resolves the tree of a material that holds a program, on itself or
			// on a leaf of the chain it heads: the leaf's program runs there, and only
			// there can the leaf be picked. As far as the walk can reach: the root and one
			// link per reserved dimension
			let resolves = false;
			for ( let link = m, step = 0; link && step <= RNG_INDEX_MIX_SHADER_COUNT; link = link.mixMaterial, step ++ ) {

				if ( svm.placements[ materials.indexOf( link ) ] ) {

					resolves = true;
					break;

				}

			}
			// and a chain whose weights are a program is walked there too
			if ( svm.mixPlacements[ i ] ) resolves = true;
			intArray[ index ++ ] = resolves ? 1 : 0;
			if ( resolves ) svmResolved ++;

			// the program of the weights of the chain, and the register of each link
			const mixPlace = svm.mixPlacements[ i ];
			const mixOutputs = mixPlace ? m.svmMixProgram.outputs : [];
			const mixReg = k => svmReg( mixOutputs[ k ] );
			intArray[ index ++ ] = mixPlace ? mixPlace.codeWord : 0;
			intArray[ index ++ ] = mixPlace ? mixPlace.count : 0;
			intArray[ index ++ ] = mixPlace ? mixPlace.constWord : 0;
			intArray[ index ++ ] = mixReg( 0 ) | ( mixReg( 1 ) << 8 ) | ( mixReg( 2 ) << 16 ) | ( mixReg( 3 ) << 24 );
			intArray[ index ++ ] = mixReg( 4 ) | ( mixReg( 5 ) << 8 ) | ( mixReg( 6 ) << 16 ) | ( 255 << 24 );

			// an emission leaf: the program's colour is light (see structs.wgsl.js)
			intArray[ index ++ ] = m.svmEmission && svmPlace ? 1 : 0;

			// the participating medium (see structs.wgsl.js): no "medium" is no medium. It is
			// heterogeneous only when its density program made it into the words: a program whose
			// image is missing does not run, and the medium falls back to the density one
			const medium = m.medium ?? null;
			const mediumPlace = svm.mediumPlacements[ i ];
			if ( medium && mediumPlace ) svmMediums ++;
			intArray[ index ++ ] = medium ? ( medium.onlyVolume ? 1 : 0 ) | 2 | ( mediumPlace ? 4 : 0 ) : 0;
			for ( const v of medium ? medium.scatter : [ 0, 0, 0 ] ) floatArray[ index ++ ] = v;
			for ( const v of medium ? medium.absorption : [ 0, 0, 0 ] ) floatArray[ index ++ ] = v;
			floatArray[ index ++ ] = medium ? medium.anisotropy : 0.0;
			for ( const v of medium ? medium.emission : [ 0, 0, 0 ] ) floatArray[ index ++ ] = v;
			intArray[ index ++ ] = mediumPlace ? mediumPlace.codeWord : 0;
			intArray[ index ++ ] = mediumPlace ? mediumPlace.count : 0;
			intArray[ index ++ ] = mediumPlace ? mediumPlace.constWord : 0;
			intArray[ index ++ ] = mediumPlace ? svmReg( m.svmMediumProgram.outputs[ 0 ] ) : 255;
			const space = medium?.textureSpace ?? { min: [ - 0.5, - 0.5, - 0.5 ], size: [ 1, 1, 1 ] };
			for ( const v of space.min ) floatArray[ index ++ ] = v;
			for ( const v of space.size ) floatArray[ index ++ ] = v;
			floatArray[ index ++ ] = medium?.densityMax ?? 0.0;

			if ( index - recordStart !== recordLength ) {

				throw new Error(
					`PathtracerBVHComputeData: il record del materiale e' ${ index - recordStart } parole ` +
					`invece di ${ recordLength }. Aggiungendo un campo alla struct va aggiunta anche la ` +
					`sua scrittura qui, e il totale va tenuto multiplo di quattro.`
				);

			}

		}

		return { materialData: intArray, textures, svmWords: svm.words, svmRegisters: svm.registers, svmResolved, svmMediums };

	}

	writeMaterialData( info, writeOffset, targetBuffer ) {

		// write material data to the transforms
		const { materialsMap } = this;
		const { object, instanceId, root } = info;

		// get the material associated with the bvh group
		let material = object.material;
		if ( Array.isArray( material ) ) {

			const { materialIndex } = object.geometry.groups[ root ];
			material = material[ materialIndex ];

		}

		// save the index
		const index = materialsMap.get( material ) || 0;
		const transformBufferU32 = new Uint32Array( targetBuffer );
		transformBufferU32[ writeOffset * transformStruct.getLength() + 33 ] = index;

		// write color
		// TODO: note that both BatchedMesh and InstancedMesh "getColorAt" functions throw
		// if colors have not been defined.
		if ( object.isInstancedMesh && object.instanceColor ) {

			object.getColorAt( instanceId, _colorVec );
			_colorVec.w = 1.0;

		} else if ( object.isBatchedMesh && object._colorsTexture ) {

			object.getColorAt( instanceId, _colorVec );

		} else {

			_colorVec.setScalar( 1 );

		}

		const transformBufferF32 = new Float32Array( targetBuffer );
		_colorVec.toArray( transformBufferF32, writeOffset * transformStruct.getLength() + 36 );

	}

	writeTransformData( info, premultiplyMatrix, writeOffset, targetBuffer ) {

		super.writeTransformData( info, premultiplyMatrix, writeOffset, targetBuffer );
		this.writeMaterialData( info, writeOffset, targetBuffer );

	}

	getBVH( object, instanceId, rangeTarget ) {

		const { bvhMap } = this;
		const bvh = super.getBVH( object, instanceId, rangeTarget );
		if ( bvhMap.has( bvh ) ) {

			const data = bvhMap.get( bvh );
			Object.assign( rangeTarget, data.range );

			// make sure the mesh and bvh are updated if it's being reused across updates
			if ( bvh !== data.bvh && bvh instanceof SkinnedMeshBVH ) {

				const sourceMesh = bvh.mesh;
				const clonedMesh = data.bvh.mesh;
				clonedMesh.matrixWorld
					.copy( sourceMesh.matrixWorld )
					.decompose( clonedMesh.position, clonedMesh.quaternion, clonedMesh.scale );

				bvh.refit();
				bvh.getBoundingBox( clonedMesh.boundingBox );
				bvh.geometry.computeBoundingBox();

			}

			return data.bvh;

		} else if ( bvh.indirect ) {

			// "indirect" bvhs are not supported since they cannot be unpacked in a way tht will allow for coherent material indices
			const proxyGeometry = new BufferGeometry();
			proxyGeometry.attributes = bvh.geometry.attributes;

			let array;
			if ( bvh.geometry.index ) {

				array = bvh.geometry.index.array.slice( rangeTarget.start, rangeTarget.count + rangeTarget.start );

			} else {

				const { start, count } = rangeTarget;
				array = new Uint32Array( count );
				for ( let i = 0, l = rangeTarget.count; i < l; i ++ ) {

					array[ i ] = start + i;

				}

			}

			proxyGeometry.index = new BufferAttribute( array, 1 );
			rangeTarget.start = 0;

			let newBVH;
			if ( bvh instanceof SkinnedMeshBVH ) {

				const sourceMesh = bvh.mesh;
				const clonedMesh = new SkinnedMesh( proxyGeometry );
				clonedMesh.copy( sourceMesh );
				clonedMesh.matrixWorld
					.copy( sourceMesh.matrixWorld )
					.decompose( clonedMesh.position, clonedMesh.quaternion, clonedMesh.scale );

				newBVH = new SkinnedMeshBVH( clonedMesh, { strategy: SAH, targetLeafSize: 5 } );

			} else {

				newBVH = new MeshBVH( proxyGeometry, { strategy: SAH, targetLeafSize: 5 } );

			}

			bvhMap.set( bvh, { bvh: newBVH, range: { ...rangeTarget } } );
			return newBVH;

		} else {

			return bvh;

		}

	}

	dispose() {

		// TODO: This belongs in three-mesh-bvh
		const { storage } = this;
		for ( const key in storage ) {

			storage[ key ].value?.dispose();

		}

	}

}
