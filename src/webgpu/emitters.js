import { Matrix4, Vector3, FrontSide, BackSide } from 'three';

// -- EMISSIVE TRIANGLES AS LIGHTS: the table next event estimation samples --
//
// A mesh with an emissive material lit the scene only when a bsdf-sampled ray happened to hit it:
// a small bright panel was a firefly generator. Cycles puts every emissive triangle in its light
// distribution (scene/light.cpp) and weighs the two estimators with MIS; this module builds that
// distribution on the CPU, and the kernels sample it (LightsInfoNode) and weigh the hits
// (MaterialKernel).
//
// Each triangle weighs its world-space AREA times the LUMINANCE of its emission, so the choice
// follows the power as the light tree of Cycles does, without the tree. The consequence that
// keeps the kernel simple: the density per unit area at any point of an emitter is
// luminance / total, the same for every triangle of a material. A bsdf-sampled ray that hits one
// needs that single number, carried by the material record, and no lookup of the triangle.
//
// A material enters only if every hit on it sees the same radiance the table promises: no
// emission map (the table carries one radiance per triangle), no transparency or alpha test (a
// hit may pass through), no mix chain (the hit may pick the other branch), and not used by a
// skinned or batched mesh, whose triangles the table cannot place. The others keep the bsdf-only
// estimator at full weight, exactly as before: their area pdf stays zero.

// floats per triangle: four vec4 - v0 and the cumulative weight, v1 and the area pdf, v2 and the
// side, the radiance
export const EMITTER_STRIDE = 16;

const LUMINANCE = [ 0.2126, 0.7152, 0.0722 ];

const _matrix = new Matrix4();
const _instance = new Matrix4();
const _a = new Vector3();
const _b = new Vector3();
const _c = new Vector3();
const _ab = new Vector3();
const _ac = new Vector3();

// Which side of a surface a ray can hit, as the material record stores it: 1 front, -1 back, 0
// both. A transmissive solid is two-sided whatever its side says, since a ray must get out of it.
// The table and the record writer both read it here, so the side NEE samples is the side the
// raycast accepts.
export function materialSideValue( material ) {

	const thickness = material.thickness ?? 0;
	const attenuationDistance = material.attenuationDistance ?? Infinity;
	const isThinWall = thickness === 0 && attenuationDistance === Infinity;
	if ( ! isThinWall && ( material.transmission ?? 0 ) > 0 ) return 0;
	if ( material.side === FrontSide ) return 1;
	if ( material.side === BackSide ) return - 1;
	return 0;

}

// The radiance an emissive material gives at every point of its surface, or null when it is not
// a light the table can carry.
export function emitterRadiance( material ) {

	if ( ! material || ! material.emissive ) return null;
	if ( material.emissiveMap || material.transparent || ( material.alphaTest ?? 0 ) > 0 || material.alphaMap ) return null;
	if ( material.mixMaterial ) return null;

	const intensity = material.emissiveIntensity ?? 1;
	const r = material.emissive.r * intensity;
	const g = material.emissive.g * intensity;
	const b = material.emissive.b * intensity;
	const luminance = r * LUMINANCE[ 0 ] + g * LUMINANCE[ 1 ] + b * LUMINANCE[ 2 ];
	return luminance > 0 ? { r, g, b, luminance } : null;

}

function materialOfGroup( mesh, group ) {

	return Array.isArray( mesh.material ) ? mesh.material[ group.materialIndex ] : mesh.material;

}

// The triangle ranges of a mesh, each with its material: the groups when the material is an
// array, the whole index otherwise - the same split the BVH packs.
function rangesOf( mesh ) {

	const geometry = mesh.geometry;
	const triangleCount = ( geometry.index ? geometry.index.count : geometry.attributes.position.count ) / 3;
	if ( Array.isArray( mesh.material ) ) {

		return geometry.groups.map( group => ( {
			material: materialOfGroup( mesh, group ),
			start: group.start / 3,
			count: Math.min( group.count / 3, triangleCount - group.start / 3 ),
		} ) );

	}

	return [ { material: mesh.material, start: 0, count: triangleCount } ];

}

/**
 * Builds the emitter table of a scene.
 *
 * @param {import('three').Object3D} root
 * @returns {{ data: Float32Array, count: number, totalPower: number, areaPdf: Map<import('three').Material, number>, bounds: number[] }}
 */
export function collectEmitters( root ) {

	root.updateMatrixWorld( true );

	// the meshes the table can place, and the materials it must refuse because some mesh it
	// cannot place uses them too
	const meshes = [];
	const refused = new Set();
	root.traverseVisible( object => {

		if ( ! object.isMesh || ! object.geometry?.attributes?.position ) return;
		const materials = Array.isArray( object.material ) ? object.material : [ object.material ];
		if ( object.isSkinnedMesh || object.isBatchedMesh ) {

			materials.forEach( m => refused.add( m ) );
			return;

		}

		meshes.push( object );

	} );

	const entries = [];
	let totalPower = 0;
	for ( const mesh of meshes ) {

		const position = mesh.geometry.attributes.position;
		const index = mesh.geometry.index;
		const instances = mesh.isInstancedMesh ? mesh.count : 1;
		for ( const range of rangesOf( mesh ) ) {

			if ( refused.has( range.material ) ) continue;
			const radiance = emitterRadiance( range.material );
			if ( ! radiance ) continue;

			const side = materialSideValue( range.material );
			for ( let instance = 0; instance < instances; instance ++ ) {

				_matrix.copy( mesh.matrixWorld );
				if ( mesh.isInstancedMesh ) {

					mesh.getMatrixAt( instance, _instance );
					_matrix.multiply( _instance );

				}

				// the raycast tells front from back in object space: a mirroring transform turns the
				// world-space winding around, and with it the side the table must sample
				const winding = _matrix.determinant() < 0 ? - 1 : 1;
				for ( let t = range.start; t < range.start + range.count; t ++ ) {

					const i0 = index ? index.getX( 3 * t ) : 3 * t;
					const i1 = index ? index.getX( 3 * t + 1 ) : 3 * t + 1;
					const i2 = index ? index.getX( 3 * t + 2 ) : 3 * t + 2;
					_a.fromBufferAttribute( position, i0 ).applyMatrix4( _matrix );
					_b.fromBufferAttribute( position, i1 ).applyMatrix4( _matrix );
					_c.fromBufferAttribute( position, i2 ).applyMatrix4( _matrix );
					const area = 0.5 * _ab.subVectors( _b, _a ).cross( _ac.subVectors( _c, _a ) ).length();
					if ( ! ( area > 0 ) ) continue;

					const power = area * radiance.luminance;
					totalPower += power;
					entries.push( {
						a: _a.toArray(), b: _b.toArray(), c: _c.toArray(),
						power, radiance, material: range.material, side: side * winding,
					} );

				}

			}

		}

	}

	// the bounding sphere of the table: NEE weighs the whole table as its power over the squared
	// distance to it when it picks which light to sample
	const min = [ Infinity, Infinity, Infinity ], max = [ - Infinity, - Infinity, - Infinity ];
	for ( const e of entries ) {

		for ( const v of [ e.a, e.b, e.c ] ) {

			for ( let k = 0; k < 3; k ++ ) {

				min[ k ] = Math.min( min[ k ], v[ k ] );
				max[ k ] = Math.max( max[ k ], v[ k ] );

			}

		}

	}

	const bounds = entries.length === 0 ? [ 0, 0, 0, 0 ] : [
		( min[ 0 ] + max[ 0 ] ) / 2, ( min[ 1 ] + max[ 1 ] ) / 2, ( min[ 2 ] + max[ 2 ] ) / 2,
		0.5 * Math.hypot( max[ 0 ] - min[ 0 ], max[ 1 ] - min[ 1 ], max[ 2 ] - min[ 2 ] ),
	];

	const areaPdf = new Map();
	const data = new Float32Array( Math.max( entries.length, 2 ) * EMITTER_STRIDE );
	let cumulative = 0;
	entries.forEach( ( e, i ) => {

		cumulative += e.power;
		const o = i * EMITTER_STRIDE;
		const pdf = e.radiance.luminance / totalPower;
		areaPdf.set( e.material, pdf );
		data.set( e.a, o );
		// the last entry is one exactly: a sample of 1 - 2^-24 must still land on a triangle
		data[ o + 3 ] = i === entries.length - 1 ? 1 : cumulative / totalPower;
		data.set( e.b, o + 4 );
		data[ o + 7 ] = pdf;
		data.set( e.c, o + 8 );
		data[ o + 11 ] = e.side;
		data[ o + 12 ] = e.radiance.r;
		data[ o + 13 ] = e.radiance.g;
		data[ o + 14 ] = e.radiance.b;

	} );

	return { data, count: entries.length, totalPower, areaPdf, bounds };

}
