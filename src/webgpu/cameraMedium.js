import { DoubleSide, Matrix4, Ray, Vector3 } from 'three';
import { MEDIUM_STACK_EMPTY, MEDIUM_STACK_SIZE } from './nodes/mediumStack.wgsl.js';

const _position = new Vector3();
const _inverse = new Matrix4();
const _ray = new Ray();

// THE MEDIA THE CAMERA STANDS IN: the stack a camera ray starts with, four words - the material in
// the low 16 bits and the transform slot of its object in the high ones, MEDIUM_STACK_EMPTY after
// the last. The object gives a heterogeneous density its coordinates.
//
// Cycles decides it per path (integrator_volume_stack_init): from the origin of the camera ray it
// traces up the Z axis, and a volume whose boundary is met from BEHIND before its front contains
// the origin. A pinhole camera has one origin for every ray, so here the same test runs once on
// the host, per mesh: the nearest hit of its own geometry along the local up axis, and a back face
// of a material with a medium means the camera is inside it - which is Cycles' rule read object by
// object. Fog boxes one inside another give one entry each. With depth of field the origins spread
// over the lens, and a lens straddling a boundary sees one stack - the declared divergence. Like
// Cycles, it wants closed meshes whose normals face out.
export function cameraMediumStack( scene, camera, bvhData ) {

	const stack = new Array( MEDIUM_STACK_SIZE ).fill( MEDIUM_STACK_EMPTY );
	const materialsMap = bvhData?.materialsMap;
	if ( ! scene || ! camera || ! materialsMap ) return stack;
	camera.getWorldPosition( _position );

	let count = 0;
	scene.traverseVisible( object => {

		if ( count >= MEDIUM_STACK_SIZE || ! object.isMesh || ! object.geometry?.boundsTree ) return;
		const materials = Array.isArray( object.material ) ? object.material : [ object.material ];
		if ( ! materials.some( m => m?.medium ) ) return;

		_inverse.copy( object.matrixWorld ).invert();
		_ray.origin.copy( _position ).applyMatrix4( _inverse );
		_ray.direction.set( 0, 0, 1 );
		const hit = object.geometry.boundsTree.raycastFirst( _ray, DoubleSide );
		if ( ! hit || hit.face.normal.dot( _ray.direction ) <= 0 ) return;

		const material = Array.isArray( object.material ) ? object.material[ hit.face.materialIndex ?? 0 ] : object.material;
		if ( material?.medium && materialsMap.has( material ) ) {

			stack[ count ++ ] = ( ( materialsMap.get( material ) & 0xffff ) | ( ( bvhData.getObjectSlot( object ) & 0xffff ) << 16 ) ) >>> 0;

		}

	} );
	return stack;

}
