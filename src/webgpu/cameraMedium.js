import { DoubleSide, Matrix4, Ray, Vector3 } from 'three';

const _position = new Vector3();
const _inverse = new Matrix4();
const _ray = new Ray();

// THE MEDIUM THE CAMERA STANDS IN, as an index in the material table, or -1.
//
// Cycles decides it per path (integrator_volume_stack_init): from the origin of the camera ray it
// traces up the Z axis, and a medium whose boundary is first met from BEHIND contains the origin.
// A pinhole camera has one origin for every ray, so here the same test runs once on the host, per
// mesh: the nearest hit of its own geometry along the local up axis, and a back face of a material
// that is only a medium means the camera is inside it. With depth of field the origins spread over
// the lens, and a lens straddling the boundary sees one medium - the declared divergence. Like
// Cycles, it wants closed meshes whose normals face out.
export function cameraMediumIndex( scene, camera, materialsMap ) {

	if ( ! scene || ! camera || ! materialsMap ) return - 1;
	camera.getWorldPosition( _position );

	let found = - 1;
	scene.traverseVisible( object => {

		if ( found !== - 1 || ! object.isMesh || ! object.geometry?.boundsTree ) return;
		const materials = Array.isArray( object.material ) ? object.material : [ object.material ];
		if ( ! materials.some( m => m?.medium?.onlyVolume ) ) return;

		_inverse.copy( object.matrixWorld ).invert();
		_ray.origin.copy( _position ).applyMatrix4( _inverse );
		_ray.direction.set( 0, 0, 1 );
		const hit = object.geometry.boundsTree.raycastFirst( _ray, DoubleSide );
		if ( ! hit || hit.face.normal.dot( _ray.direction ) <= 0 ) return;

		const material = Array.isArray( object.material ) ? object.material[ hit.face.materialIndex ?? 0 ] : object.material;
		if ( material?.medium?.onlyVolume && materialsMap.has( material ) ) found = materialsMap.get( material );

	} );
	return found;

}
