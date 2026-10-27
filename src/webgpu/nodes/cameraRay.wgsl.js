import { wgslTagFn, rayStruct } from 'three-mesh-bvh/webgpu';

// ── THE CAMERA RAY, built in CAMERA space ──
//
// three-mesh-bvh's ndcToCameraRay unprojects two points through the inverse view-projection - one on the near plane,
// one on the FAR plane - and takes the direction between them. The far point's w is about 1/near - 1/near + 1/far,
// and in float32 that last term drowns in the cancellation once far/near is large: with a view clipped at 0.05 and
// 1,000,000 the far points came out quantised, neighbouring pixels got the same ray, and the whole image was made of
// blocks ten pixels wide - at every sample count. The camera of the same scene, clipped at 10 and 10,000, was fine.
//
// Here only the NEAR point is unprojected, in camera space, where its w is about 1/near and nothing cancels: for a
// perspective camera the direction is that point itself, the eye being the origin of camera space, and for an
// orthographic one it is the axis. Then both go to the world through the camera matrix. The far plane is only the
// ray's length.
export const cameraRayFunc = wgslTagFn/* wgsl */`

	fn cameraRay( ndc: vec2f, inverseProjection: mat4x4f, cameraWorld: mat4x4f, far: f32, orthographic: u32 ) -> ${ rayStruct } {

		let nearPoint = inverseProjection * vec4f( ndc, 0.0, 1.0 );
		let local = nearPoint.xyz / nearPoint.w;
		var ray: ${ rayStruct };
		var direction = vec3f( 0.0, 0.0, - 1.0 );
		// along the ray, from the near plane to the far one: the far plane is at depth "far", and the near point at -local.z
		ray.maxDist = far + local.z;
		if ( orthographic == 0u ) {

			direction = normalize( local );
			ray.maxDist = far / - direction.z - length( local );

		}
		ray.origin = ( cameraWorld * vec4f( local, 1.0 ) ).xyz;
		ray.direction = normalize( ( cameraWorld * vec4f( direction, 0.0 ) ).xyz );
		return ray;

	}

`;
