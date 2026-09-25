import { Matrix4, WebGPUCoordinateSystem } from 'three';
import { uniform, PI } from 'three/tsl';
import { wgslTagFn, rayStruct, ndcToCameraRay } from 'three-mesh-bvh/webgpu';
import { rand3, RNG_INDEX_APERTURE_SAMPLE } from '../nodes/random.wgsl.js';
import { PhysicalCamera } from '../../objects/PhysicalCamera.js';

// Aperture sampling as Cycles does it (kernel/camera/camera.h camera_sample_aperture, and
// kernel/sample/mapping.h regular_polygon_sample at v5.2.0): a disk, or a regular polygon with
// its corners on the unit circle, turned by the blade rotation. With no rotation a flat side
// faces +x. The anamorphic ratio then divides x alone, applied by the caller.
const sampleDisk = wgslTagFn/* wgsl */`
	fn sampleDisk( uv: vec2f ) -> vec2f {

		let angle = 2.0 * ${ PI } * uv.x;
		let radius = sqrt( uv.y );
		return vec2f( cos( angle ), sin( angle ) ) * radius;

	}
`;

const samplePolygon = wgslTagFn/* wgsl */`
	fn samplePolygon( corners: f32, rotationIn: f32, rand: vec2f ) -> vec2f {

		var u = rand.x;
		var v = rand.y;

		// pick a corner and reuse u
		let corner = floor( u * corners );
		u = u * corners - corner;

		// uniform weights over the triangle
		u = sqrt( u );
		v = v * u;
		u = 1.0 - u;

		let angle = ${ PI } / corners;
		let p = vec2f( ( u + v ) * cos( angle ), ( u - v ) * sin( angle ) );

		let rotation = rotationIn + corner * 2.0 * angle;
		let cr = cos( rotation );
		let sr = sin( rotation );
		return vec2f( cr * p.x - sr * p.y, sr * p.x + cr * p.y );

	}
`;

PhysicalCamera.prototype.getCameraRayFn = function getCameraRayFn() {

	// camera transform fields
	const invViewProjectionMatrix = uniform( new Matrix4() );
	const cameraWorldMatrix = uniform( new Matrix4() );

	// bokeh shape fields
	const focusDistance = uniform( 0 );
	const bokehSize = uniform( 0 );
	const apertureBlades = uniform( 0, 'int' );
	const apertureRotation = uniform( 0 );
	const anamorphicRatio = uniform( 1 );

	const fn = wgslTagFn/* wgsl */`
		fn getCameraRay( uv: vec2f, resolution: vec2f, ray: ptr<function, ${ rayStruct }> ) -> bool {

			// base ray
			let ndc = uv * 2.0 - vec2f( 1.0 );
			*ray = ${ ndcToCameraRay }( ndc, ${ invViewProjectionMatrix } );

			// depth of field
			// measure focus distance along the optical axis so the focal surface is a flat
			// plane perpendicular to the camera forward vector rather than a sphere.
			let rayDir = ray.direction;
			let forward = normalize( ( ${ cameraWorldMatrix } * vec4f( 0.0, 0.0, - 1.0, 0.0 ) ).xyz );
			let focalPoint = ray.origin + rayDir * ( ${ focusDistance } / dot( rayDir, forward ) );

			// sample the aperture shape: under three blades there is no polygon, and Cycles
			// draws a disk (scene/camera.cpp uploads blades below 3 as 0)
			let shapeUVW = ${ rand3 }( ${ RNG_INDEX_APERTURE_SAMPLE } );
			var apertureSample = ${ sampleDisk }( shapeUVW.xy );
			if ( ${ apertureBlades } >= 3 ) {

				apertureSample = ${ samplePolygon }( f32( ${ apertureBlades } ), ${ apertureRotation }, shapeUVW.xy );

			}

			// the radius is lens / ( 2 fstop ) in meters, blender/camera.cpp
			apertureSample *= ${ bokehSize } * 0.5 * 1e-3;

			// the anamorphic ratio divides x alone, as camera_sample_aperture does
			apertureSample.x /= ${ anamorphicRatio };

			ray.origin += ( ${ cameraWorldMatrix } * vec4f( apertureSample, 0.0, 0.0 ) ).xyz;
			ray.direction = focalPoint - ray.origin;

			return true;

		}
	`;

	const update = () => {

		this.coordinateSystem = WebGPUCoordinateSystem;
		this.updateMatrixWorld();
		this.updateProjectionMatrix();

		invViewProjectionMatrix.value.multiplyMatrices( this.matrixWorld, this.projectionMatrixInverse );
		cameraWorldMatrix.value.copy( this.matrixWorld );

		focusDistance.value = this.focusDistance;
		bokehSize.value = this.bokehSize;
		apertureBlades.value = this.apertureBlades;
		apertureRotation.value = this.apertureRotation;
		anamorphicRatio.value = this.anamorphicRatio;
		return false;

	};

	return { fn, update };

};
