import { wgslFn } from 'three/tsl';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';

// ── THE SURFACE the Texture Coordinate node reads, rebuilt from a triangle hit ──
//
// Two kernels run a surface program at a hit: SvmKernel, for the leaf and the weights of a chain, and
// TraceShadowRayKernel, for the weights of a chain a shadow ray crosses. One function, so the two read the same
// point: two copies would be two answers the day one of them changes.
//
// vertexData is local: Generated and Object travel with their derivatives along the uv of the triangle - the
// columns of a mat3x3f are the value, d/du and d/dv - so a relief that reads a point beside the hit moves them as
// far as the uv moved. The normal is the shading normal on the hit side, as sd->N in Cycles: the face's where the
// material is flat, and flipped when the ray meets the back. World is the same frame through the object's matrix.
//
// It returns the uv, and the rest through pointers and not a struct: the attribute struct's type lives in the BVH
// data, and naming it here would tie this module to how that data is built. A function without a return type is
// refused by three's parser ("Function is not a WGSL code"), which is why neither function here is void.
export const svmSurfacePointFunc = ( attributes, transforms, sampleTrianglePoint, getUvFromChannel, getGenerated ) => wgslTagFn/* wgsl */`

	fn svmSurfacePoint(
		tri: vec3u, bary: vec3f, objectIndex: u32, direction: vec3f, flat: bool,
		generated: ptr<function, mat3x3f>, objectPos: ptr<function, mat3x3f>,
		world: ptr<function, mat3x3f>, normal: ptr<function, vec3f>,
	) -> vec2f {

		let vertexData = ${ sampleTrianglePoint }( bary, tri );
		let a0 = ${ attributes }[ tri.x ];
		let a1 = ${ attributes }[ tri.y ];
		let a2 = ${ attributes }[ tri.z ];
		let e1 = a1.position.xyz - a0.position.xyz;
		let e2 = a2.position.xyz - a0.position.xyz;
		let t0 = ${ getUvFromChannel }( a0, 0 );
		let t1 = ${ getUvFromChannel }( a1, 0 ) - t0;
		let t2 = ${ getUvFromChannel }( a2, 0 ) - t0;
		let det = t1.x * t2.y - t2.x * t1.y;
		let inv = select( 0.0, 1.0 / det, abs( det ) > 1e-20 );
		let g0 = ${ getGenerated }( a0 );
		let g1 = ${ getGenerated }( a1 ) - g0;
		let g2 = ${ getGenerated }( a2 ) - g0;
		*generated = mat3x3f(
			${ getGenerated }( vertexData ),
			( g1 * t2.y - g2 * t1.y ) * inv,
			( g2 * t1.x - g1 * t2.x ) * inv,
		);
		*objectPos = mat3x3f(
			vertexData.position.xyz,
			( e1 * t2.y - e2 * t1.y ) * inv,
			( e2 * t1.x - e1 * t2.x ) * inv,
		);
		let toWorld = ${ transforms }[ objectIndex ].matrixWorld;
		let linear = mat3x3f( toWorld[ 0 ].xyz, toWorld[ 1 ].xyz, toWorld[ 2 ].xyz );
		*world = mat3x3f(
			( toWorld * vec4f( ( *objectPos )[ 0 ], 1.0 ) ).xyz,
			linear * ( *objectPos )[ 1 ],
			linear * ( *objectPos )[ 2 ],
		);
		let face = cross( e1, e2 );
		var n = select( normalize( vertexData.normal.xyz ), normalize( face ), flat );
		let localDirection = ( ${ transforms }[ objectIndex ].inverseMatrixWorld * vec4f( direction, 0.0 ) ).xyz;
		if ( dot( face, localDirection ) > 0.0 ) {

			n = - n;

		}
		*normal = n;
		return ${ getUvFromChannel }( vertexData, 0 );

	}

`;

// a register read as a link weight: 255 is no link, and weighs zero
export const svmWeightFunc = ( registers ) => wgslFn( /* wgsl */ `
	fn svmWeight( regs: ptr<function, array<vec4f, ${ registers }>>, r: u32 ) -> f32 {

		return select( 0.0, ( *regs )[ min( r, ${ registers - 1 }u ) ].x, r != 255u );

	}
` );

// the weights of a chain's links from the registers its program left: the two words of svmMixOutputs, a byte each
export const svmLinkWeightsFunc = ( svmWeight, registers ) => wgslTagFn/* wgsl */`

	fn svmLinkWeights( regs: ptr<function, array<vec4f, ${ registers }>>, w0: u32, w1: u32, second: ptr<function, vec4f> ) -> vec4f {

		*second = vec4f(
			${ svmWeight }( regs, w1 & 0xffu ), ${ svmWeight }( regs, ( w1 >> 8u ) & 0xffu ),
			${ svmWeight }( regs, ( w1 >> 16u ) & 0xffu ), 0.0,
		);
		return vec4f(
			${ svmWeight }( regs, w0 & 0xffu ), ${ svmWeight }( regs, ( w0 >> 8u ) & 0xffu ),
			${ svmWeight }( regs, ( w0 >> 16u ) & 0xffu ), ${ svmWeight }( regs, ( w0 >> 24u ) & 0xffu ),
		);

	}

`;
