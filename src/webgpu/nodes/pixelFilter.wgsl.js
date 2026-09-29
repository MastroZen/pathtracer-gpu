import { wgslTagFn } from 'three-mesh-bvh/webgpu';

// THE PIXEL FILTER OF CYCLES: the inverted CDF of the filter (scene/film.cpp filter_table),
// FILTER_TABLE_SIZE values, four to a vec4 because a uniform array element is padded to sixteen
// bytes. The app builds it (src/viewport/pixelFilter.ts) and the kernel reads it per camera ray.
export const PIXEL_FILTER_TABLE_SIZE = 1024;

// lookup_table_read of Cycles (kernel/util/lookup_table.h): linear between two entries
export const pixelFilterReadFunc = ( table ) => wgslTagFn/* wgsl */`

	fn pixelFilterRead( x: f32 ) -> f32 {

		let size = ${ PIXEL_FILTER_TABLE_SIZE }u;
		let xs = clamp( x, 0.0, 1.0 ) * f32( size - 1u );
		let index = min( u32( xs ), size - 1u );
		let nindex = min( index + 1u, size - 1u );
		let t = xs - f32( index );
		let d0 = ${ table }[ index >> 2u ][ index & 3u ];
		let d1 = ${ table }[ nindex >> 2u ][ nindex & 3u ];
		return mix( d0, d1, t );

	}

`;
