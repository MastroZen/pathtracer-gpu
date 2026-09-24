import { texture, uint, uvec2 } from 'three/tsl';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';
import { rand4, rngInit, rngNextBounce } from './sobol.wgsl.js';
import { RNG_INDEX_ALPHA_TEST } from '../random.wgsl.js';
import { BlueNoiseTexture } from '../../../textures/BlueNoiseTexture.js';

// Based in part on the "stratified" random sample implement from the WebGLPathTracer which
// is in turn based on "hoverinc/ray-tracing-renderer" sampling strategy.
// This random sampling strategy uses the exact sample stratified sobol sequence for every
// pixel on screen, with a fractional blue-noise offset applied. This means that pixels
// along the blue noise stride sample the same directions leading to a blue noise pattern
// emerging in screen space.
//
// References
// - "Blue-Noise Dithered Sampling", Georgiev & Fajardo, SIGGRAPH 2016 Talks
//   https://www.arnoldrenderer.com/research/dither_abstract.pdf
// - Blue noise discussion:
//   https://developer.nvidia.com/blog/rendering-in-real-time-with-spatiotemporal-blue-noise-textures-part-2/

// constants
const BN_SIZE = 64;

// the dimensions of one bounce: one per effect index, and alpha test is the highest. Odd, so
// with the bijection of blueDitherShift no two dimensions share a texel within 20 bounces
const EFFECTS_PER_BOUNCE = RNG_INDEX_ALPHA_TEST + 1;

// construct nodes
const blueNoiseTex = new BlueNoiseTexture( BN_SIZE, 1 );
const blueNoiseTexNode = texture( blueNoiseTex );
const pixelCoord = uvec2( 0 ).toVar( 'blueDitherPixel' );
const bounceIndexVar = uint( 0 ).toVar( 'blueDitherBounce' );

// When dithering, the sobol sampler is seeded with a constant pixel so every pixel
// uses the same sequence, which is modified with the per-pixel blue noise sample.
const blueDitherInitFunc = wgslTagFn/* wgsl */`
	fn blueDitherInitialize( pixel: vec2u, pathIndex: u32, bounceIndex: u32 ) -> void {

		${ pixelCoord } = pixel % vec2u( ${ BN_SIZE }u );
		${ bounceIndexVar } = bounceIndex;
		${ rngInit }( vec2u( 0 ), pathIndex, bounceIndex );

	}
`;

const blueDitherNextBounceFunc = wgslTagFn/* wgsl */`
	fn blueDitherNextBounce() -> void {

		${ bounceIndexVar }++;
		${ rngNextBounce }();

	}
`;

// ONE INDEPENDENT SHIFT PER DIMENSION: the same blue noise, read at an offset that differs for
// every bounce, every effect and every component. A single scalar added to all of them kept the
// pairing between dimensions - which lobe, which direction - the one of the shared sequence in
// every pixel, and its error did not average out across the image: a white furnace with a two
// lobe material came out 4.9% dark at grazing angles with 256 samples. With a shift of its own,
// each dimension still reads as blue noise across the screen, and across the image the shifts
// of any two dimensions are independent, so the pairing changes from pixel to pixel.
// The bounce is in the dimension because the Sobol seed hashes it: with the effect alone, two
// bounces of the same effect shared one shift, the same coupling a bounce apart. A furnace
// with a floor and a wall did not show it (0.1% from Sobol at 256 samples); it is closed
// because it is the coupling that made the single shift biased, not because it was measured.
const blueDitherShiftFunc = wgslTagFn/* wgsl */`
	fn blueDitherShift( dimension: u32 ) -> f32 {

		// 1597 is odd, so this is a bijection of the texels: consecutive dimensions land far apart
		let texel = ( dimension * 1597u ) % ${ BN_SIZE * BN_SIZE }u;
		let offset = vec2u( texel % ${ BN_SIZE }u, texel / ${ BN_SIZE }u );
		let coord = ( ${ pixelCoord } + offset ) % vec2u( ${ BN_SIZE }u );
		return textureLoad( ${ blueNoiseTexNode }, vec2i( coord ), 0 ).r;

	}
`;

const blueDitherRand4Func = wgslTagFn/* wgsl */`
	${ [ blueDitherShiftFunc ] }
	fn blueDitherRand4( effect: u32 ) -> vec4f {

		// the scrambled sobol stratified sample, the same in every pixel
		let stratifiedSample = ${ rand4 }( effect );

		let first = ( ${ bounceIndexVar } * ${ EFFECTS_PER_BOUNCE }u + effect ) * 4u;
		let shift = vec4f(
			${ blueDitherShiftFunc }( first ),
			${ blueDitherShiftFunc }( first + 1u ),
			${ blueDitherShiftFunc }( first + 2u ),
			${ blueDitherShiftFunc }( first + 3u ),
		);
		return fract( stratifiedSample + shift );

	}
`;

const blueDitherRand3Func = wgslTagFn/* wgsl */`
	fn blueDitherRand3( effect: u32 ) -> vec3f {

		return ${ blueDitherRand4Func }( effect ).xyz;

	}
`;

const blueDitherRand2Func = wgslTagFn/* wgsl */`
	fn blueDitherRand2( effect: u32 ) -> vec2f {

		return ${ blueDitherRand4Func }( effect ).xy;

	}
`;

const blueDitherRand1Func = wgslTagFn/* wgsl */`
	fn blueDitherRand1( effect: u32 ) -> f32 {

		return ${ blueDitherRand4Func }( effect ).x;

	}
`;

export {
	blueDitherNextBounceFunc as rngNextBounce,
	blueDitherInitFunc as rngInit,
	blueDitherRand1Func as rand1,
	blueDitherRand2Func as rand2,
	blueDitherRand3Func as rand3,
	blueDitherRand4Func as rand4,
};
