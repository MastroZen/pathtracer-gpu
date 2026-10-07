// THE NODE MACHINE IN WGSL: the interpreter that runs, at every hit, a program
// compiled from the texture graph. It is the Shader Virtual Machine of Cycles
// (kernel/svm/svm.h), and the reason is the same: a program is a buffer, so
// changing a node rewrites a buffer instead of recompiling the kernel.
//
// The compiler and a JavaScript twin live in the application (src/graph/svm.ts and
// src/graph/svmEval.ts): a library does not import from its user. The twin is the
// reference, and checks/e2e/check_svmWgsl.mjs runs both on the same points and
// demands the same numbers. Every function below is a line-by-line transcription
// of the twin, which itself calls the application's baking code.
//
// ── THE FORMAT ──
//
// Four words per instruction: opcode | flags << 8, four packed output registers,
// four packed input registers (8 bits each, 255 = none), and the offset of the
// instruction's constants. A register is a vec4f.
//
// ── ONE FUNCTION PER BLOCK ──
//
// wgslFn parses one function at a time, so every function is its own node with
// its dependencies declared, and the sources stay exported next to the nodes for
// the probes, which compile WGSL by hand.
//
// ── THE WORDS AND THE IMAGES COME FROM THE INCLUDER ──
//
// The program is read through svmWord( i ), and IMAGE reads texels
// through svmSample( ref, uv, extension, interpolation ): this file defines
// neither. The kernel reads the words from a data TEXTURE, not a storage buffer —
// its material kernel binds exactly eight, the guaranteed minimum, and a ninth
// would break it where only eight are granted — and samples its atlas; a probe
// reads a buffer and draws a synthetic pattern. The image reference is a raw word
// (packSvmPrograms writes it), everything else a float's bits.
import { wgslFn } from 'three/tsl';

/** The opcodes. They must equal SVM_OP in src/graph/svm.ts, and a test compares them. Three was BAKED, a baked
 * map of a node the machine did not run: it left with every node running live, and the number is not reused. */
export const SVM_OPCODES = Object.freeze( {
	UV: 1, CONST: 2, IMAGE: 4, MAPPING: 5, NOISE: 6, VORONOI: 7, WAVE: 8,
	MAGIC: 9, GRADIENT: 10, WHITE_NOISE: 11, COLOR_RAMP: 12, MIX: 13, INVERT: 14,
	HUE_SATURATION: 15, BRIGHT_CONTRAST: 16, MATH: 17, MAP_RANGE: 18,
	SEPARATE_COLOR: 19, COMBINE_COLOR: 20, COORD: 21, NORMAL_MAP: 22, UV_MAP: 23, ALPHA: 24,
} );

/** The outputs of the Texture Coordinate node COORD reads, in the order its flag numbers them. */
export const SVM_COORD_ORDER = Object.freeze( [ 'generated', 'object', 'normal' ] );

/** The index tables, in the order the compiler numbers them. A test compares them too. */
export const SVM_BLEND_ORDER = Object.freeze( [
	'normal', 'darken', 'multiply', 'colorBurn', 'lighten', 'screen', 'colorDodge', 'add',
	'overlay', 'softLight', 'linearLight', 'difference', 'exclusion', 'subtract', 'divide',
	'hue', 'saturation', 'color', 'value',
] );
export const SVM_MATH_ORDER = Object.freeze( [
	'add', 'subtract', 'multiply', 'divide', 'power', 'minimum', 'maximum', 'absolute',
	'greaterThan', 'lessThan',
] );
export const SVM_GRADIENT_ORDER = Object.freeze( [ 'linear', 'quadratic', 'easing', 'diagonal', 'spherical', 'radial' ] );
export const SVM_WAVE_PROFILE_ORDER = Object.freeze( [ 'sine', 'saw', 'triangle' ] );
/** The LARGEST private register file of one invocation: 32 vec4f, 512 bytes. */
export const SVM_REGISTERS = 32;

// ── THE REGISTER FILE IS SIZED TO THE SCENE ──
//
// A private array indexed at run time costs by its DECLARED size, not by the part a
// program uses - and it is measured, on a room where every bounce runs the program:
// a gradient (one register) took the path tracer from 61 iterations a second to 37
// with a file of 32, to 55 with 16, to 60 with 8. Noise times Voronoi, three
// registers, goes from 24 to 58. The nodes cost little; the file cost everything.
// The graphs of the real projects use one to three.
//
// So the kernel that runs the programs is built with the smallest of these that
// holds the widest program of the scene, and rebuilt when a scene crosses one. The
// interpreter is the same text at every size.
export const SVM_REGISTER_BUCKETS = Object.freeze( [ 8, 16, SVM_REGISTERS ] );

/** The smallest register file that holds a program of "registers". */
export function svmRegisterBucket( registers ) {

	return SVM_REGISTER_BUCKETS.find( size => size >= registers ) ?? SVM_REGISTERS;

}

// The constants of the baking code (raster/proceduralTextures.ts), which this
// library cannot import. A copy of the distortion offsets already diverged once,
// in the live preview, so a test compares these with the originals.
export const SVM_DISTORTION_OFFSETS = Object.freeze( [ 13.5, 27.9, 41.3 ] );
export const SVM_NOISE_COLOR_OFFSETS = Object.freeze( [ Object.freeze( [ 51.7, 17.3, 89.1 ] ), Object.freeze( [ 103.4, 71.9, 5.5 ] ) ] );
export const SVM_WHITE_NOISE_GRAIN = 8192;

/** A number as a WGSL float literal: 8192 would be an integer. */
const f = ( value ) => ( Number.isInteger( value ) ? value.toFixed( 1 ) : String( value ) );
const [ D0, D1, D2 ] = SVM_DISTORTION_OFFSETS.map( f );
const [ G, B ] = SVM_NOISE_COLOR_OFFSETS.map( offsets => offsets.map( f ) );

const O = SVM_OPCODES;

// ── THE HASH, bit for bit ──
//
// The twin uses Math.imul and unsigned shifts on int32: the same bits as u32
// multiplication, which wraps at runtime. The arguments are always runtime values
// here (floors of positions, seeds read from the constants), so no product is
// folded at shader creation, where an overflow would be an error instead of a wrap.
export const SVM_HASH_SOURCE = /* wgsl */ `
fn svmHash( x: i32, y: i32, z: i32, w: i32 ) -> f32 {

	var h = bitcast<u32>( x ) * 374761393u + bitcast<u32>( y ) * 668265263u
		+ bitcast<u32>( z ) * 1103515245u + bitcast<u32>( w ) * 1013904223u;
	h = ( h ^ ( h >> 13u ) ) * 1274126177u;
	h = h ^ ( h >> 16u );
	return f32( h ) / 4294967296.0;

}
`;
export const svmHashFn = wgslFn( SVM_HASH_SOURCE, [] );

export const SVM_SMOOTH_SOURCE = /* wgsl */ `
fn svmSmooth( t: f32 ) -> f32 { return t * t * ( 3.0 - 2.0 * t ); }
`;
export const svmSmoothFn = wgslFn( SVM_SMOOTH_SOURCE, [] );

export const SVM_VALUE_NOISE_SOURCE = /* wgsl */ `
fn svmValueNoise( x: f32, y: f32, z: f32, w: f32 ) -> f32 {

	let fx = floor( x ); let fy = floor( y ); let fz = floor( z ); let fw = floor( w );
	let ix = i32( fx ); let iy = i32( fy ); let iz = i32( fz ); let iw = i32( fw );
	let tx = svmSmooth( x - fx ); let ty = svmSmooth( y - fy );
	let tz = svmSmooth( z - fz ); let tw = svmSmooth( w - fw );
	var total = 0.0;
	for ( var dw = 0; dw < 2; dw ++ ) {

		let pw = select( 1.0 - tw, tw, dw == 1 );
		for ( var dz = 0; dz < 2; dz ++ ) {

			let pz = pw * select( 1.0 - tz, tz, dz == 1 );
			for ( var dy = 0; dy < 2; dy ++ ) {

				let py = pz * select( 1.0 - ty, ty, dy == 1 );
				total += py * ( 1.0 - tx ) * svmHash( ix, iy + dy, iz + dz, iw + dw );
				total += py * tx * svmHash( ix + 1, iy + dy, iz + dz, iw + dw );

			}

		}

	}
	return total;

}
`;
export const svmValueNoiseFn = wgslFn( SVM_VALUE_NOISE_SOURCE, [ svmHashFn, svmSmoothFn ] );

// The octave count is capped at 16 (Blender stops at 15); the twin has no cap.
export const SVM_FBM_SOURCE = /* wgsl */ `
fn svmFbm( x: f32, y: f32, z: f32, w: f32, detail: f32, roughness: f32, lacunarity: f32 ) -> f32 {

	let whole = max( 0.0, floor( detail ) );
	var sum = 0.0;
	var amplitude = 1.0;
	var amplitudeMax = 0.0;
	var frequency = 1.0;
	let octaves = i32( min( whole, 16.0 ) );
	for ( var o = 0; o <= octaves; o ++ ) {

		sum += svmValueNoise( x * frequency, y * frequency, z * frequency, w * frequency ) * amplitude;
		amplitudeMax += amplitude;
		amplitude *= roughness;
		frequency *= lacunarity;

	}
	let fraction = detail - whole;
	if ( fraction <= 0.0 || amplitude == 0.0 ) { return sum / amplitudeMax; }
	let extra = svmValueNoise( x * frequency, y * frequency, z * frequency, w * frequency ) * amplitude;
	let without = sum / amplitudeMax;
	let withIt = ( sum + extra ) / ( amplitudeMax + amplitude );
	return without + ( withIt - without ) * fraction;

}
`;
export const svmFbmFn = wgslFn( SVM_FBM_SOURCE, [ svmValueNoiseFn ] );

// The two HSV conversions of raster/colorOps.ts, with their branches: the
// branchless form is the same function on paper, and here the twin is the paper.
export const SVM_RGB_TO_HSV_SOURCE = /* wgsl */ `
fn svmRgbToHsv( c: vec3f ) -> vec3f {

	let hi = max( c.x, max( c.y, c.z ) );
	let lo = min( c.x, min( c.y, c.z ) );
	let d = hi - lo;
	var h = 0.0;
	if ( d > 1e-9 ) {

		if ( hi == c.x ) { h = ( ( c.y - c.z ) / d ) % 6.0; }
		else if ( hi == c.y ) { h = ( c.z - c.x ) / d + 2.0; }
		else { h = ( c.x - c.y ) / d + 4.0; }
		h = h / 6.0;
		if ( h < 0.0 ) { h += 1.0; }

	}
	return vec3f( h, select( 0.0, d / hi, hi > 1e-9 ), hi );

}
`;
export const svmRgbToHsvFn = wgslFn( SVM_RGB_TO_HSV_SOURCE, [] );

export const SVM_HSV_TO_RGB_SOURCE = /* wgsl */ `
fn svmHsvToRgb( h: f32, s: f32, v: f32 ) -> vec3f {

	let fi = floor( h * 6.0 );
	let f = h * 6.0 - fi;
	let p = v * ( 1.0 - s );
	let q = v * ( 1.0 - f * s );
	let t = v * ( 1.0 - ( 1.0 - f ) * s );
	let sector = ( ( i32( fi ) % 6 ) + 6 ) % 6;
	switch sector {

		case 0: { return vec3f( v, t, p ); }
		case 1: { return vec3f( q, v, p ); }
		case 2: { return vec3f( p, v, t ); }
		case 3: { return vec3f( p, q, v ); }
		case 4: { return vec3f( t, p, v ); }
		default: { return vec3f( v, p, q ); }

	}

}
`;
export const svmHsvToRgbFn = wgslFn( SVM_HSV_TO_RGB_SOURCE, [] );

/** The per-channel blend modes, indices 0..14 of SVM_BLEND_ORDER: b is the background, s the foreground. */
export const SVM_BLEND_CHANNEL_SOURCE = /* wgsl */ `
fn svmBlendChannel( mode: u32, b: f32, s: f32 ) -> f32 {

	switch mode {

		case 1u: { return min( b, s ); }
		case 2u: { return b * s; }
		case 3u: { return select( 1.0 - min( 1.0, ( 1.0 - b ) / s ), 0.0, s <= 0.0 ); }
		case 4u: { return max( b, s ); }
		case 5u: { return b + s - b * s; }
		case 6u: {

			if ( b <= 0.0 ) { return 0.0; }
			if ( s >= 1.0 ) { return 1.0; }
			return min( 1.0, b / ( 1.0 - s ) );

		}
		case 7u: { return b + s; }
		case 8u: { return select( 1.0 - 2.0 * ( 1.0 - b ) * ( 1.0 - s ), 2.0 * b * s, b <= 0.5 ); }
		case 9u: {

			if ( s <= 0.5 ) { return b - ( 1.0 - 2.0 * s ) * b * ( 1.0 - b ); }
			let d = select( sqrt( b ), ( ( 16.0 * b - 12.0 ) * b + 4.0 ) * b, b <= 0.25 );
			return b + ( 2.0 * s - 1.0 ) * ( d - b );

		}
		case 10u: { return b + 2.0 * s - 1.0; }
		case 11u: { return abs( b - s ); }
		case 12u: { return b + s - 2.0 * b * s; }
		case 13u: { return b - s; }
		case 14u: { return select( b / s, 0.0, s <= 0.0 ); }
		default: { return s; }

	}

}
`;
export const svmBlendChannelFn = wgslFn( SVM_BLEND_CHANNEL_SOURCE, [] );

/** Indices 15..18 work on the hue, saturation and value instead of the channels. */
export const SVM_BLEND_SOURCE = /* wgsl */ `
fn svmBlend( mode: u32, bg: vec3f, fg: vec3f ) -> vec3f {

	if ( mode >= 15u ) {

		let b = svmRgbToHsv( bg );
		let s = svmRgbToHsv( fg );
		var hsv = b;
		if ( mode == 15u ) { hsv = vec3f( s.x, b.y, b.z ); }
		else if ( mode == 16u ) { hsv = vec3f( b.x, s.y, b.z ); }
		else if ( mode == 17u ) { hsv = vec3f( s.x, s.y, b.z ); }
		else { hsv = vec3f( b.x, b.y, s.z ); }
		return svmHsvToRgb( hsv.x, hsv.y, hsv.z );

	}
	return vec3f(
		svmBlendChannel( mode, bg.x, fg.x ),
		svmBlendChannel( mode, bg.y, fg.y ),
		svmBlendChannel( mode, bg.z, fg.z ),
	);

}
`;
export const svmBlendFn = wgslFn( SVM_BLEND_SOURCE, [ svmRgbToHsvFn, svmHsvToRgbFn, svmBlendChannelFn ] );

// ── THE MATH NODE, with the answers of Blender where WGSL has none ──
//
// A division by zero gives zero, a negative base gives zero, and pow( 0, b ) is
// spelled out: WGSL leaves it undefined, JavaScript gives 1 for b = 0 and
// infinity for b < 0.
export const SVM_MATH_SOURCE = /* wgsl */ `
fn svmMath( op: u32, a: f32, b: f32 ) -> f32 {

	switch op {

		case 1u: { return a - b; }
		case 2u: { return a * b; }
		case 3u: { return select( a / b, 0.0, b == 0.0 ); }
		case 4u: {

			if ( a < 0.0 ) { return 0.0; }
			if ( a == 0.0 ) {

				if ( b == 0.0 ) { return 1.0; }
				if ( b > 0.0 ) { return 0.0; }
				// the infinity is built from a runtime value: as a constant expression
				// the compiler folds it and refuses "value inf cannot be represented"
				return bitcast<f32>( 0x7f800000u | ( bitcast<u32>( a ) & 0u ) );

			}
			return pow( a, b );

		}
		case 5u: { return min( a, b ); }
		case 6u: { return max( a, b ); }
		case 7u: { return abs( a ); }
		case 8u: { return select( 0.0, 1.0, a > b ); }
		case 9u: { return select( 0.0, 1.0, a < b ); }
		default: { return a + b; }

	}

}
`;
export const svmMathFn = wgslFn( SVM_MATH_SOURCE, [] );

export const SVM_LUM_SOURCE = /* wgsl */ `
fn svmLum( c: vec4f ) -> f32 { return 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z; }
`;
export const svmLumFn = wgslFn( SVM_LUM_SOURCE, [] );

/** The distortion is SEQUENTIAL, as in the twin: the displaced x enters the y. */
export const SVM_NOISE_FAC_SOURCE = /* wgsl */ `
fn svmNoiseFac( p: vec3f, w: f32, distortion: f32, detail: f32, roughness: f32, lacunarity: f32 ) -> f32 {

	var x = p.x; var y = p.y; var z = p.z;
	if ( distortion != 0.0 ) {

		x += ( svmValueNoise( x + ${ D0 }, y, z, w ) - 0.5 ) * distortion;
		y += ( svmValueNoise( x, y + ${ D1 }, z, w ) - 0.5 ) * distortion;
		z += ( svmValueNoise( x, y, z + ${ D2 }, w ) - 0.5 ) * distortion;

	}
	return svmFbm( x, y, z, w, detail, roughness, lacunarity );

}
`;
export const svmNoiseFacFn = wgslFn( SVM_NOISE_FAC_SOURCE, [ svmValueNoiseFn, svmFbmFn ] );

export const SVM_NOISE_COLOR_SOURCE = /* wgsl */ `
fn svmNoiseColor( p: vec3f, w: f32, detail: f32, roughness: f32, lacunarity: f32 ) -> vec4f {

	return vec4f(
		svmFbm( p.x, p.y, p.z, w, detail, roughness, lacunarity ),
		svmFbm( p.x + ${ G[ 0 ] }, p.y + ${ G[ 1 ] }, p.z + ${ G[ 2 ] }, w, detail, roughness, lacunarity ),
		svmFbm( p.x + ${ B[ 0 ] }, p.y + ${ B[ 1 ] }, p.z + ${ B[ 2 ] }, w, detail, roughness, lacunarity ),
		1.0,
	);

}
`;
export const svmNoiseColorFn = wgslFn( SVM_NOISE_COLOR_SOURCE, [ svmFbmFn ] );

/** Distance in x, the cell colour in yzw: one search, two outputs. */
export const SVM_VORONOI_SOURCE = /* wgsl */ `
fn svmVoronoi( p: vec3f, randomness: f32, seed: i32 ) -> vec4f {

	let c = floor( p );
	let f = p - c;
	let ic = vec3i( c );
	var best = 1e30;
	var cell = ic;
	for ( var dz = -1; dz <= 1; dz ++ ) {

		for ( var dy = -1; dy <= 1; dy ++ ) {

			for ( var dx = -1; dx <= 1; dx ++ ) {

				let g = ic + vec3i( dx, dy, dz );
				let o = vec3f(
					f32( dx ) + 0.5 + ( svmHash( g.x, g.y, g.z, seed ) - 0.5 ) * randomness,
					f32( dy ) + 0.5 + ( svmHash( g.x, g.y, g.z, seed + 7919 ) - 0.5 ) * randomness,
					f32( dz ) + 0.5 + ( svmHash( g.x, g.y, g.z, seed + 104729 ) - 0.5 ) * randomness,
				);
				let d = o - f;
				let d2 = d.x * d.x + d.y * d.y + d.z * d.z;
				if ( d2 < best ) { best = d2; cell = g; }

			}

		}

	}
	return vec4f(
		sqrt( best ),
		svmHash( cell.x, cell.y, cell.z, seed + 31 ),
		svmHash( cell.x, cell.y, cell.z, seed + 6151 ),
		svmHash( cell.x, cell.y, cell.z, seed + 262147 ),
	);

}
`;
export const svmVoronoiFn = wgslFn( SVM_VORONOI_SOURCE, [ svmHashFn ] );

export const SVM_WAVE_SOURCE = /* wgsl */ `
fn svmWave( p: vec3f, phase: f32, distortion: f32, detail: f32, w: f32, flags: u32 ) -> f32 {

	var t = select( p.x, length( p ), ( flags & 1u ) != 0u ) + phase;
	if ( distortion != 0.0 ) { t += distortion * ( svmFbm( p.x, p.y, p.z, w, detail, 0.5, 2.0 ) * 2.0 - 1.0 ); }
	let fr = t - floor( t );
	let profile = ( flags >> 1u ) & 3u;
	if ( profile == 1u ) { return fr; }
	if ( profile == 2u ) { return abs( 2.0 * fr - 1.0 ); }
	return 0.5 + 0.5 * sin( ( fr - 0.25 ) * 6.283185307179586 );

}
`;
export const svmWaveFn = wgslFn( SVM_WAVE_SOURCE, [ svmFbmFn ] );

export const SVM_MAGIC_SOURCE = /* wgsl */ `
fn svmMagic( p: vec3f, depth: f32, d: f32 ) -> vec4f {

	var x = sin( ( p.x + p.y + p.z ) * 5.0 );
	var y = cos( ( -p.x + p.y - p.z ) * 5.0 );
	var z = -cos( ( -p.x - p.y + p.z ) * 5.0 );
	if ( depth > 0.0 ) {

		x *= d; y *= d; z *= d;
		y = -cos( x - y + z ) * d;
		if ( depth > 1.0 ) { x = cos( x - y - z ) * d; }
		if ( depth > 2.0 ) { z = sin( -x - y - z ) * d; }
		if ( depth > 3.0 ) { x = -cos( -x + y - z ) * d; }
		if ( depth > 4.0 ) { y = -sin( -x + y + z ) * d; }

	}
	if ( d != 0.0 ) { x /= d * 2.0; y /= d * 2.0; z /= d * 2.0; }
	return vec4f( 0.5 - x, 0.5 - y, 0.5 - z, 1.0 );

}
`;
export const svmMagicFn = wgslFn( SVM_MAGIC_SOURCE, [] );

export const SVM_GRADIENT_SOURCE = /* wgsl */ `
fn svmGradient( p: vec3f, kind: u32 ) -> f32 {

	switch kind {

		case 1u: { let r = max( p.x, 0.0 ); return r * r; }
		case 2u: { let r = clamp( p.x, 0.0, 1.0 ); return r * r * ( 3.0 - 2.0 * r ); }
		case 3u: { return ( p.x + p.y ) * 0.5; }
		case 4u: { return max( 1.0 - length( p ), 0.0 ); }
		case 5u: { return atan2( p.y, p.x ) / 6.283185307179586 + 0.5; }
		default: { return p.x; }

	}

}
`;
export const svmGradientFn = wgslFn( SVM_GRADIENT_SOURCE, [] );

/**
 * The value in x, the colour in yzw. The coordinates are rounded as Math.round
 * does, half UP: WGSL round() goes to the even neighbour by the specification,
 * some adapters go away from zero (measured on a virtual one: round( -2.5 ) = -3),
 * and on a bake grid of 8192 every texel sits on a half.
 */
export const SVM_WHITE_NOISE_SOURCE = /* wgsl */ `
fn svmWhiteNoise( p: vec3f, seed: i32 ) -> vec4f {

	let q = vec3i( floor( p * ${ f( SVM_WHITE_NOISE_GRAIN ) } + 0.5 ) );
	return vec4f(
		svmHash( q.x, q.y, q.z, seed ),
		svmHash( q.x, q.y, q.z, seed + 17 ),
		svmHash( q.x, q.y, q.z, seed + 3389 ),
		svmHash( q.x, q.y, q.z, seed + 49193 ),
	);

}
`;
export const svmWhiteNoiseFn = wgslFn( SVM_WHITE_NOISE_SOURCE, [ svmHashFn ] );

// ── B OVER A, the general W3C formula ──
//
// Opacity and the mask reduce the foreground alpha BEFORE compositing, as the
// twin and the bake do; a missing A is a transparent black.
export const SVM_MIX_SOURCE = /* wgsl */ `
fn svmMix( a: vec4f, hasA: bool, b: vec4f, factor: vec4f, hasFactor: bool, opacity: f32, mode: u32 ) -> vec4f {

	let bg = select( vec4f( 0.0 ), a, hasA );
	let mask = select( 1.0, svmLum( factor ) * factor.w, hasFactor );
	let over = b.w * opacity * mask;
	let outAlpha = over + bg.w * ( 1.0 - over );
	if ( outAlpha <= 0.0 ) { return vec4f( 0.0, 0.0, 0.0, outAlpha ); }
	let fused = svmBlend( mode, bg.xyz, b.xyz );
	let rgb = ( over * ( 1.0 - bg.w ) * b.xyz + over * bg.w * fused + ( 1.0 - over ) * bg.w * bg.xyz ) / outAlpha;
	return vec4f( rgb, outAlpha );

}
`;
export const svmMixFn = wgslFn( SVM_MIX_SOURCE, [ svmLumFn, svmBlendFn ] );

/** A constant is a word read as a float: the program is one array of u32. */
export const SVM_CONST_SOURCE = /* wgsl */ `
fn svmConst( i: u32 ) -> f32 { return bitcast<f32>( svmWord( i ) ); }
`;

/** A register read that tolerates the empty one: 255 reads as zero, and is never an index. */
const svmRegSource = ( registers ) => /* wgsl */ `
fn svmReg( regs: ptr<function, array<vec4f, ${ registers }>>, r: u32 ) -> vec4f {

	return select( vec4f( 0.0 ), ( *regs )[ min( r, ${ registers - 1 }u ) ], r != 255u );

}
`;
export const SVM_REG_SOURCE = svmRegSource( SVM_REGISTERS );

// ── THE INTERPRETER ──
//
// It RETURNS the instruction count, and the reason is three's function parser:
// wgslFn finds the header with a regular expression that wants a return type, so a
// function without one is read on into its body until the first ") - name" — here
// "svmConst( c + 1u ) - low" came out as "-> low", and the kernel did not compile.
// The fork's kernels write "-> void" for this; that is not WGSL, and the probe
// compiles the raw text, so the answer that holds in both places is a real type.
//
// All inputs are read before any output is written: the compiler hands an
// input's register to an output of the same instruction once it is read for the
// last time. An output nobody reads is register 255, and is not computed where
// computing it costs (the colour of the noise).
const svmRunSource = ( registers ) => /* wgsl */ `
fn svmRun(
	codeWord: u32, count: u32, constWord: u32, uv: vec2f,
	generated: mat3x3f, objectPos: mat3x3f, normal: vec3f,
	tri: vec3u, bary: vec3f,
	regs: ptr<function, array<vec4f, ${ registers }>>,
) -> u32 {

	for ( var k = 0u; k < count; k ++ ) {

		let at = codeWord + k * 4u;
		let head = svmWord( at );
		let op = head & 0xffu;
		let flags = head >> 8u;
		let outs = svmWord( at + 1u );
		let ins = svmWord( at + 2u );
		let c = constWord + svmWord( at + 3u );
		let i0 = ins & 0xffu;
		let i1 = ( ins >> 8u ) & 0xffu;
		let i2 = ( ins >> 16u ) & 0xffu;
		let o0 = outs & 0xffu;
		let o1 = ( outs >> 8u ) & 0xffu;
		let o2 = ( outs >> 16u ) & 0xffu;
		let a = svmReg( regs, i0 );
		let b = svmReg( regs, i1 );
		let f = svmReg( regs, i2 );
		var r0 = vec4f( 0.0 );
		var r1 = vec4f( 0.0 );
		var r2 = vec4f( 0.0 );

		switch op {

			case ${ O.UV }u: { r0 = vec4f( uv + vec2f( svmConst( c ), svmConst( c + 1u ) ), 0.0, 1.0 ); }
			case ${ O.CONST }u: { r0 = vec4f( svmConst( c ), svmConst( c + 1u ), svmConst( c + 2u ), svmConst( c + 3u ) ); }
			case ${ O.COORD }u: {

				// the columns are the value and its derivatives along u and v, so a relief
				// that reads a point beside the hit moves Generated and Object as far as the
				// uv moved; the normal does not move, as in Cycles, whose bump moves P only
				let shift = vec3f( 1.0, svmConst( c ), svmConst( c + 1u ) );
				switch flags {

					case 0u: { r0 = vec4f( generated * shift, 1.0 ); }
					case 1u: { r0 = vec4f( objectPos * shift, 1.0 ); }
					default: { r0 = vec4f( normal, 1.0 ); }

				}

			}
			case ${ O.ALPHA }u: { r0 = vec4f( a.w, a.w, a.w, 1.0 ); }
			case ${ O.UV_MAP }u: {

				// a named UV map (the UV Map node): the uv of one channel of the hit triangle with its
				// derivatives along the default uv, moved like UV by the relief. The includer reads the
				// channel from its vertex data (svmUvMap), so only a program that names a map pays for it
				let frame = svmUvMap( tri, bary, u32( svmConst( c + 2u ) ) );
				r0 = vec4f( frame[ 0 ] + frame[ 1 ] * svmConst( c ) + frame[ 2 ] * svmConst( c + 1u ), 0.0, 1.0 );

			}
			case ${ O.IMAGE }u: {

				r0 = svmSample( svmWord( c ), a.xy, u32( svmConst( c + 1u ) ), u32( svmConst( c + 2u ) ) );

			}
			case ${ O.MAPPING }u: {

				let m0 = vec3f( svmConst( c ), svmConst( c + 1u ), svmConst( c + 2u ) );
				let m1 = vec3f( svmConst( c + 3u ), svmConst( c + 4u ), svmConst( c + 5u ) );
				let m2 = vec3f( svmConst( c + 6u ), svmConst( c + 7u ), svmConst( c + 8u ) );
				let location = vec3f( svmConst( c + 9u ), svmConst( c + 10u ), svmConst( c + 11u ) );
				let scale = vec3f( svmConst( c + 12u ), svmConst( c + 13u ), svmConst( c + 14u ) );
				if ( ( flags & 1u ) != 0u ) {

					// texture mode: move, rotate back (the transpose), divide; a zero scale gives zero
					let p = a.xyz - location;
					let rotated = vec3f(
						m0.x * p.x + m1.x * p.y + m2.x * p.z,
						m0.y * p.x + m1.y * p.y + m2.y * p.z,
						m0.z * p.x + m1.z * p.y + m2.z * p.z,
					);
					r0 = vec4f( select( rotated / scale, vec3f( 0.0 ), scale == vec3f( 0.0 ) ), 1.0 );

				} else {

					let p = a.xyz * scale;
					r0 = vec4f( dot( m0, p ) + location.x, dot( m1, p ) + location.y, dot( m2, p ) + location.z, 1.0 );

				}

			}
			case ${ O.NOISE }u: {

				let s = svmConst( c );
				let p = a.xyz * s;
				let w = svmConst( c + 1u );
				let detail = svmConst( c + 3u );
				let roughness = svmConst( c + 4u );
				let lacunarity = svmConst( c + 5u );
				if ( o0 != 255u ) {

					let v = svmNoiseFac( p, w, svmConst( c + 2u ), detail, roughness, lacunarity );
					r0 = vec4f( v, v, v, 1.0 );

				}
				if ( o1 != 255u ) { r1 = svmNoiseColor( p, w, detail, roughness, lacunarity ); }

			}
			case ${ O.VORONOI }u: {

				let v = svmVoronoi( a.xyz * svmConst( c ), svmConst( c + 1u ), i32( svmConst( c + 2u ) ) );
				r0 = vec4f( v.x, v.x, v.x, 1.0 );
				r1 = vec4f( v.yzw, 1.0 );

			}
			case ${ O.WAVE }u: {

				let v = svmWave( a.xyz * svmConst( c ), svmConst( c + 1u ), svmConst( c + 2u ), svmConst( c + 3u ), svmConst( c + 4u ), flags );
				r0 = vec4f( v, v, v, 1.0 );

			}
			case ${ O.MAGIC }u: { r0 = svmMagic( a.xyz * svmConst( c ), svmConst( c + 1u ), svmConst( c + 2u ) ); }
			case ${ O.GRADIENT }u: { let v = svmGradient( a.xyz, flags ); r0 = vec4f( v, v, v, 1.0 ); }
			case ${ O.WHITE_NOISE }u: {

				let v = svmWhiteNoise( a.xyz, i32( svmConst( c ) ) );
				r0 = vec4f( v.x, v.x, v.x, 1.0 );
				r1 = vec4f( v.yzw, 1.0 );

			}
			case ${ O.COLOR_RAMP }u: {

				let low = svmConst( c );
				let span = svmConst( c + 1u ) - low;
				let l = svmLum( a );
				let t = select( clamp( ( l - low ) / span, 0.0, 1.0 ), select( 0.0, 1.0, l >= low ), span == 0.0 );
				let c1 = vec3f( svmConst( c + 2u ), svmConst( c + 3u ), svmConst( c + 4u ) );
				let c2 = vec3f( svmConst( c + 5u ), svmConst( c + 6u ), svmConst( c + 7u ) );
				r0 = vec4f( c1 + ( c2 - c1 ) * t, a.w );

			}
			case ${ O.MIX }u: { r0 = svmMix( a, i0 != 255u, b, f, i2 != 255u, svmConst( c ), flags ); }
			case ${ O.INVERT }u: {

				let fac = select( svmConst( c ), svmLum( b ) * svmConst( c ), i1 != 255u );
				r0 = vec4f( a.xyz + ( 1.0 - 2.0 * a.xyz ) * fac, a.w );

			}
			case ${ O.HUE_SATURATION }u: {

				let hsv = svmRgbToHsv( a.xyz );
				var hue = ( hsv.x + svmConst( c ) ) % 1.0;
				if ( hue < 0.0 ) { hue += 1.0; }
				r0 = vec4f( svmHsvToRgb( hue, min( 1.0, hsv.y * svmConst( c + 1u ) ), hsv.z * svmConst( c + 2u ) ), a.w );

			}
			case ${ O.BRIGHT_CONTRAST }u: {

				let contrast = svmConst( c + 1u );
				r0 = vec4f( max( vec3f( 0.0 ), ( 1.0 + contrast ) * a.xyz + ( svmConst( c ) - contrast * 0.5 ) ), a.w );

			}
			case ${ O.MATH }u: {

				let x = select( 0.0, svmLum( a ), i0 != 255u );
				let y = select( svmConst( c ), svmLum( b ), i1 != 255u );
				var v = svmMath( flags & 0xfu, x, y );
				if ( ( flags & 16u ) != 0u ) { v = clamp( v, 0.0, 1.0 ); }
				r0 = vec4f( v, v, v, 1.0 );

			}
			case ${ O.MAP_RANGE }u: {

				let span = svmConst( c + 1u );
				var t = select( ( svmLum( a ) - svmConst( c ) ) / span, 0.0, span == 0.0 );
				if ( ( flags & 1u ) != 0u ) { t = clamp( t, 0.0, 1.0 ); }
				let v = svmConst( c + 2u ) + t * ( svmConst( c + 3u ) - svmConst( c + 2u ) );
				r0 = vec4f( v, v, v, 1.0 );

			}
			case ${ O.SEPARATE_COLOR }u: {

				r0 = vec4f( a.xxx, 1.0 );
				r1 = vec4f( a.yyy, 1.0 );
				r2 = vec4f( a.zzz, 1.0 );

			}
			case ${ O.NORMAL_MAP }u: {

				// Cycles 5.2 svm_node_normal_map in tangent space: x and y scaled by the
				// strength, z taken toward one by the saturated strength, one normalize. The
				// second constant is the sign of green (DirectX flips it). The select is
				// safe_normalize: a zero vector would give NaN
				let up = vec3f( 0.0, 0.0, 1.0 );
				let raw = select( vec3f( 0.5, 0.5, 1.0 ), a.xyz, i0 != 255u ) * 2.0 - 1.0;
				let s = svmConst( c );
				let m = vec3f( raw.x * s, raw.y * svmConst( c + 1u ) * s, 1.0 + ( raw.z - 1.0 ) * saturate( s ) );
				let lm = length( m );
				r0 = vec4f( select( up, m / lm, lm > 0.0 ), 1.0 );

			}
			case ${ O.COMBINE_COLOR }u: {

				r0 = vec4f( select( 0.0, svmLum( a ), i0 != 255u ), select( 0.0, svmLum( b ), i1 != 255u ), select( 0.0, svmLum( f ), i2 != 255u ), 1.0 );

			}
			default: {}

		}

		if ( o0 != 255u ) { ( *regs )[ min( o0, ${ registers - 1 }u ) ] = r0; }
		if ( o1 != 255u ) { ( *regs )[ min( o1, ${ registers - 1 }u ) ] = r1; }
		if ( o2 != 255u ) { ( *regs )[ min( o2, ${ registers - 1 }u ) ] = r2; }

	}
	return count;

}
`;
export const SVM_RUN_SOURCE = svmRunSource( SVM_REGISTERS );

/** Every helper, in one list: the interpreter's dependencies minus the sampler and the register read. */
export const SVM_HELPER_FNS = [
	svmHashFn, svmSmoothFn, svmValueNoiseFn, svmFbmFn, svmRgbToHsvFn, svmHsvToRgbFn,
	svmBlendChannelFn, svmBlendFn, svmMathFn, svmLumFn, svmNoiseFacFn, svmNoiseColorFn,
	svmVoronoiFn, svmWaveFn, svmMagicFn, svmGradientFn, svmWhiteNoiseFn, svmMixFn,
];

/**
 * Packs compiled programs into the one array of words the interpreter reads: each
 * program's code, then its constants as float bits, with the constant of every
 * IMAGE instruction replaced by the entry's textureRef( slot ) as a RAW
 * word — the reference carries the atlas index and the wrap and filter bits, and
 * 2^31 does not survive as a float. Slots are numbered per program, so each entry
 * brings its own resolver. A missing entry packs as null.
 */
export function packSvmPrograms( entries ) {

	const words = [];
	const placements = [];
	const mixPlacements = [];
	const mediumPlacements = [];
	const bits = new Uint32Array( 1 );
	const float = new Float32Array( bits.buffer );

	// one program into the words, or null when there is none
	const place = ( program, textureRef ) => {

		if ( ! program ) return null;
		const codeWord = words.length;
		const count = program.code.length / 4;
		const refs = new Set();
		for ( let k = 0; k < count; k ++ ) {

			words.push( program.code[ k * 4 ] >>> 0, program.code[ k * 4 + 1 ] >>> 0, program.code[ k * 4 + 2 ] >>> 0, program.code[ k * 4 + 3 ] >>> 0 );
			const op = program.code[ k * 4 ] & 0xff;
			if ( op === O.IMAGE ) refs.add( program.code[ k * 4 + 3 ] );

		}

		const constWord = words.length;
		program.consts.forEach( ( value, i ) => {

			if ( refs.has( i ) ) {

				words.push( textureRef( value ) >>> 0 );

			} else {

				float[ 0 ] = value;
				words.push( bits[ 0 ] );

			}

		} );
		return { codeWord, count, constWord };

	};

	// a material may bring the program of its sockets and the program of the weights
	// of the chain it heads, each with its own texture slots
	for ( const entry of entries ) {

		placements.push( place( entry?.program, entry?.textureRef ) );
		mixPlacements.push( place( entry?.mixProgram, entry?.mixTextureRef ) );
		// and the density of a heterogeneous medium, run at points inside it
		mediumPlacements.push( place( entry?.mediumProgram, entry?.mediumTextureRef ) );

	}

	// the widest program, which sizes the register file of the kernels that run them
	const registers = Math.max( 0, ...entries.flatMap( entry => [
		entry?.program?.registers ?? 0, entry?.mixProgram?.registers ?? 0, entry?.mediumProgram?.registers ?? 0,
	] ) );
	return { words: Uint32Array.from( words ), placements, mixPlacements, mediumPlacements, registers };

}

/**
 * The interpreter node for a register file of "registers", bound to the includer's svmSample, svmWord and
 * svmUvMap - the last returns the uv of a channel at a triangle point and its derivatives along the default
 * uv, as the columns of a mat3x2f.
 */
export function svmRunFn( sampleFn, wordFn, uvMapFn, registers = SVM_REGISTERS ) {

	const constFn = wgslFn( SVM_CONST_SOURCE, [ wordFn ] );
	const regFn = wgslFn( svmRegSource( registers ), [] );
	return wgslFn( svmRunSource( registers ), [ ...SVM_HELPER_FNS, regFn, constFn, sampleFn, wordFn, uvMapFn ] );

}

/** Everything but svmSample, svmWord and svmUvMap, as text: what a probe pastes before its own three. */
export const SVM_SOURCE = [
	SVM_CONST_SOURCE,
	SVM_HASH_SOURCE, SVM_SMOOTH_SOURCE, SVM_VALUE_NOISE_SOURCE, SVM_FBM_SOURCE,
	SVM_RGB_TO_HSV_SOURCE, SVM_HSV_TO_RGB_SOURCE, SVM_BLEND_CHANNEL_SOURCE, SVM_BLEND_SOURCE,
	SVM_MATH_SOURCE, SVM_LUM_SOURCE, SVM_NOISE_FAC_SOURCE, SVM_NOISE_COLOR_SOURCE,
	SVM_VORONOI_SOURCE, SVM_WAVE_SOURCE, SVM_MAGIC_SOURCE, SVM_GRADIENT_SOURCE,
	SVM_WHITE_NOISE_SOURCE, SVM_MIX_SOURCE, SVM_REG_SOURCE, SVM_RUN_SOURCE,
].join( '\n' );
