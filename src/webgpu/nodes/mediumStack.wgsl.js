import { wgslFn } from 'three/tsl';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';
import { MEDIUM_MAJORANT_FLOOR, MEDIUM_MAX_STEPS, mediumRandFn } from './medium.wgsl.js';

// THE MEDIUM STACK (Cycles kernel/integrator/volume_stack.h): the media a path is inside of, in a
// vec4u - one word an entry, the material in the low 16 bits and the object slot in the high ones,
// the form the shadow rays already carried. Four entries and not the 32 Cycles sizes on the scene:
// the stack travels in every slot and in every queued shadow ray, so its size is fixed, and an
// entry past the fourth is ignored, as Cycles ignores one past a full stack. Entries are packed at
// the front, the empty word after them.
export const MEDIUM_STACK_EMPTY = 0xffffffff;
export const MEDIUM_STACK_SIZE = 4;

// A path crosses a medium's boundary from the FRONT: its entry goes on the stack, unless it is
// there already or the stack is full (volume_stack_enter_exit, the enter branch)
export const mediumStackEnterFn = wgslFn( /* wgsl */ `

	fn mediumStackEnter( stack: vec4u, word: u32 ) -> vec4u {

		var s = stack;
		for ( var i = 0u; i < 4u; i ++ ) {

			if ( s[ i ] == word ) { return s; }
			if ( s[ i ] == 0xffffffffu ) {

				s[ i ] = word;
				return s;

			}

		}
		return s;

	}

` );

// ... and from the BACK: its entry leaves, and the ones after it move up (the exit branch)
export const mediumStackExitFn = wgslFn( /* wgsl */ `

	fn mediumStackExit( stack: vec4u, word: u32 ) -> vec4u {

		var s = stack;
		var found = false;
		for ( var i = 0u; i < 4u; i ++ ) {

			if ( ! found && s[ i ] == word ) { found = true; }
			if ( found ) { s[ i ] = select( 0xffffffffu, s[ min( i + 1u, 3u ) ], i < 3u ); }

		}
		return s;

	}

` );

// What the stack holds, from the records of its entries: x is the OR of their flags (bit 1 a
// medium, bit 2 a heterogeneous one), y the Volume Sampling of the stack by the rule of
// volume_stack_sample_method - 0 distance, 1 equiangular, 2 multiple importance: one entry asking
// for MIS gives MIS, and distance and equiangular together give MIS too
export const mediumStackFlagsFunc = ( materials ) => wgslTagFn/* wgsl */`

	fn mediumStackFlags( stack: vec4u ) -> vec2u {

		var flags = 0u;
		var method = 3u;
		for ( var i = 0u; i < 4u; i ++ ) {

			if ( stack[ i ] == 0xffffffffu ) { break; }
			let f = ${ materials }[ stack[ i ] & 0xffffu ].mediumFlags;
			flags |= f;
			var own = 2u;
			if ( ( f & 8u ) != 0u ) { own = 0u; }
			if ( ( f & 16u ) != 0u ) { own = 1u; }
			if ( method == 3u ) { method = own; }
			else if ( method != own ) { method = 2u; }

		}
		return vec2u( flags, select( method, 0u, method == 3u ) );

	}

`;

// The coefficients of a stack of HOMOGENEOUS media, summed as volume_shader_eval accumulates the
// closures of every entry: the columns are sigma_s, sigma_t and the emission
export const mediumStackCoefficientsFunc = ( materials ) => wgslTagFn/* wgsl */`

	fn mediumStackCoefficients( stack: vec4u ) -> mat3x3f {

		var sigmaS = vec3f( 0.0 );
		var sigmaT = vec3f( 0.0 );
		var emission = vec3f( 0.0 );
		for ( var i = 0u; i < 4u; i ++ ) {

			if ( stack[ i ] == 0xffffffffu ) { break; }
			let m = stack[ i ] & 0xffffu;
			let s = vec3f( ${ materials }[ m ].mediumScatterR, ${ materials }[ m ].mediumScatterG, ${ materials }[ m ].mediumScatterB );
			sigmaS += s;
			sigmaT += s + vec3f( ${ materials }[ m ].mediumAbsorptionR, ${ materials }[ m ].mediumAbsorptionG, ${ materials }[ m ].mediumAbsorptionB );
			emission += vec3f( ${ materials }[ m ].mediumEmissionR, ${ materials }[ m ].mediumEmissionG, ${ materials }[ m ].mediumEmissionB );

		}
		return mat3x3f( sigmaS, sigmaT, emission );

	}

`;

// The weights of the phases of a homogeneous stack: each entry's scattering, on its luminance.
// Cycles picks a phase closure on its sample weight; the mixture of the Henyey-Greensteins of the
// entries, weighed so, is the phase of the point
export const mediumStackPhaseWeightsFunc = ( materials ) => wgslTagFn/* wgsl */`

	fn mediumStackPhaseWeights( stack: vec4u ) -> vec4f {

		var weights = vec4f( 0.0 );
		for ( var i = 0u; i < 4u; i ++ ) {

			if ( stack[ i ] == 0xffffffffu ) { break; }
			let m = stack[ i ] & 0xffffu;
			weights[ i ] = dot( vec3f( ${ materials }[ m ].mediumScatterR, ${ materials }[ m ].mediumScatterG, ${ materials }[ m ].mediumScatterB ), vec3f( 0.2126, 0.7152, 0.0722 ) );

		}
		return weights;

	}

`;

// The phase of the point: the mixture of the entries' Henyey-Greensteins by the weights. It is the
// value AND the pdf, since a mixture is sampled by picking an entry on its weight
export const mediumStackPhaseFunc = ( materials, henyeyGreenstein ) => wgslTagFn/* wgsl */`

	fn mediumStackPhase( stack: vec4u, weights: vec4f, wo: vec3f, wi: vec3f ) -> f32 {

		var value = 0.0;
		var total = 0.0;
		for ( var i = 0u; i < 4u; i ++ ) {

			if ( stack[ i ] == 0xffffffffu ) { break; }
			if ( weights[ i ] <= 0.0 ) { continue; }
			value += weights[ i ] * ${ henyeyGreenstein }( wo, wi, ${ materials }[ stack[ i ] & 0xffffu ].mediumAnisotropy );
			total += weights[ i ];

		}
		return select( 0.0, value / total, total > 0.0 );

	}

`;

// The entry a scatter takes its phase from, picked on the weights with one number in [0, 1); its
// anisotropy is what the direction is drawn with
export const mediumStackPickPhaseFunc = ( materials ) => wgslTagFn/* wgsl */`

	fn mediumStackPickPhase( stack: vec4u, weights: vec4f, u: f32 ) -> f32 {

		let total = weights.x + weights.y + weights.z + weights.w;
		var pick = u * total;
		var g = ${ materials }[ stack.x & 0xffffu ].mediumAnisotropy;
		for ( var i = 0u; i < 4u; i ++ ) {

			if ( stack[ i ] == 0xffffffffu ) { break; }
			if ( weights[ i ] <= 0.0 ) { continue; }
			g = ${ materials }[ stack[ i ] & 0xffffu ].mediumAnisotropy;
			if ( pick < weights[ i ] ) { break; }
			pick -= weights[ i ];

		}
		return g;

	}

`;

// THE STACK AT A POINT, heterogeneous entries included: each one's density from its program, the
// homogeneous ones at the coefficients of their record. The columns are sigma_s, sigma_t and the
// emission; "weights" receives each entry's scattering on its luminance, for the phase. The loop
// runs to a count read from the DATA: with a constant bound FXC, the HLSL compiler under D3D,
// unrolls it, and every copy holds the interpreter of the density programs
export const mediumStackPointFunc = ( materials, pointFn ) => wgslTagFn/* wgsl */`

	fn mediumStackPoint( stack: vec4u, point: vec3f, weights: ptr<function, vec4f> ) -> mat3x3f {

		var sigmaS = vec3f( 0.0 );
		var sigmaT = vec3f( 0.0 );
		var emission = vec3f( 0.0 );
		var phase = vec4f( 0.0 );
		let count = u32( stack.x != 0xffffffffu ) + u32( stack.y != 0xffffffffu ) + u32( stack.z != 0xffffffffu ) + u32( stack.w != 0xffffffffu );
		for ( var i = 0u; i < count; i ++ ) {

			// the entry by SELECTION, not by a dynamic index: FXC failed (E_FAIL) on the kernel
			// that indexed a vector and wrote through a pointer by the loop counter
			let lane = vec4f( f32( i == 0u ), f32( i == 1u ), f32( i == 2u ), f32( i == 3u ) );
			let word = select( select( select( stack.w, stack.z, i == 2u ), stack.y, i == 1u ), stack.x, i == 0u );
			let m = word & 0xffffu;
			let s = vec3f( ${ materials }[ m ].mediumScatterR, ${ materials }[ m ].mediumScatterG, ${ materials }[ m ].mediumScatterB );
			let a = vec3f( ${ materials }[ m ].mediumAbsorptionR, ${ materials }[ m ].mediumAbsorptionG, ${ materials }[ m ].mediumAbsorptionB );
			var density = 1.0;
			if ( ( ${ materials }[ m ].mediumFlags & 4u ) != 0u ) {

				let sample = ${ pointFn }( m, ( word >> 16u ) & 0xffffu, point );
				density = sample.w;
				emission += sample.xyz;

			} else {

				emission += vec3f( ${ materials }[ m ].mediumEmissionR, ${ materials }[ m ].mediumEmissionG, ${ materials }[ m ].mediumEmissionB );

			}
			sigmaS += s * density;
			sigmaT += ( s + a ) * density;
			phase += lane * dot( s * density, vec3f( 0.2126, 0.7152, 0.0722 ) );

		}
		*weights = phase;
		return mat3x3f( sigmaS, sigmaT, emission );

	}

`;

// The majorant of the stack: the sum of the entries', each the largest channel of its extinction
// times its density bound - one for a homogeneous entry, and for a heterogeneous one the bound the
// HOST measured on a grid (densityBound in the app, which writes one whenever a program drives the
// medium). Not the four-sample estimate of a single medium, which would run the interpreter four
// times per entry for a number the record already holds. A medium without a measured bound takes
// the floor, and weighted tracking stays unbiased - it pays in noise
export const mediumStackMajorantFunc = ( materials ) => wgslTagFn/* wgsl */`

	fn mediumStackMajorant( stack: vec4u ) -> f32 {

		var sigmaBar = 0.0;
		for ( var i = 0u; i < 4u; i ++ ) {

			if ( stack[ i ] == 0xffffffffu ) { break; }
			let m = stack[ i ] & 0xffffu;
			let unit = vec3f(
				${ materials }[ m ].mediumScatterR + ${ materials }[ m ].mediumAbsorptionR,
				${ materials }[ m ].mediumScatterG + ${ materials }[ m ].mediumAbsorptionG,
				${ materials }[ m ].mediumScatterB + ${ materials }[ m ].mediumAbsorptionB,
			);
			var bound = 1.0;
			if ( ( ${ materials }[ m ].mediumFlags & 4u ) != 0u ) {

				bound = max( ${ materials }[ m ].mediumDensityMax, ${ MEDIUM_MAJORANT_FLOOR } );

			}
			sigmaBar += max( unit.r, max( unit.g, unit.b ) ) * bound;

		}
		return sigmaBar;

	}

`;

// THE TRANSMITTANCE of a stretch through the stack: in closed form when every entry is
// homogeneous, by RATIO TRACKING on the summed extinction when one is not (Cycles
// volume_transmittance with null scattering)
export const mediumStackTransmittanceFunc = ( materials, coefficientsFn, stackPointFn, stackMajorantFn ) => wgslTagFn/* wgsl */`

	fn mediumStackTransmittance( stack: vec4u, origin: vec3f, direction: vec3f, len: f32, rng: ptr<function, u32> ) -> vec3f {

		if ( stack.x == 0xffffffffu ) { return vec3f( 1.0 ); }
		var heterogeneous = false;
		for ( var i = 0u; i < 4u; i ++ ) {

			if ( stack[ i ] == 0xffffffffu ) { break; }
			heterogeneous = heterogeneous || ( ${ materials }[ stack[ i ] & 0xffffu ].mediumFlags & 4u ) != 0u;

		}
		if ( ! heterogeneous ) {

			return exp( - ${ coefficientsFn }( stack )[ 1 ] * len );

		}

		let sigmaBar = ${ stackMajorantFn }( stack );
		if ( sigmaBar <= 0.0 ) { return vec3f( 1.0 ); }
		var transmittance = vec3f( 1.0 );
		var weights = vec4f( 0.0 );
		var t = 0.0;
		for ( var k = 0u; k < ${ MEDIUM_MAX_STEPS }u; k ++ ) {

			t -= log( max( 1.0 - ${ mediumRandFn }( rng ), 1e-9 ) ) / sigmaBar;
			if ( t >= len ) { break; }
			let point = ${ stackPointFn }( stack, origin + direction * t, &weights );
			transmittance *= 1.0 - point[ 1 ] / sigmaBar;
			if ( all( abs( transmittance ) < vec3f( 1e-5 ) ) ) { break; }

		}
		return transmittance;

	}

`;
