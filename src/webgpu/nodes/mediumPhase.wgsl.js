// THIRD PARTY: contains code derived from Cycles (Apache-2.0), modified. Notices and license: NOTICE
import { wgslFn } from 'three/tsl';

// THE PHASES OF A VOLUME SCATTER, as Cycles 5.2 writes them (kernel/closure/volume_util.h): Henyey-Greenstein,
// Rayleigh, Draine and Mie - four of its five, without Fournier-Forand. The kind is in bits 12-13 of mediumFlags in the
// order of the app's MEDIUM_PHASES (0 HG, 1 Rayleigh, 2 Draine, 3 Mie); the g is mediumAnisotropy, and the second
// parameter mediumPhaseParam - the alpha of a Draine, the droplet diameter of a Mie, in micrometres.
//
// "cosTheta" is between the TRAVEL direction and the outgoing one, as Cycles' phase functions take it (D = -wi): a
// positive g peaks forward. Each function is a wgslFn of its own whose source starts with "fn" - three's parser
// refuses anything else - and the comments stay out of the sources.

// x^(-1/3) with its sign, Cycles' fast_inv_cbrtf
export const MEDIUM_PHASE_INV_CBRT_SOURCE = /* wgsl */ `
fn mediumInvCbrt( x: f32 ) -> f32 {

	return sign( x ) * pow( max( abs( x ), 1e-30 ), - 1.0 / 3.0 );

}
`;
export const mediumInvCbrtFn = wgslFn( MEDIUM_PHASE_INV_CBRT_SOURCE );

// phase_henyey_greenstein: uniform below |g| = 1e-3
export const MEDIUM_PHASE_HG_SOURCE = /* wgsl */ `
fn mediumPhaseHg( cosTheta: f32, g: f32 ) -> f32 {

	if ( abs( g ) < 1e-3 ) { return 0.07957747154594767; }
	let fac = 1.0 + g * ( g - 2.0 * cosTheta );
	return ( 1.0 - g * g ) / ( 12.566370614359172 * fac * sqrt( max( fac, 0.0 ) ) );

}
`;
export const mediumPhaseHgFn = wgslFn( MEDIUM_PHASE_HG_SOURCE );

// phase_rayleigh: 3 / (16 pi) (1 + cos^2)
export const MEDIUM_PHASE_RAYLEIGH_SOURCE = /* wgsl */ `
fn mediumPhaseRayleigh( cosTheta: f32 ) -> f32 {

	return 0.05968310365946075 * ( 1.0 + cosTheta * cosTheta );

}
`;
export const mediumPhaseRayleighFn = wgslFn( MEDIUM_PHASE_RAYLEIGH_SOURCE );

// phase_draine: HG at alpha = 0, Rayleigh at g = 0 and alpha = 1
export const MEDIUM_PHASE_DRAINE_SOURCE = /* wgsl */ `
fn mediumPhaseDraine( cosTheta: f32, g: f32, alpha: f32 ) -> f32 {

	if ( abs( g ) < 1e-3 && alpha > 0.999 ) { return mediumPhaseRayleigh( cosTheta ); }
	if ( abs( alpha ) < 1e-3 ) { return mediumPhaseHg( cosTheta, g ); }
	let g2 = g * g;
	let fac = 1.0 + g2 - 2.0 * g * cosTheta;
	return ( ( 1.0 - g2 ) * ( 1.0 + alpha * cosTheta * cosTheta ) )
		/ ( ( 1.0 + ( alpha * ( 1.0 + 2.0 * g2 ) ) * ( 1.0 / 3.0 ) ) * 12.566370614359172 * fac * sqrt( max( fac, 0.0 ) ) );

}
`;
export const mediumPhaseDraineFn = wgslFn( MEDIUM_PHASE_DRAINE_SOURCE, [ mediumPhaseRayleighFn, mediumPhaseHgFn ] );

// phase_mie_fitted_parameters: the mixture that approximates the Mie phase of water droplets of diameter d (in
// micrometres), from NVIDIA's approximate Mie: x the g of the HG lobe, y the g of the Draine, z its alpha, w the weight
// of the Draine (the HG weighs 1 - w)
export const MEDIUM_PHASE_MIE_PARAMETERS_SOURCE = /* wgsl */ `
fn mediumMieParameters( diameter: f32 ) -> vec4f {

	let d = max( diameter, 0.0 );
	if ( d <= 0.1 ) {

		return vec4f( 13.8 * d * d, 1.1456 * d * sin( 9.29044 * d ), 250.0, 0.252977 - 312.983 * pow( d, 4.3 ) );

	}
	if ( d < 1.5 ) {

		let logD = log( d );
		let a = ( logD - 0.238604 ) * ( logD + 1.00667 );
		let b = 0.507522 - 0.15677 * logD;
		let c = 1.19692 * cos( a / b ) + 1.37932 * logD + 0.0625835;
		return vec4f(
			0.862 - 0.143 * logD * logD, 0.379685 * cos( c ) + 0.344213, 250.0,
			0.146209 * cos( 3.38707 * logD + 2.11193 ) + 0.316072 + 0.0778917 * logD,
		);

	}
	if ( d < 5.0 ) {

		let logD = log( d );
		let temp = cos( 5.68947 * ( log( logD ) - 0.0292149 ) );
		return vec4f(
			0.0604931 * log( logD ) + 0.940256, 0.500411 - ( 0.081287 / ( - 2.0 * logD + tan( logD ) + 1.27551 ) ),
			7.30354 * logD + 6.31675, 0.026914 * ( logD - temp ) + 0.3764,
		);

	}
	return vec4f(
		exp( - 0.0990567 / ( d - 1.67154 ) ), exp( - 2.20679 / ( d + 3.91029 ) - 0.428934 ),
		exp( 3.62489 - 8.29288 / ( d + 5.52825 ) ), exp( - 0.599085 / ( d - 0.641583 ) - 0.665888 ),
	);

}
`;
export const mediumMieParametersFn = wgslFn( MEDIUM_PHASE_MIE_PARAMETERS_SOURCE );

// the phase of one entry of the medium stack, per steradian, for light travelling along "direction" and leaving along
// "outgoing"
export const MEDIUM_PHASE_EVAL_SOURCE = /* wgsl */ `
fn mediumPhase( kind: u32, g: f32, param: f32, direction: vec3f, outgoing: vec3f ) -> f32 {

	let c = dot( direction, outgoing );
	if ( kind == 1u ) { return mediumPhaseRayleigh( c ); }
	if ( kind == 2u ) { return mediumPhaseDraine( c, g, param ); }
	if ( kind == 3u ) {

		let mie = mediumMieParameters( param );
		return ( 1.0 - mie.w ) * mediumPhaseHg( c, mie.x ) + mie.w * mediumPhaseDraine( c, mie.y, mie.z );

	}
	return mediumPhaseHg( c, g );

}
`;
export const mediumPhaseFn = wgslFn( MEDIUM_PHASE_EVAL_SOURCE, [ mediumPhaseHgFn, mediumPhaseRayleighFn, mediumPhaseDraineFn, mediumMieParametersFn ] );

// phase_rayleigh_sample, the cosine
export const MEDIUM_PHASE_RAYLEIGH_COS_SOURCE = /* wgsl */ `
fn mediumRayleighCos( u: f32 ) -> f32 {

	let a = 2.0 - 4.0 * u;
	let invU = - mediumInvCbrt( sqrt( 1.0 + a * a ) + a );
	return 1.0 / invU - invU;

}
`;
export const mediumRayleighCosFn = wgslFn( MEDIUM_PHASE_RAYLEIGH_COS_SOURCE, [ mediumInvCbrtFn ] );

// phase_henyey_greenstein_sample, the cosine
export const MEDIUM_PHASE_HG_COS_SOURCE = /* wgsl */ `
fn mediumHgCos( g: f32, u: f32 ) -> f32 {

	var c = 1.0 - 2.0 * u;
	if ( abs( g ) >= 1e-3 ) {

		let k = ( 1.0 - g * g ) / ( 1.0 - g * c );
		c = ( 1.0 + g * g - k * k ) / ( 2.0 * g );

	}
	return c;

}
`;
export const mediumHgCosFn = wgslFn( MEDIUM_PHASE_HG_COS_SOURCE );

// phase_draine_sample_cos, with the special cases of phase_draine_sample
export const MEDIUM_PHASE_DRAINE_COS_SOURCE = /* wgsl */ `
fn mediumDraineCos( g: f32, alpha: f32, u: f32 ) -> f32 {

	if ( abs( g ) < 1e-3 && alpha > 0.999 ) { return mediumRayleighCos( u ); }
	if ( abs( alpha ) < 1e-3 ) { return mediumHgCos( g, u ); }
	if ( abs( g ) < 1e-2 ) {

		let invAlpha = 1.0 / alpha;
		let b2 = ( 3.0 + alpha ) * invAlpha * ( 0.5 - u );
		let invU = - mediumInvCbrt( b2 + sqrt( b2 * b2 + invAlpha * invAlpha * invAlpha ) );
		return 1.0 / invU - invU / alpha;

	}
	let cbrt2 = 1.2599210498948732;
	let g2 = g * g;
	let g3 = g * g2;
	let g4 = g2 * g2;
	let g6 = g2 * g4;
	let pgp1_2 = ( 1.0 + g2 ) * ( 1.0 + g2 );
	let t1a = alpha * ( g4 - 1.0 );
	let t1a3 = t1a * t1a * t1a;
	let t2 = - 1296.0 * ( g2 - 1.0 ) * ( alpha - alpha * g2 ) * t1a * ( 4.0 * g2 + alpha * pgp1_2 );
	let t9 = 2.0 + g2 + g3 * ( 1.0 + 2.0 * g2 ) * ( 2.0 * u - 1.0 );
	let t3 = 3.0 * g2 * ( 1.0 + g * ( 2.0 * u - 1.0 ) ) + alpha * t9;
	let t4a = 432.0 * t1a3 + t2 + 432.0 * ( alpha * ( 1.0 - g2 ) ) * t3 * t3;
	let t10 = alpha * ( 2.0 * g4 - g2 - g6 );
	let t4b = 144.0 * t10;
	let t4b3 = t4b * t4b * t4b;
	let t4 = t4a + sqrt( max( - 4.0 * t4b3 + t4a * t4a, 0.0 ) );
	let invT4p3 = mediumInvCbrt( t4 );
	let t8 = 48.0 * cbrt2 * t10;
	let t6 = ( 2.0 * t1a + t8 * invT4p3 + 1.0 / ( 3.0 * cbrt2 * invT4p3 ) ) / ( alpha * ( 1.0 - g2 ) );
	let t5 = 6.0 * ( 1.0 + g2 ) + t6;
	let t7 = 6.0 * ( 1.0 + g2 ) - ( 8.0 * t3 ) / ( alpha * ( g2 - 1.0 ) * sqrt( max( t5, 0.0 ) ) ) - t6;
	let r = sqrt( max( t7, 0.0 ) ) - sqrt( max( t5, 0.0 ) );
	return ( 1.0 + g2 - 0.25 * r * r ) / ( 2.0 * g );

}
`;
export const mediumDraineCosFn = wgslFn( MEDIUM_PHASE_DRAINE_COS_SOURCE, [ mediumRayleighCosFn, mediumHgCosFn, mediumInvCbrtFn ] );

// a direction drawn from the phase of kind 1, 2 or 3 around the TRAVEL direction (phase_sample_direction); the HG kind is
// drawn by the caller's sampleHenyeyGreenstein, as before. A Mie picks its lobe on u.x and reuses what is left of it
export const MEDIUM_PHASE_SAMPLE_SOURCE = /* wgsl */ `
fn mediumPhaseSample( kind: u32, g: f32, param: f32, direction: vec3f, u: vec2f ) -> vec3f {

	var c = 0.0;
	if ( kind == 1u ) {

		c = mediumRayleighCos( u.x );

	} else if ( kind == 2u ) {

		c = mediumDraineCos( g, param, u.x );

	} else {

		let mie = mediumMieParameters( param );
		if ( u.x < mie.w ) {

			c = mediumDraineCos( mie.y, mie.z, u.x / max( mie.w, 1e-6 ) );

		} else {

			c = mediumHgCos( mie.x, ( u.x - mie.w ) / max( 1.0 - mie.w, 1e-6 ) );

		}

	}
	c = clamp( c, - 1.0, 1.0 );
	let s = sqrt( max( 0.0, 1.0 - c * c ) );
	let phi = 6.283185307179586 * u.y;
	let n = direction;
	let a = select( vec3f( 1.0, 0.0, 0.0 ), vec3f( 0.0, 1.0, 0.0 ), abs( n.x ) > 0.9 );
	let t = normalize( cross( a, n ) );
	let b = cross( n, t );
	return normalize( t * ( s * cos( phi ) ) + b * ( s * sin( phi ) ) + n * c );

}
`;
export const mediumPhaseSampleFn = wgslFn( MEDIUM_PHASE_SAMPLE_SOURCE, [ mediumRayleighCosFn, mediumDraineCosFn, mediumHgCosFn, mediumMieParametersFn ] );

// Every source as text, in an order where each comes after what it calls: what a probe pastes in a shader of its own
export const MEDIUM_PHASE_SOURCE = [
	MEDIUM_PHASE_INV_CBRT_SOURCE,
	MEDIUM_PHASE_HG_SOURCE,
	MEDIUM_PHASE_RAYLEIGH_SOURCE,
	MEDIUM_PHASE_DRAINE_SOURCE,
	MEDIUM_PHASE_MIE_PARAMETERS_SOURCE,
	MEDIUM_PHASE_EVAL_SOURCE,
	MEDIUM_PHASE_RAYLEIGH_COS_SOURCE,
	MEDIUM_PHASE_HG_COS_SOURCE,
	MEDIUM_PHASE_DRAINE_COS_SOURCE,
	MEDIUM_PHASE_SAMPLE_SOURCE,
].join( String.fromCharCode( 10 ) );
