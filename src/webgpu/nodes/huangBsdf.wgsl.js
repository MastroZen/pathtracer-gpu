// IL BSDF DEI PELI DI HUANG, IN WGSL: la fibra come microfacet, con sezione
// ellittica. Lo stesso di `src/hair/huangBsdf.ts` dell'applicazione.
//
// Il modello, le divergenze dichiarate (farfield sempre) e da dove viene stanno
// scritti nel gemello: qui ci sono solo le note che riguardano il WGSL.
//
// ── IL GENERATORE VIAGGIA PER PUNTATORE ──
//
// Huang consuma sei-dieci numeri casuali per valutazione — una o due
// micro-normali per ogni punto della sezione — e quei numeri non si possono
// passare come argomenti. E' lo stesso LCG di Cycles (`lcg_step_float`), con lo
// stato in una variabile di funzione che si passa per puntatore.
//
// ── UNA FUNZIONE PER BLOCCO, ED E' IL VINCOLO DI TSL ──
//
// `wgslFn` analizza UNA funzione per volta. Le sorgenti restano esportate
// accanto ai nodi perche' le sonde compilano WGSL a mano.
import { wgslFn } from 'three/tsl';
import { StructTypeNode } from 'three/webgpu';
import { ggxGlassEFn } from './ggxGlassTable.wgsl.js';
import { hairLongitudinalFn } from './hairBsdf.wgsl.js';

/** Il materiale, gia' preparato. */
export const huangHairStruct = new StructTypeNode( {
	sigma: 'vec3f',
	roughness: 'float',
	tilt: 'float',
	eta: 'float',
	aspectRatio: 'float',
	r: 'float',
	tt: 'float',
	trt: 'float',
}, 'HuangHair' );

/** Il frame della fibra e quel che dipende dalla vista. */
export const huangFrameStruct = new StructTypeNode( {
	x: 'vec3f',
	y: 'vec3f',
	z: 'vec3f',
	wi: 'vec3f',
	e2: 'float',
	radius: 'float',
	h: 'float',
	valid: 'uint',
}, 'HuangFrame' );

/** Quel che il campionamento restituisce. */
export const huangScatterStruct = new StructTypeNode( {
	f: 'vec3f',
	direction: 'vec3f',
	valid: 'uint',
}, 'HuangScatter' );

export const HUANG_STRUCTS_SOURCE = /* wgsl */ `
struct HuangHair {
	sigma: vec3f,
	roughness: f32,
	tilt: f32,
	eta: f32,
	aspectRatio: f32,
	r: f32,
	tt: f32,
	trt: f32,
}

struct HuangFrame {
	x: vec3f,
	y: vec3f,
	z: vec3f,
	wi: vec3f,
	e2: f32,
	radius: f32,
	h: f32,
	valid: u32,
}

struct HuangScatter {
	f: vec3f,
	direction: vec3f,
	valid: u32,
}
`;

/** Il generatore di Cycles: uno stato a 31 bit, un numero per passo. */
export const HUANG_LCG_SOURCE = /* wgsl */ `
fn huangLcg( state: ptr<function, u32> ) -> f32 {

	// SENZA MASCHERA, e non per semplicita': con la maschera sul bit alto il
	// risultato usciva SOPRA UNO attraverso il puntatore — 1,597 invece di 0,597,
	// cioe' il valore non mascherato — mentre le stesse istruzioni scritte in
	// linea davano il numero giusto. Invece di inseguire perche', si e' tolta:
	// l'intero pieno diviso due alla trentaduesima e' un generatore altrettanto
	// buono, e il gemello in Node fa lo stesso conto.
	//
	// Il segnale che erano DUE CAMMINI diversi e non due numeri diversi: il lobo
	// R, che non usa il generatore, concordava a 3,5e-5 mentre il resto era
	// completamente scorrelato.
	var s = *state;
	s = s * 1103515245u + 12345u;
	*state = s;
	return f32( s ) * 2.3283064365386963e-10;

}
`;
export const huangLcgFn = wgslFn( HUANG_LCG_SOURCE, [] );

// ── LE COORDINATE DELLA FIBRA ──
//
// Y e' l'asse del pelo, quindi il seno di theta e' la componente Y e il coseno e'
// la lunghezza della proiezione sul piano normale.
export const HUANG_ANGLES_SOURCE = /* wgsl */ `
fn huangCosTheta( w: vec3f ) -> f32 { return sqrt( max( 0.0, w.x * w.x + w.z * w.z ) ); }
`;
export const huangCosThetaFn = wgslFn( HUANG_ANGLES_SOURCE, [] );

export const HUANG_TAN_THETA_SOURCE = /* wgsl */ `
fn huangTanTheta( w: vec3f ) -> f32 { return w.y / max( huangCosTheta( w ), 1e-9 ); }
`;
export const huangTanThetaFn = wgslFn( HUANG_TAN_THETA_SOURCE, [ huangCosThetaFn ] );

export const HUANG_DIR_PHI_SOURCE = /* wgsl */ `
fn huangDirPhi( w: vec3f ) -> f32 { return atan2( w.x, w.z ); }
`;
export const huangDirPhiFn = wgslFn( HUANG_DIR_PHI_SOURCE, [] );

export const HUANG_SINCOS_PHI_SOURCE = /* wgsl */ `
fn huangSincosPhi( w: vec3f ) -> vec2f {

	let c = huangCosTheta( w );
	if ( c <= 0.0 ) { return vec2f( 0.0, 1.0 ); }
	return vec2f( w.x / c, w.z / c );

}
`;
export const huangSincosPhiFn = wgslFn( HUANG_SINCOS_PHI_SOURCE, [ huangCosThetaFn ] );

/** Da gamma (l'angolo sul cerchio) a phi (quello sull'ellisse), e ritorno. */
export const HUANG_TO_PHI_SOURCE = /* wgsl */ `
fn huangToPhi( gamma: f32, b: f32 ) -> f32 {

	if ( b == 1.0 ) { return gamma; }
	return atan2( b * sin( gamma ), cos( gamma ) );

}
`;
export const huangToPhiFn = wgslFn( HUANG_TO_PHI_SOURCE, [] );

export const HUANG_TO_GAMMA_SOURCE = /* wgsl */ `
fn huangToGamma( phi: f32, b: f32 ) -> f32 {

	if ( b == 1.0 ) { return phi; }
	return atan2( sin( phi ), b * cos( phi ) );

}
`;
export const huangToGammaFn = wgslFn( HUANG_TO_GAMMA_SOURCE, [] );

/** Dove il raggio taglia l'ellisse, per un dato phi. */
export const HUANG_PHI_TO_H_SOURCE = /* wgsl */ `
fn huangPhiToH( phi: f32, b: f32, wi: vec3f ) -> f32 {

	if ( b == 1.0 ) { return - sin( phi ); }
	let g = huangToGamma( phi, b );
	let sc = huangSincosPhi( wi );
	return - sc.y * sin( g ) + b * sc.x * cos( g );

}
`;
export const huangPhiToHFn = wgslFn( HUANG_PHI_TO_H_SOURCE, [ huangToGammaFn, huangSincosPhiFn ] );

/** E l'inverso: da h (gia' diviso per il raggio proiettato) a gamma. */
export const HUANG_H_TO_GAMMA_SOURCE = /* wgsl */ `
fn huangHToGamma( hDivR: f32, b: f32, wi: vec3f ) -> f32 {

	if ( b == 1.0 ) { return - asin( clamp( hDivR, -1.0, 1.0 ) ); }
	return atan2( wi.z, - b * wi.x ) - acos( clamp( - hDivR, -1.0, 1.0 ) );

}
`;
export const huangHToGammaFn = wgslFn( HUANG_H_TO_GAMMA_SOURCE, [] );

/** Lo jacobiano del cambio di variabile da h a gamma. */
export const HUANG_DGAMMA_SOURCE = /* wgsl */ `
fn huangDGammaDH( sc: vec2f, gamma: f32, b: f32 ) -> f32 {

	if ( b == 1.0 ) {

		let c = cos( gamma );
		if ( c == 0.0 ) { return 0.0; }
		return 1.0 / c;

	}
	let d = sc.y * cos( gamma ) + b * sc.x * sin( gamma );
	if ( d == 0.0 ) { return 0.0; }
	return 1.0 / d;

}
`;
export const huangDGammaDHFn = wgslFn( HUANG_DGAMMA_SOURCE, [] );

export const HUANG_TO_POINT_SOURCE = /* wgsl */ `
fn huangToPoint( gamma: f32, b: f32 ) -> vec2f { return vec2f( sin( gamma ), b * cos( gamma ) ); }
`;
export const huangToPointFn = wgslFn( HUANG_TO_POINT_SOURCE, [] );

/** La direzione data da theta e gamma, tenendo conto dell'ellisse. */
export const HUANG_SPHG_SOURCE = /* wgsl */ `
fn huangSphgDir( theta: f32, gamma: f32, b: f32 ) -> vec3f {

	let st = sin( theta );
	let ct = cos( theta );
	let sg = sin( gamma );
	let cg = cos( gamma );
	var sp = sg;
	var cp = cg;
	if ( b != 1.0 && abs( cg ) >= 1e-6 ) {

		let tanPhi = b * ( sg / cg );
		cp = sign( cg ) * inverseSqrt( tanPhi * tanPhi + 1.0 );
		sp = cp * tanPhi;

	}
	return vec3f( sp * ct, st, cp * ct );

}
`;
export const huangSphgDirFn = wgslFn( HUANG_SPHG_SOURCE, [] );

export const HUANG_ARC_SOURCE = /* wgsl */ `
fn huangArcLength( e2: f32, gamma: f32 ) -> f32 {

	if ( e2 == 0.0 ) { return 1.0; }
	let s = sin( gamma );
	return sqrt( max( 0.0, 1.0 - e2 * s * s ) );

}
`;
export const huangArcLengthFn = wgslFn( HUANG_ARC_SOURCE, [] );

// ── IL MICROFACET GGX ──
export const HUANG_LAMBDA_SOURCE = /* wgsl */ `
fn huangLambda( alpha2: f32, cosN: f32 ) -> f32 {

	let t = alpha2 * max( 1.0 / ( cosN * cosN ) - 1.0, 0.0 );
	return 0.5 * ( sqrt( 1.0 + t ) - 1.0 );

}
`;
export const huangLambdaFn = wgslFn( HUANG_LAMBDA_SOURCE, [] );

export const HUANG_G2_SOURCE = /* wgsl */ `
fn huangG2( alpha2: f32, ci: f32, co: f32 ) -> f32 {

	return 1.0 / ( 1.0 + huangLambda( alpha2, ci ) + huangLambda( alpha2, co ) );

}
`;
export const huangG2Fn = wgslFn( HUANG_G2_SOURCE, [ huangLambdaFn ] );

/** L'ombreggiamento combinato diviso quello della direzione entrante. */
export const HUANG_GO_SOURCE = /* wgsl */ `
fn huangGo( alpha2: f32, ci: f32, co: f32 ) -> f32 {

	let li = huangLambda( alpha2, ci );
	return ( 1.0 + li ) / ( 1.0 + li + huangLambda( alpha2, co ) );

}
`;
export const huangGoFn = wgslFn( HUANG_GO_SOURCE, [ huangLambdaFn ] );

export const HUANG_D_SOURCE = /* wgsl */ `
fn huangD( alpha2: f32, cosNH: f32 ) -> f32 {

	let c2 = min( cosNH * cosNH, 1.0 );
	let d = 1.0 - c2 + alpha2 * c2;
	return alpha2 / ( 3.141592653589793 * d * d );

}
`;
export const huangDFn = wgslFn( HUANG_D_SOURCE, [] );

/** Due assi ortogonali a n: la costruzione di Duff, la stessa di Cycles. */
export const HUANG_ORTHO_SOURCE = /* wgsl */ `
fn huangOrthoS( n: vec3f ) -> vec3f {

	let s = select( -1.0, 1.0, n.z >= 0.0 );
	let a = -1.0 / ( s + n.z );
	return vec3f( 1.0 + s * n.x * n.x * a, s * n.x * n.y * a, - s * n.x );

}
`;
export const huangOrthoSFn = wgslFn( HUANG_ORTHO_SOURCE, [] );

export const HUANG_ORTHO_T_SOURCE = /* wgsl */ `
fn huangOrthoT( n: vec3f ) -> vec3f {

	let s = select( -1.0, 1.0, n.z >= 0.0 );
	let a = -1.0 / ( s + n.z );
	return vec3f( n.x * n.y * a, s + n.y * n.y * a, - n.y );

}
`;
export const huangOrthoTFn = wgslFn( HUANG_ORTHO_T_SOURCE, [] );

/** Il campionamento della normale visibile (Heitz), isotropo. */
export const HUANG_VNDF_SOURCE = /* wgsl */ `
fn huangSampleVndf( wi: vec3f, alpha: f32, rand: vec2f ) -> vec3f {

	let wiS = normalize( vec3f( alpha * wi.x, alpha * wi.y, wi.z ) );
	let lensq = wiS.x * wiS.x + wiS.y * wiS.y;
	var t1 = vec3f( 1.0, 0.0, 0.0 );
	var t2 = vec3f( 0.0, 1.0, 0.0 );
	if ( lensq > 1e-7 ) {

		let inv = inverseSqrt( lensq );
		t1 = vec3f( - wiS.y * inv, wiS.x * inv, 0.0 );
		t2 = cross( wiS, t1 );

	}
	let r = sqrt( rand.x );
	let phi = 6.283185307179586 * rand.y;
	let tx = r * cos( phi );
	var ty = r * sin( phi );
	let s = 0.5 * ( 1.0 + wiS.z );
	ty = ( 1.0 - s ) * sqrt( max( 0.0, 1.0 - tx * tx ) ) + s * ty;
	let tz = sqrt( max( 0.0, 1.0 - tx * tx - ty * ty ) );
	let h = t1 * tx + t2 * ty + wiS * tz;
	return normalize( vec3f( alpha * h.x, alpha * h.y, max( 0.0, h.z ) ) );

}
`;
export const huangSampleVndfFn = wgslFn( HUANG_VNDF_SOURCE, [] );

/** La normale campionata attorno a una mesonormale inclinata. */
export const HUANG_SAMPLE_WH_SOURCE = /* wgsl */ `
fn huangSampleWh( roughness: f32, wi: vec3f, wm: vec3f, rand: vec2f ) -> vec3f {

	let s = huangOrthoS( wm );
	let t = huangOrthoT( wm );
	let wiWm = vec3f( dot( wi, s ), dot( wi, t ), dot( wi, wm ) );
	let whWm = huangSampleVndf( wiWm, roughness, rand );
	return s * whWm.x + t * whWm.y + wm * whWm.z;

}
`;
export const huangSampleWhFn = wgslFn( HUANG_SAMPLE_WH_SOURCE, [ huangOrthoSFn, huangOrthoTFn, huangSampleVndfFn ] );

export const HUANG_VIS1_SOURCE = /* wgsl */ `
fn huangVisible1( v: vec3f, m: vec3f, h: vec3f ) -> bool { return dot( v, h ) > 0.0 && dot( v, m ) > 0.0; }
`;
export const huangVisible1Fn = wgslFn( HUANG_VIS1_SOURCE, [] );

export const HUANG_VIS2_SOURCE = /* wgsl */ `
fn huangVisible2( wi: vec3f, wo: vec3f, m: vec3f, h: vec3f ) -> bool {

	return huangVisible1( wi, m, h ) && huangVisible1( wo, m, h );

}
`;
export const huangVisible2Fn = wgslFn( HUANG_VIS2_SOURCE, [ huangVisible1Fn ] );

/** Il Fresnel che restituisce ANCHE il coseno dell'angolo trasmesso. */
export const HUANG_FRESNEL_T_SOURCE = /* wgsl */ `
fn huangFresnelT( cosThetaI: f32, eta: f32 ) -> vec2f {

	let etaCos2 = eta * eta - ( 1.0 - cosThetaI * cosThetaI );
	if ( etaCos2 <= 0.0 ) { return vec2f( 1.0, 0.0 ); }
	let ci = abs( cosThetaI );
	let cosT = - sqrt( etaCos2 ) / eta;
	let ct = abs( cosT );
	let rs = ( ci - eta * ct ) / ( ci + eta * ct );
	let rp = ( eta * ci - ct ) / ( eta * ci + ct );
	return vec2f( 0.5 * ( rs * rs + rp * rp ), cosT );

}
`;
export const huangFresnelTFn = wgslFn( HUANG_FRESNEL_T_SOURCE, [] );

export const HUANG_FRESNEL_SOURCE = /* wgsl */ `
fn huangFresnelCos( cosi: f32, eta: f32 ) -> f32 {

	let c = abs( cosi );
	var g = eta * eta - 1.0 + c * c;
	if ( g > 0.0 ) {

		g = sqrt( g );
		let a = ( g - c ) / ( g + c );
		let b = ( c * ( g + c ) - 1.0 ) / ( c * ( g - c ) + 1.0 );
		return 0.5 * a * a * ( 1.0 + b * b );

	}
	return 1.0;

}
`;
export const huangFresnelCosFn = wgslFn( HUANG_FRESNEL_SOURCE, [] );

export const HUANG_REFRACT_SOURCE = /* wgsl */ `
fn huangRefract( incident: vec3f, normal: vec3f, cosT: f32, invEta: f32 ) -> vec3f {

	return normal * ( invEta * dot( normal, incident ) + cosT ) - incident * invEta;

}
`;
export const huangRefractFn = wgslFn( HUANG_REFRACT_SOURCE, [] );

export const HUANG_REFLECT_SOURCE = /* wgsl */ `
fn huangReflect( v: vec3f, n: vec3f ) -> vec3f { return n * ( 2.0 * dot( v, n ) ) - v; }
`;
export const huangReflectFn = wgslFn( HUANG_REFLECT_SOURCE, [] );

/** La correzione di energia: la tabella del vetro GGX. */
export const HUANG_ENERGY_SOURCE = /* wgsl */ `
fn huangEnergy( tab: ptr<storage, array<f32>, read>, mu: f32, rough: f32, ior: f32 ) -> f32 {

	let z = sqrt( abs( ( ior - 1.0 ) / ( ior + 1.0 ) ) );
	let e = ggxGlassE( tab, rough, mu, z );
	if ( e <= 0.0 ) { return 1.0; }
	return 1.0 / e;

}
`;

export const huangEnergyFn = wgslFn( HUANG_ENERGY_SOURCE, [ ggxGlassEFn ] );

/** Il frame della fibra, e il raggio PROIETTATO. */
export const HUANG_FRAME_SOURCE = /* wgsl */ `
fn huangBuildFrame( tangent: vec3f, view: vec3f, geomNormal: vec3f, aspectRatio: f32 ) -> HuangFrame {

	var f: HuangFrame;
	f.valid = 0u;

	let y = normalize( tangent );
	let xAxis = normalize( cross( y, view ) );
	f.h = - dot( xAxis, geomNormal );

	let b = clamp( aspectRatio, 0.05, 1.0 );
	var x = xAxis;
	if ( b != 1.0 ) {

		// con la sezione ellittica il frame si allinea alla NORMALE della curva:
		// e' l'asse maggiore, e senza di lui l'ellisse girerebbe col raggio
		let inner = cross( geomNormal, y );
		if ( length( inner ) > 1e-8 ) {

			let n = normalize( cross( y, normalize( inner ) ) );
			if ( length( n ) > 0.5 ) { x = n; }

		}

	}
	let z = cross( x, y );
	if ( length( z ) < 1e-8 ) { return f; }
	f.x = x;
	f.y = y;
	f.z = normalize( z );
	f.wi = vec3f( dot( view, f.x ), dot( view, f.y ), dot( view, f.z ) );
	f.e2 = 1.0 - b * b;
	let denom = f.wi.x * f.wi.x + f.wi.z * f.wi.z;
	f.radius = 1.0;
	if ( f.e2 != 0.0 && denom != 0.0 ) {

		f.radius = sqrt( max( 0.0, 1.0 - f.e2 * f.wi.x * f.wi.x / denom ) );

	}
	if ( abs( f.h ) >= f.radius ) { return f; }
	f.valid = 1u;
	return f;

}
`;
export const huangBuildFrameFn = wgslFn( HUANG_FRAME_SOURCE, [ huangFrameStruct ] );

/** L'intervallo di h visibile, gia' diviso per il raggio. Il terzo valore dice se c'e'. */
export const HUANG_VISIBLE_H_SOURCE = /* wgsl */ `
fn huangVisibleH( hair: HuangHair, frame: HuangFrame, localO: vec3f ) -> vec3f {

	let tanTilt = tan( hair.tilt );
	if ( tanTilt * huangTanTheta( localO ) < -1.0 ) { return vec3f( 0.0, 0.0, 0.0 ); }
	let arg = max( - tanTilt * huangTanTheta( frame.wi ), 0.0 );
	if ( arg > 1.0 ) { return vec3f( 0.0, 0.0, 0.0 ); }
	let halfSpan = acos( clamp( arg, -1.0, 1.0 ) );

	let b = hair.aspectRatio;
	var phiI = 0.0;
	if ( b != 1.0 ) { phiI = huangDirPhi( frame.wi ); }
	var lo = huangPhiToH( phiI + halfSpan, b, frame.wi ) / frame.radius;
	var hi = huangPhiToH( phiI - halfSpan, b, frame.wi ) / frame.radius;
	lo = max( lo, -0.999 );
	hi = min( hi, 0.999 );
	if ( hi <= lo ) { return vec3f( 0.0, 0.0, 0.0 ); }
	return vec3f( lo, hi, 1.0 );

}
`;
export const huangVisibleHFn = wgslFn( HUANG_VISIBLE_H_SOURCE, [ huangTanThetaFn, huangDirPhiFn, huangPhiToHFn, huangHairStruct, huangFrameStruct ] );

/** Il lobo R: una quadratura di Simpson lungo la sezione. */
export const HUANG_EVAL_R_SOURCE = /* wgsl */ `
fn huangEvalR( tab: ptr<storage, array<f32>, read>, hair: HuangHair, frame: HuangFrame, wo: vec3f, hLo: f32, hHi: f32 ) -> f32 {

	if ( hair.r <= 0.0 ) { return 0.0; }
	let wi = frame.wi;
	let b = hair.aspectRatio;
	let wh = normalize( wi + wo );
	let rough = hair.roughness;
	let rough2 = rough * rough;
	let sc = huangSincosPhi( wi );

	let span = hHi - hLo;
	let intervals = 2 * i32( ceil( span / ( rough * 0.7 ) * 0.5 ) );
	let res = span / f32( intervals );

	var integral = 0.0;
	for ( var i = 0; i <= intervals; i = i + 1 ) {

		let h = hLo + f32( i ) * res;
		let gammaM = huangHToGamma( h, b, wi );
		let wm = huangSphgDir( hair.tilt, gammaM, b );
		if ( huangVisible2( wi, wo, vec3f( wm.x, 0.0, wm.z ), wh ) ) {

			let jacobian = huangDGammaDH( sc, gammaM, b );
			var coeff = 2.0;
			if ( i == 0 || i == intervals ) { coeff = 0.5; } else { coeff = f32( i % 2 + 1 ); }
			let weight = coeff * jacobian;
			let cosMi = dot( wm, wi );
			let g = huangG2( rough2, cosMi, dot( wm, wo ) );
			integral = integral + weight * huangD( rough2, dot( wm, wh ) ) * g
				* huangArcLength( frame.e2, gammaM ) * huangEnergy( tab, cosMi, sqrt( rough ), hair.eta );

		}

	}
	integral = integral * ( 2.0 / 3.0 ) * res;
	return hair.r * 0.25 * huangFresnelCos( dot( wi, wh ), hair.eta ) * integral;

}
`;

export const huangEvalRFn = wgslFn( HUANG_EVAL_R_SOURCE, [
	huangSincosPhiFn, huangHToGammaFn, huangSphgDirFn, huangVisible2Fn, huangDGammaDHFn,
	huangG2Fn, huangDFn, huangArcLengthFn, huangEnergyFn, huangFresnelCosFn,
	huangHairStruct, huangFrameStruct,
] );

/** La coda TRRT+: una serie geometrica. */
export const HUANG_TRRT_SOURCE = /* wgsl */ `
fn huangEvalTrrt( t: f32, r: f32, a: vec3f ) -> vec3f {

	let tAvg = max( 1.0 - r, 1e-5 );
	let num = t * r * r * tAvg * a * a * a;
	let den = vec3f( 1.0 ) - a * ( 1.0 - tAvg );
	return num / max( den, vec3f( 1e-6 ) );

}
`;
export const huangEvalTrrtFn = wgslFn( HUANG_TRRT_SOURCE, [] );

/**
 * TT, TRT e la coda: Simpson sulla sezione, Monte Carlo sulle micro-normali.
 *
 * E' il pezzo che rende Huang caro: per ogni punto della sezione si campionano
 * una o due micro-normali, si rifrange, si riflette dentro, si rifrange di nuovo.
 */
export const HUANG_RESIDUAL_SOURCE = /* wgsl */ `
fn huangEvalResidual( tab: ptr<storage, array<f32>, read>, hair: HuangHair, frame: HuangFrame, wo: vec3f, hLo: f32, hHi: f32, rng: ptr<function, u32> ) -> vec3f {

	if ( hair.tt <= 0.0 && hair.trt <= 0.0 ) { return vec3f( 0.0 ); }

	let wi = frame.wi;
	let b = hair.aspectRatio;
	let absorption = hair.sigma;
	let eta = hair.eta;
	let invEta = 1.0 / eta;
	let rough = hair.roughness;
	let rough2 = rough * rough;
	let sqrtRough = sqrt( rough );
	let sc = huangSincosPhi( wi );

	let span = hHi - hLo;
	let intervals = 2 * i32( ceil( span / ( rough * 0.8 ) * 0.5 ) );
	let res = span / f32( intervals );

	var sTT = vec3f( 0.0 );
	var sTRT = vec3f( 0.0 );
	var sTRRT = vec3f( 0.0 );

	for ( var i = 0; i <= intervals; i = i + 1 ) {

		let h = hLo + f32( i ) * res;
		let gammaMi = huangHToGamma( h, b, wi );
		let wmi = huangSphgDir( hair.tilt, gammaMi, b );
		let wmiFlat = huangSphgDir( 0.0, gammaMi, b );

		let wh1 = huangSampleWh( rough, wi, wmi, vec2f( huangLcg( rng ), huangLcg( rng ) ) );
		let cosHi1 = dot( wi, wh1 );
		if ( !( cosHi1 > 0.0 ) ) { continue; }

		let cosMi1 = dot( wi, wmi );
		let fr1 = huangFresnelT( cosHi1, eta );
		let t1 = 1.0 - fr1.x;
		let scale1 = huangEnergy( tab, cosMi1, sqrtRough, eta );

		let wt = huangRefract( wi, wh1, fr1.y, invEta );
		let negWt = - wt;
		let phiT = huangDirPhi( wt );
		let gammaMt = 2.0 * huangToPhi( phiT, b ) - gammaMi;
		let wmt = huangSphgDir( - hair.tilt, gammaMt, b );
		let wmtFlat = huangSphgDir( 0.0, gammaMt, b );

		let cosMo1 = dot( negWt, wmi );
		let cosMi2 = dot( negWt, wmt );
		let g1o = huangGo( rough2, cosMi1, cosMo1 );
		if ( !huangVisible2( wi, negWt, wmi, wh1 ) || !huangVisible2( wi, negWt, wmiFlat, wh1 ) ) { continue; }

		let jacobian = huangDGammaDH( sc, gammaMi, b );
		var coeff = 2.0;
		if ( i == 0 || i == intervals ) { coeff = 0.5; } else { coeff = f32( i % 2 + 1 ); }
		let weight = coeff * jacobian;

		var pathLen = 2.0 * cos( gammaMi - phiT );
		if ( b != 1.0 ) {

			let p0 = huangToPoint( gammaMi, b );
			let p1 = huangToPoint( gammaMt + 3.141592653589793, b );
			pathLen = - length( p0 - p1 );

		}
		let ctWt = huangCosTheta( wt );
		let aT = exp( absorption / max( ctWt, 1e-6 ) * pathLen );
		let scale2 = huangEnergy( tab, cosMi2, sqrtRough, invEta );

		if ( hair.tt > 0.0 && dot( wo, wt ) >= invEta - 1e-5 ) {

			let wh2raw = negWt + wo * invEta;
			let rcp = 1.0 / max( length( wh2raw ), 1e-9 );
			let wh2 = wh2raw * rcp;
			let cosMh2 = dot( wmt, wh2 );
			if ( cosMh2 >= 0.0 ) {

				let cosHi2 = dot( negWt, wh2 );
				let cosHo2 = dot( - wo, wh2 );
				let cosMo2 = dot( - wo, wmt );
				let t2 = ( 1.0 - huangFresnelCos( cosHi2, invEta ) ) * scale2;
				let d2 = huangD( rough2, cosMh2 );
				let g2 = huangG2( rough2, cosMi2, cosMo2 );
				let k = weight * t1 * scale1 * t2 * d2 * g1o * g2 / cosMo1 * cosMi1 * cosHi2 * cosHo2 * rcp * rcp;
				let v = hair.tt * k * aT * huangArcLength( frame.e2, gammaMt );
				if ( all( v == v ) ) { sTT = sTT + v; }

			}

		}

		if ( hair.trt > 0.0 ) {

			let wh2s = huangSampleWh( rough, negWt, wmt, vec2f( huangLcg( rng ), huangLcg( rng ) ) );
			let cosHi2s = dot( negWt, wh2s );
			if ( !( cosHi2s > 0.0 ) ) { continue; }
			let r2 = huangFresnelCos( cosHi2s, invEta );
			let wtr = - huangReflect( wt, wh2s );
			let negWtr = - wtr;

			if ( dot( negWtr, wo ) < invEta - 1e-5 ) {

				sTRRT = sTRRT + weight * huangEvalTrrt( t1, r2, aT );
				continue;

			}
			if ( !huangVisible2( negWt, negWtr, wmt, wh2s ) || !huangVisible2( negWt, negWtr, wmtFlat, wh2s ) ) { continue; }

			let phiTr = huangDirPhi( wtr );
			let gammaMtr = gammaMi - 2.0 * ( huangToPhi( phiT, b ) - huangToPhi( phiTr, b ) ) + 3.141592653589793;
			let wmtr = huangSphgDir( - hair.tilt, gammaMtr, b );
			let wmtrFlat = huangSphgDir( 0.0, gammaMtr, b );

			let wh3raw = wtr + wo * invEta;
			let rcp3 = 1.0 / max( length( wh3raw ), 1e-9 );
			let wh3 = wh3raw * rcp3;
			let cosMh3 = dot( wmtr, wh3 );
			let negWo = - wo;
			if ( cosMh3 < 0.0 || !huangVisible2( wtr, negWo, wmtr, wh3 ) || !huangVisible2( wtr, negWo, wmtrFlat, wh3 ) ) {

				sTRRT = sTRRT + weight * huangEvalTrrt( t1, r2, aT );
				continue;

			}

			let cosHi3 = dot( wh3, wtr );
			let cosHo3 = dot( wh3, negWo );
			let cosMi3 = dot( wmtr, wtr );
			let t3 = ( 1.0 - huangFresnelCos( cosHi3, invEta ) ) * huangEnergy( tab, cosMi3, sqrtRough, invEta );
			let d3 = huangD( rough2, cosMh3 );

			var trPath = - 2.0 * abs( cos( phiTr - gammaMt ) );
			if ( b != 1.0 ) {

				let q0 = huangToPoint( gammaMtr, b );
				let q1 = huangToPoint( gammaMt, b );
				trPath = - length( q0 - q1 );

			}
			let ctWtr = huangCosTheta( wtr );
			let aTr = exp( absorption / max( ctWtr, 1e-6 ) * trPath );

			let cosMo2 = dot( wmt, negWtr );
			let g2o = huangGo( rough2, cosMi2, cosMo2 );
			let g3 = huangG2( rough2, cosMi3, dot( wmtr, negWo ) );
			let k = weight * t1 * scale1 * r2 * scale2 * t3 * d3 * g1o * g2o * g3
				/ ( cosMo1 * cosMo2 ) * cosMi1 * cosMi2 * cosHi3 * cosHo3 * rcp3 * rcp3;
			let v = hair.trt * k * aT * aTr * huangArcLength( frame.e2, gammaMtr );
			if ( all( v == v ) ) { sTRT = sTRT + v; }

			sTRRT = sTRRT + weight * huangEvalTrrt( t1, r2, aT );

		}

	}

	let m = hairLongitudinal( wi.y, huangCosTheta( wi ), wo.y, huangCosTheta( wo ), 4.0 * rough );
	let n = 1.0 / 6.283185307179586;
	let simpson = ( 2.0 / 3.0 ) * res;
	return ( ( sTT + sTRT ) * invEta * invEta + sTRRT * m * n * ( 2.0 / 3.141592653589793 ) ) * simpson;

}
`;

/** La valutazione: il lobo R piu' il resto, diviso l'area proiettata. */
export const huangEvalResidualFn = wgslFn( HUANG_RESIDUAL_SOURCE, [
	huangSincosPhiFn, huangHToGammaFn, huangSphgDirFn, huangSampleWhFn, huangLcgFn,
	huangFresnelTFn, huangFresnelCosFn, huangRefractFn, huangReflectFn, huangDirPhiFn,
	huangToPhiFn, huangToPointFn, huangGoFn, huangG2Fn, huangDFn, huangVisible2Fn,
	huangDGammaDHFn, huangArcLengthFn, huangCosThetaFn, huangEnergyFn, huangEvalTrrtFn,
	hairLongitudinalFn, huangHairStruct, huangFrameStruct,
] );

export const HUANG_EVAL_SOURCE = /* wgsl */ `
fn huangEval( tab: ptr<storage, array<f32>, read>, hair: HuangHair, frame: HuangFrame, light: vec3f, rng: ptr<function, u32> ) -> vec3f {

	let localO = vec3f( dot( light, frame.x ), dot( light, frame.y ), dot( light, frame.z ) );
	let range = huangVisibleH( hair, frame, localO );
	if ( range.z == 0.0 ) { return vec3f( 0.0 ); }

	// FARFIELD: l'intervallo e' tutta la sezione, quindi l'area proiettata e' due
	// volte il coseno. Di la' un pelo piu' largo di un pixel lo restringe coi
	// differenziali del raggio, che qui non ci sono.
	let projected = huangCosTheta( frame.wi ) * 2.0;
	if ( !( projected > 0.0 ) ) { return vec3f( 0.0 ); }

	let r = huangEvalR( tab, hair, frame, localO, range.x, range.y );
	let rest = huangEvalResidual( tab, hair, frame, localO, range.x, range.y, rng );
	return ( vec3f( r ) + rest ) / projected;

}
`;

export const huangEvalFn = wgslFn( HUANG_EVAL_SOURCE, [
	huangVisibleHFn, huangCosThetaFn, huangEvalRFn, huangEvalResidualFn,
	huangHairStruct, huangFrameStruct,
] );

/**
 * Il campionamento: si SEGUE il raggio dentro la fibra.
 *
 * Non si sceglie un lobo e poi una direzione come in Chiang: si rifrange, si
 * riflette dentro, si rifrange di nuovo, e alla fine si prende uno dei quattro
 * punti di uscita con probabilita' proporzionale all'energia.
 */
export const HUANG_SAMPLE_SOURCE = /* wgsl */ `
fn huangSample( tab: ptr<storage, array<f32>, read>, hair: HuangHair, frame: HuangFrame, rng: ptr<function, u32> ) -> HuangScatter {

	var out: HuangScatter;
	out.valid = 0u;
	out.f = vec3f( 0.0 );
	out.direction = vec3f( 0.0, 0.0, 1.0 );

	let wi = frame.wi;
	let b = hair.aspectRatio;
	let rough = hair.roughness;
	let rough2 = rough * rough;
	let sqrtRough = sqrt( rough );
	let eta = hair.eta;
	let invEta = 1.0 / eta;

	let sampleLobe0 = huangLcg( rng );
	// FARFIELD: h si estrae invece di prenderlo dal colpo
	let hDivR = huangLcg( rng ) * 2.0 - 1.0;
	let gammaMi = huangHToGamma( hDivR, b, wi );
	let wmiFlat = huangSphgDir( 0.0, gammaMi, b );
	let st = sin( hair.tilt );
	let ct = cos( hair.tilt );
	let wmi = vec3f( wmiFlat.x * ct, st, wmiFlat.z * ct );
	let cosMi1 = dot( wmi, wi );
	if ( cosMi1 < 0.0 || dot( wmiFlat, wi ) < 0.0 ) { return out; }

	let wh1 = huangSampleWh( rough, wi, wmi, vec2f( huangLcg( rng ), huangLcg( rng ) ) );
	let wr = - huangReflect( wi, wh1 );
	if ( !huangVisible1( wi, wmiFlat, wh1 ) ) { return out; }

	let fr1 = huangFresnelT( dot( wi, wh1 ), eta );
	let scale1 = huangEnergy( tab, cosMi1, sqrtRough, eta );
	var visR = 0.0;
	if ( huangVisible1( wr, wmiFlat, wh1 ) ) { visR = 1.0; }
	let rWeight = hair.r * fr1.x * scale1 * visR * huangGo( rough2, cosMi1, dot( wmi, wr ) );

	let wt = huangRefract( wi, wh1, fr1.y, invEta );
	let negWt = - wt;
	let phiT = huangDirPhi( wt );
	let gammaMt = 2.0 * huangToPhi( phiT, b ) - gammaMi;
	let wmt = huangSphgDir( - hair.tilt, gammaMt, b );
	let wmtFlat = huangSphgDir( 0.0, gammaMt, b );
	let wh2 = huangSampleWh( rough, negWt, wmt, vec2f( huangLcg( rng ), huangLcg( rng ) ) );
	let wtr = - huangReflect( wt, wh2 );

	var tt = vec3f( 0.0 );
	var trt = vec3f( 0.0 );
	var trrt = vec3f( 0.0 );
	var wtt = vec3f( 0.0, 0.0, 1.0 );
	var wtrt = vec3f( 0.0, 0.0, 1.0 );
	var wtrrt = vec3f( 0.0, 0.0, 1.0 );
	let cosMi2 = dot( negWt, wmt );

	if ( cosMi2 > 0.0 && huangVisible1( negWt, wmiFlat, wh1 ) && huangVisible1( negWt, wmtFlat, wh2 ) ) {

		let absorption = hair.sigma;
		var pathLen = 2.0 * cos( phiT - gammaMi );
		if ( b != 1.0 ) {

			let p0 = huangToPoint( gammaMi, b );
			let p1 = huangToPoint( gammaMt + 3.141592653589793, b );
			pathLen = - length( p0 - p1 );

		}
		let ctWt = huangCosTheta( wt );
		let aT = exp( absorption / max( ctWt, 1e-6 ) * pathLen );
		let fr2 = huangFresnelT( dot( negWt, wh2 ), invEta );
		let t1 = ( 1.0 - fr1.x ) * scale1 * huangGo( rough2, cosMi1, dot( wmi, negWt ) );
		let t2 = 1.0 - fr2.x;
		let scale2 = huangEnergy( tab, cosMi2, sqrtRough, invEta );

		wtt = huangRefract( negWt, wh2, fr2.y, eta );
		let negWtt = - wtt;
		if ( dot( wmt, negWtt ) > 0.0 && t2 > 0.0 && huangVisible1( negWtt, wmtFlat, wh2 ) ) {

			let k = hair.tt * t1 * t2 * scale2 * huangGo( rough2, cosMi2, dot( wmt, negWtt ) );
			tt = aT * k;

		}

		let phiTr = huangDirPhi( wtr );
		let gammaMtr = gammaMi - 2.0 * ( huangToPhi( phiT, b ) - huangToPhi( phiTr, b ) ) + 3.141592653589793;
		let wmtr = huangSphgDir( - hair.tilt, gammaMtr, b );
		let wh3 = huangSampleWh( rough, wtr, wmtr, vec2f( huangLcg( rng ), huangLcg( rng ) ) );
		let fr3 = huangFresnelT( dot( wtr, wh3 ), invEta );
		wtrt = huangRefract( wtr, wh3, fr3.y, eta );
		let cosMi3 = dot( wmtr, wtr );

		if ( cosMi3 > 0.0 ) {

			var trPath = - 2.0 * abs( cos( phiTr - gammaMt ) );
			if ( b != 1.0 ) {

				let q0 = huangToPoint( gammaMt, b );
				let q1 = huangToPoint( gammaMtr, b );
				trPath = - length( q0 - q1 );

			}
			let ctWtr = huangCosTheta( wtr );
			let aTr = exp( absorption / max( ctWtr, 1e-6 ) * trPath );
			let trK = t1 * fr2.x * scale2 * huangEnergy( tab, cosMi3, sqrtRough, invEta )
				* huangGo( rough2, cosMi2, dot( wmt, - wtr ) );
			let tr = aT * aTr * trK;
			let t3 = 1.0 - fr3.x;
			let negWtrt = - wtrt;
			let wmtrFlat = vec3f( wmtr.x, 0.0, wmtr.z );
			if ( t3 > 0.0 && huangVisible1( negWtrt, wmtrFlat, wh3 ) && huangVisible1( wtr, wmtrFlat, wh3 ) ) {

				trt = hair.trt * tr * t3 * huangGo( rough2, cosMi3, dot( wmtr, negWtrt ) );

			}

			// la coda: una direzione dalla longitudinale larga, e un'attenuazione
			// che somma la serie geometrica dei giri successivi
			let randTheta = max( huangLcg( rng ), 1e-5 );
			let fac = 1.0 + 4.0 * rough * log( randTheta + ( 1.0 - randTheta ) * exp( -0.5 / rough ) );
			let sinThetaO = - fac * wi.y + sqrt( max( 0.0, 1.0 - fac * fac ) ) * cos( 6.283185307179586 * huangLcg( rng ) ) * huangCosTheta( wi );
			let cosThetaO = sqrt( max( 0.0, 1.0 - sinThetaO * sinThetaO ) );
			let phiO = 6.283185307179586 * huangLcg( rng );
			wtrrt = vec3f( sin( phiO ) * cosThetaO, sinThetaO, cos( phiO ) * cosThetaO );

			let tAvg = max( 0.5 * ( t2 + t3 ), 1e-5 );
			let aAvg = sqrt( aT * aTr );
			let aRes = aAvg * tAvg / max( vec3f( 1.0 ) - aAvg * ( 1.0 - tAvg ), vec3f( 1e-6 ) );
			trrt = tr * fr3.x * aRes * huangGo( rough2, cosMi3, dot( wmtr, - huangReflect( wtr, wh3 ) ) );

		}

	}

	let eR = rWeight;
	let eTT = ( tt.x + tt.y + tt.z ) / 3.0;
	let eTRT = ( trt.x + trt.y + trt.z ) / 3.0;
	let eTRRT = ( trrt.x + trrt.y + trrt.z ) / 3.0;
	let total = eR + eTT + eTRT + eTRRT;
	if ( !( total > 0.0 ) ) { return out; }

	let pick = sampleLobe0 * total;
	var localO = wr;
	var f = vec3f( total );
	if ( pick >= eR && pick < eR + eTT ) { localO = wtt; f = tt / eTT * total; }
	else if ( pick >= eR + eTT && pick < eR + eTT + eTRT ) { localO = wtrt; f = trt / eTRT * total; }
	else if ( pick >= eR + eTT + eTRT ) { localO = wtrrt; f = trrt / eTRRT * total; }

	out.f = f;
	out.direction = frame.x * localO.x + frame.y * localO.y + frame.z * localO.z;
	out.valid = 1u;
	return out;

}
`;

export const huangSampleFn = wgslFn( HUANG_SAMPLE_SOURCE, [
	huangLcgFn, huangHToGammaFn, huangSphgDirFn, huangSampleWhFn, huangVisible1Fn,
	huangFresnelTFn, huangRefractFn, huangReflectFn, huangDirPhiFn, huangToPhiFn,
	huangToPointFn, huangGoFn, huangCosThetaFn, huangEnergyFn,
	huangHairStruct, huangFrameStruct, huangScatterStruct,
] );

/**
 * Tutte le sorgenti in fila, per chi compila WGSL a mano.
 *
 * L'ordine e' quello delle dipendenze: WGSL vuole che una funzione sia
 * dichiarata prima di essere chiamata, e sbagliarlo non da' un errore leggibile.
 *
 * Il testo NON si basta: chiama `ggxGlassE` (la tabella dell'albedo) e
 * `hairLongitudinal` (la gaussiana di Chiang, che Huang riusa per la sola coda).
 * Chi lo compila deve metterci davanti quelle due sorgenti.
 */
export const HUANG_SOURCE = [
	HUANG_STRUCTS_SOURCE,
	HUANG_LCG_SOURCE,
	HUANG_ANGLES_SOURCE,
	HUANG_TAN_THETA_SOURCE,
	HUANG_DIR_PHI_SOURCE,
	HUANG_SINCOS_PHI_SOURCE,
	HUANG_TO_PHI_SOURCE,
	HUANG_TO_GAMMA_SOURCE,
	HUANG_PHI_TO_H_SOURCE,
	HUANG_H_TO_GAMMA_SOURCE,
	HUANG_DGAMMA_SOURCE,
	HUANG_TO_POINT_SOURCE,
	HUANG_SPHG_SOURCE,
	HUANG_ARC_SOURCE,
	HUANG_LAMBDA_SOURCE,
	HUANG_G2_SOURCE,
	HUANG_GO_SOURCE,
	HUANG_D_SOURCE,
	HUANG_ORTHO_SOURCE,
	HUANG_ORTHO_T_SOURCE,
	HUANG_VNDF_SOURCE,
	HUANG_SAMPLE_WH_SOURCE,
	HUANG_VIS1_SOURCE,
	HUANG_VIS2_SOURCE,
	HUANG_FRESNEL_T_SOURCE,
	HUANG_FRESNEL_SOURCE,
	HUANG_REFRACT_SOURCE,
	HUANG_REFLECT_SOURCE,
	HUANG_ENERGY_SOURCE,
	HUANG_FRAME_SOURCE,
	HUANG_VISIBLE_H_SOURCE,
	HUANG_EVAL_R_SOURCE,
	HUANG_TRRT_SOURCE,
	HUANG_RESIDUAL_SOURCE,
	HUANG_EVAL_SOURCE,
	HUANG_SAMPLE_SOURCE,
].join( '\n' );
