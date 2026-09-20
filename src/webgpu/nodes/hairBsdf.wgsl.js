// IL BSDF DEI PELI, IN WGSL: i tre lobi di Chiang, gli stessi di
// `src/hair/hairBsdf.ts` dell'applicazione.
//
// STA QUI E NON DI LA' perche' e' il kernel a includerlo, e una libreria non
// importa da chi la usa. Il gemello in TypeScript e' dove le PROPRIETA' si
// provano (campionamento e valutazione d'accordo, pdf che integra a uno, energia
// che non si crea): quelle prove girano in Node, nel cancello, e qui non
// potrebbero. Una sonda confronta i due sugli stessi ingressi.
//
// ── UNA FUNZIONE PER BLOCCO, ED E' UN VINCOLO DI TSL ──
//
// `wgslFn` analizza UNA funzione per volta. Un testo con venti funzioni non si
// puo' anteprendere a un kernel: risponde «Function is not a WGSL code» e il
// tracer muore in silenzio. Quindi ogni funzione e' un nodo suo, con le sue
// dipendenze dichiarate — la stessa forma di `hair.wgsl.js`.
//
// Il modello, i lobi e da dove viene tutto stanno scritti nel gemello: qui ci
// sono solo le note che riguardano il WGSL.
import { wgslFn } from 'three/tsl';
import { StructTypeNode } from 'three/webgpu';

/** Quel che serve a entrambi i versi: attenuazioni, pesi, angoli della fibra. */
export const hairGeomStruct = new StructTypeNode( {
	ap0: 'vec3f',
	ap1: 'vec3f',
	ap2: 'vec3f',
	ap3: 'vec3f',
	w: 'vec4f',
	gammaO: 'float',
	gammaT: 'float',
	ar: 'vec2f',
	att: 'vec2f',
	atrt: 'vec2f',
}, 'HairGeom' );

/** Il valore del BSDF e la sua pdf. */
export const hairLobesStruct = new StructTypeNode( {
	f: 'vec3f',
	pdf: 'float',
}, 'HairLobes' );

/** Lo stesso, piu' la direzione scelta e quale lobo l'ha scelta. */
export const hairScatterStruct = new StructTypeNode( {
	f: 'vec3f',
	pdf: 'float',
	direction: 'vec3f',
	lobe: 'uint',
}, 'HairScatter' );

/** Le tre struct come testo: le sonde compilano WGSL a mano, senza TSL. */
export const HAIR_BSDF_STRUCTS_SOURCE = /* wgsl */ `
struct HairGeom {
	ap0: vec3f,
	ap1: vec3f,
	ap2: vec3f,
	ap3: vec3f,
	w: vec4f,
	gammaO: f32,
	gammaT: f32,
	ar: vec2f,
	att: vec2f,
	atrt: vec2f,
}

struct HairLobes {
	f: vec3f,
	pdf: f32,
}

struct HairScatter {
	f: vec3f,
	pdf: f32,
	direction: vec3f,
	lobe: u32,
}
`;

/** La scala con cui un colore diventa assorbimento: un polinomio in quinta. */
export const HAIR_SCALE_SOURCE = /* wgsl */ `
fn hairScale( x: f32 ) -> f32 {

	return (((((0.245 * x) + 5.574) * x - 10.73) * x + 2.532) * x - 0.215) * x + 5.969;

}
`;
export const hairScaleFn = wgslFn( HAIR_SCALE_SOURCE, [] );

// il colore si taglia a 1e-5: a zero il logaritmo darebbe meno infinito, e un
// assorbimento infinito diventa NaN al primo prodotto
export const HAIR_SIGMA_SOURCE = /* wgsl */ `
fn hairSigma( color: vec3f, azimuthalRoughness: f32 ) -> vec3f {

	let scale = hairScale( azimuthalRoughness );
	let l = log( max( color, vec3f( 1e-5 ) ) ) / scale;
	return l * l;

}
`;
export const hairSigmaFn = wgslFn( HAIR_SIGMA_SOURCE, [ hairScaleFn ] );

// ── I DUE PIGMENTI, e i loro colori sono MISURATI ──
//
// Eumelanina e feomelanina hanno due spettri di assorbimento diversi: il primo
// fa i capelli neri e castani, il secondo il rosso. I due terzetti stanno in
// bsdf_util.h di Cycles, e il rimappaggio logaritmico della quantita' e' quello
// di Bitterli — serve a rendere la manopola 0..1 percettivamente lineare, perche'
// l'assorbimento vero va da zero a infinito.
export const HAIR_MELANIN_SOURCE = /* wgsl */ `
fn hairMelanin( amount: f32, redness: f32 ) -> vec3f {

	if ( amount <= 0.0 ) { return vec3f( 0.0 ); }
	let m = - log( max( 1.0 - clamp( amount, 0.0, 1.0 ), 0.0001 ) );
	let r = clamp( redness, 0.0, 1.0 );
	let eu = m * ( 1.0 - r );
	let pheo = m * r;
	return eu * vec3f( 0.506, 0.841, 1.653 ) + pheo * vec3f( 0.343, 0.733, 1.924 );

}
`;
export const hairMelaninFn = wgslFn( HAIR_MELANIN_SOURCE, [] );

/** Il Fresnel dielettrico da un coseno. */
export const HAIR_FRESNEL_SOURCE = /* wgsl */ `
fn hairFresnel( cosi: f32, eta: f32 ) -> f32 {

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
export const hairFresnelFn = wgslFn( HAIR_FRESNEL_SOURCE, [] );

// ── LA SERIE DEL BESSEL HA UN NUMERO FISSO DI GIRI, e non puo' averne altri ──
//
// Di la' esce appena due termini coincidono; qui il ciclo e' a otto passi secchi.
// Su un'architettura a warp un'uscita anticipata non fa risparmiare niente — le
// corsie aspettano la piu' lenta — e in cambio toglie la divergenza.
export const HAIR_BESSEL_SOURCE = /* wgsl */ `
fn hairBessel( x0: f32 ) -> f32 {

	let x = x0 * x0;
	var val = 1.0 + 0.25 * x;
	var powX2i = x * x;
	var iFac2 = 1.0;
	var pow4i = 16.0;
	for ( var i = 2; i < 10; i = i + 1 ) {

		let fi = f32( i );
		iFac2 = iFac2 * fi * fi;
		val = val + powX2i / ( pow4i * iFac2 );
		powX2i = powX2i * x;
		pow4i = pow4i * 4.0;

	}
	return val;

}
`;
export const hairBesselFn = wgslFn( HAIR_BESSEL_SOURCE, [] );

export const HAIR_LOG_BESSEL_SOURCE = /* wgsl */ `
fn hairLogBessel( x: f32 ) -> f32 {

	if ( x > 12.0 ) {

		return x + 0.5 * ( 1.0 / ( 8.0 * x ) - 1.8378770664093453 - log( x ) );

	}
	return log( hairBessel( x ) );

}
`;
export const hairLogBesselFn = wgslFn( HAIR_LOG_BESSEL_SOURCE, [ hairBesselFn ] );

export const HAIR_LOGISTIC_SOURCE = /* wgsl */ `
fn hairLogistic( x: f32, s: f32 ) -> f32 {

	let v = exp( - abs( x ) / s );
	let d = 1.0 + v;
	return v / ( s * d * d );

}
`;
export const hairLogisticFn = wgslFn( HAIR_LOGISTIC_SOURCE, [] );

export const HAIR_LOGISTIC_CDF_SOURCE = /* wgsl */ `
fn hairLogisticCdf( x: f32, s: f32 ) -> f32 {

	let arg = - x / s;
	if ( arg > 88.0 ) { return 0.0; }
	return 1.0 / ( 1.0 + exp( arg ) );

}
`;
export const hairLogisticCdfFn = wgslFn( HAIR_LOGISTIC_CDF_SOURCE, [] );

/** La logistica tagliata su [-pi, pi], rinormalizzata. */
export const HAIR_TRIMMED_SOURCE = /* wgsl */ `
fn hairTrimmed( x: f32, s: f32 ) -> f32 {

	let scaling = 1.0 - 2.0 * hairLogisticCdf( - 3.141592653589793, s );
	if ( scaling <= 0.0 ) { return 0.0; }
	return hairLogistic( x, s ) / scaling;

}
`;
export const hairTrimmedFn = wgslFn( HAIR_TRIMMED_SOURCE, [ hairLogisticFn, hairLogisticCdfFn ] );

export const HAIR_SAMPLE_TRIMMED_SOURCE = /* wgsl */ `
fn hairSampleTrimmed( u: f32, s: f32 ) -> f32 {

	let cdf = hairLogisticCdf( - 3.141592653589793, s );
	let x = - s * log( 1.0 / ( u * ( 1.0 - 2.0 * cdf ) + cdf ) - 1.0 );
	return clamp( x, - 3.141592653589793, 3.141592653589793 );

}
`;
export const hairSampleTrimmedFn = wgslFn( HAIR_SAMPLE_TRIMMED_SOURCE, [ hairLogisticCdfFn ] );

export const HAIR_WRAP_SOURCE = /* wgsl */ `
fn hairWrap( a: f32 ) -> f32 {

	let p = 3.141592653589793;
	return ( a + p ) - 2.0 * p * floor( ( a + p ) / ( 2.0 * p ) ) - p;

}
`;
export const hairWrapFn = wgslFn( HAIR_WRAP_SOURCE, [] );

/** Di quanto gira il piano normale dopo `p` rimbalzi interni. */
export const HAIR_DELTA_PHI_SOURCE = /* wgsl */ `
fn hairDeltaPhi( p: f32, gammaO: f32, gammaT: f32 ) -> f32 {

	return 2.0 * p * gammaT - 2.0 * gammaO + p * 3.141592653589793;

}
`;
export const hairDeltaPhiFn = wgslFn( HAIR_DELTA_PHI_SOURCE, [] );

export const HAIR_AZIMUTHAL_SOURCE = /* wgsl */ `
fn hairAzimuthal( phi: f32, p: f32, s: f32, gammaO: f32, gammaT: f32 ) -> f32 {

	return hairTrimmed( hairWrap( phi - hairDeltaPhi( p, gammaO, gammaT ) ), s );

}
`;
export const hairAzimuthalFn = wgslFn( HAIR_AZIMUTHAL_SOURCE, [ hairTrimmedFn, hairWrapFn, hairDeltaPhiFn ] );

/** La diffusione longitudinale, col ramo logaritmico sotto varianza 0,1. */
export const HAIR_LONGITUDINAL_SOURCE = /* wgsl */ `
fn hairLongitudinal( sinI: f32, cosI: f32, sinO: f32, cosO: f32, v: f32 ) -> f32 {

	let invV = 1.0 / v;
	let cosArg = cosI * cosO * invV;
	let sinArg = sinI * sinO * invV;
	if ( v <= 0.1 ) {

		return exp( hairLogBessel( cosArg ) - sinArg - invV + 0.6931 + log( 0.5 * invV ) );

	}
	return ( exp( - sinArg ) * hairBessel( cosArg ) ) / ( sinh( invV ) * 2.0 * v );

}
`;
export const hairLongitudinalFn = wgslFn( HAIR_LONGITUDINAL_SOURCE, [ hairLogBesselFn, hairBesselFn ] );

// ── LA GEOMETRIA DEL COLPO: quanto cammino fa la luce, e quanto ne resta ──
//
// `h` decide tutto: passare vicino al bordo accorcia il tragitto dentro la fibra,
// quindi il TT esce piu' chiaro. I quattro pesi sono i grigi delle attenuazioni,
// normalizzati — un lobo che porta poca energia si campiona di rado.
export const HAIR_GEOM_SOURCE = /* wgsl */ `
fn hairGeom( sigma: vec3f, alpha: f32, eta: f32, sinThetaO: f32, cosThetaO: f32, h: f32 ) -> HairGeom {

	var g: HairGeom;

	let sinThetaT = sinThetaO / eta;
	let cosThetaT = sqrt( max( 0.0, 1.0 - sinThetaT * sinThetaT ) );

	let sinGammaO = h;
	let cosGammaO = sqrt( max( 0.0, 1.0 - sinGammaO * sinGammaO ) );
	g.gammaO = asin( clamp( sinGammaO, -1.0, 1.0 ) );

	let sinGammaT = sinGammaO * cosThetaO / sqrt( max( 1e-8, eta * eta - sinThetaO * sinThetaO ) );
	let cosGammaT = sqrt( max( 0.0, 1.0 - sinGammaT * sinGammaT ) );
	g.gammaT = asin( clamp( sinGammaT, -1.0, 1.0 ) );

	let t = exp( - sigma * ( 2.0 * cosGammaT / max( cosThetaT, 1e-6 ) ) );
	let fr = hairFresnel( cosThetaO * cosGammaO, eta );

	g.ap0 = vec3f( fr );
	g.ap1 = ( 1.0 - fr ) * ( 1.0 - fr ) * t;
	g.ap2 = g.ap1 * t * fr;
	let tf = t * fr;
	g.ap3 = g.ap2 * ( tf / max( vec3f( 1e-6 ), vec3f( 1.0 ) - tf ) );

	let gray = vec3f( 0.2126729, 0.7151522, 0.072175 );
	var e = vec4f( dot( g.ap0, gray ), dot( g.ap1, gray ), dot( g.ap2, gray ), dot( g.ap3, gray ) );
	let sum = e.x + e.y + e.z + e.w;
	if ( sum > 0.0 ) { e = e / sum; } else { e = vec4f( 0.0 ); }
	g.w = e;

	// i tre lobi sono inclinati di 2a, a, 4a: e' questo che li stacca a schermo
	let s1 = sin( alpha );
	let c1 = sqrt( max( 0.0, 1.0 - s1 * s1 ) );
	let s2 = 2.0 * s1 * c1;
	let c2 = c1 * c1 - s1 * s1;
	let s4 = 2.0 * s2 * c2;
	let c4 = c2 * c2 - s2 * s2;
	g.ar = vec2f( sinThetaO * c2 - cosThetaO * s2, abs( cosThetaO * c2 + sinThetaO * s2 ) );
	g.att = vec2f( sinThetaO * c1 + cosThetaO * s1, abs( cosThetaO * c1 - sinThetaO * s1 ) );
	g.atrt = vec2f( sinThetaO * c4 + cosThetaO * s4, abs( cosThetaO * c4 - sinThetaO * s4 ) );

	return g;

}
`;
export const hairGeomFn = wgslFn( HAIR_GEOM_SOURCE, [ hairFresnelFn, hairGeomStruct ] );

/** Somma i quattro lobi per una coppia di direzioni gia' ridotta ad angoli. */
export const HAIR_LOBES_SOURCE = /* wgsl */ `
fn hairLobes( g: HairGeom, v: f32, m0: f32, sAz: f32, sinThetaO: f32, cosThetaO: f32, sinThetaI: f32, cosThetaI: f32, phi: f32 ) -> HairLobes {

	var out: HairLobes;
	out.f = vec3f( 0.0 );
	out.pdf = 0.0;

	let mp0 = hairLongitudinal( sinThetaI, cosThetaI, g.ar.x, g.ar.y, m0 );
	let np0 = hairAzimuthal( phi, 0.0, sAz, g.gammaO, g.gammaT );
	out.f = out.f + g.ap0 * mp0 * np0;
	out.pdf = out.pdf + g.w.x * mp0 * np0;

	let mp1 = hairLongitudinal( sinThetaI, cosThetaI, g.att.x, g.att.y, 0.25 * v );
	let np1 = hairAzimuthal( phi, 1.0, sAz, g.gammaO, g.gammaT );
	out.f = out.f + g.ap1 * mp1 * np1;
	out.pdf = out.pdf + g.w.y * mp1 * np1;

	let mp2 = hairLongitudinal( sinThetaI, cosThetaI, g.atrt.x, g.atrt.y, 4.0 * v );
	let np2 = hairAzimuthal( phi, 2.0, sAz, g.gammaO, g.gammaT );
	out.f = out.f + g.ap2 * mp2 * np2;
	out.pdf = out.pdf + g.w.z * mp2 * np2;

	// la coda TRRT+: longitudinale larga e azimutale UNIFORME, che e' quel che
	// resta quando la luce ha girato dentro abbastanza da perdere ogni direzione
	let mp3 = hairLongitudinal( sinThetaI, cosThetaI, sinThetaO, cosThetaO, 4.0 * v );
	let np3 = 1.0 / 6.283185307179586;
	out.f = out.f + g.ap3 * mp3 * np3;
	out.pdf = out.pdf + g.w.w * mp3 * np3;

	return out;

}
`;
export const hairLobesFn = wgslFn( HAIR_LOBES_SOURCE, [ hairLongitudinalFn, hairAzimuthalFn, hairGeomStruct, hairLobesStruct ] );

// ── I DUE VERSI ──
//
// `view` e `light` puntano ENTRAMBI via dal pelo, come in Cycles: un pelo
// trasmette e riflette con la stessa formula, quindi non c'e' un verso entrante e
// uno uscente.
export const HAIR_EVAL_SOURCE = /* wgsl */ `
fn hairEval( sigma: vec3f, v: f32, s: f32, m0: f32, alpha: f32, eta: f32, fx: vec3f, fy: vec3f, fz: vec3f, h: f32, view: vec3f, light: vec3f ) -> HairLobes {

	let sinThetaO = dot( view, fx );
	let cosThetaO = sqrt( max( 0.0, 1.0 - sinThetaO * sinThetaO ) );
	let phiO = atan2( dot( view, fz ), dot( view, fy ) );

	let g = hairGeom( sigma, alpha, eta, sinThetaO, cosThetaO, h );

	let sinThetaI = dot( light, fx );
	let cosThetaI = sqrt( max( 0.0, 1.0 - sinThetaI * sinThetaI ) );
	let phiI = atan2( dot( light, fz ), dot( light, fy ) );

	return hairLobes( g, v, m0, s, sinThetaO, cosThetaO, sinThetaI, cosThetaI, phiI - phiO );

}
`;
export const hairEvalFn = wgslFn( HAIR_EVAL_SOURCE, [ hairGeomFn, hairLobesFn, hairLobesStruct ] );

export const HAIR_SCATTER_SOURCE = /* wgsl */ `
fn hairScatter( sigma: vec3f, v: f32, s: f32, m0: f32, alpha: f32, eta: f32, fx: vec3f, fy: vec3f, fz: vec3f, h: f32, view: vec3f, rand: vec3f ) -> HairScatter {

	let sinThetaO = dot( view, fx );
	let cosThetaO = sqrt( max( 0.0, 1.0 - sinThetaO * sinThetaO ) );
	let phiO = atan2( dot( view, fz ), dot( view, fy ) );

	let g = hairGeom( sigma, alpha, eta, sinThetaO, cosThetaO, h );

	// il lobo si sorteggia col suo peso, e il RESTO di quel numero si riusa per
	// l'angolo longitudinale: e' la stratificazione di Cycles, e toglierla
	// vorrebbe dire una dimensione in piu' per ogni colpo
	var r = rand.z;
	var p = 0u;
	var wSel = g.w.x;
	if ( r >= g.w.x ) { r = r - g.w.x; p = 1u; wSel = g.w.y; }
	if ( p == 1u && r >= g.w.y ) { r = r - g.w.y; p = 2u; wSel = g.w.z; }
	if ( p == 2u && r >= g.w.z ) { r = r - g.w.z; p = 3u; wSel = g.w.w; }
	if ( wSel > 0.0 ) { r = r / wSel; } else { r = 0.0; }

	// LA VARIANZA DEL CAMPIONAMENTO NON E' QUELLA DEL LOBO R: per il primo si
	// campiona con la varianza piena e si VALUTA con quella della vernice. Quando
	// la vernice ha stretto R, la direzione si sceglie sulla forma larga e il peso
	// la corregge.
	var vSample = v;
	if ( p == 1u ) { vSample = 0.25 * v; }
	if ( p >= 2u ) { vSample = 4.0 * v; }

	var sinTilted = sinThetaO;
	var cosTilted = cosThetaO;
	if ( p == 0u ) { sinTilted = g.ar.x; cosTilted = g.ar.y; }
	if ( p == 1u ) { sinTilted = g.att.x; cosTilted = g.att.y; }
	if ( p == 2u ) { sinTilted = g.atrt.x; cosTilted = g.atrt.y; }

	let rz = max( r, 1e-5 );
	let fac = 1.0 + vSample * log( rz + ( 1.0 - rz ) * exp( - 2.0 / vSample ) );
	let sinThetaI = - fac * sinTilted + sqrt( max( 0.0, 1.0 - fac * fac ) ) * cos( 6.283185307179586 * rand.y ) * cosTilted;
	let cosThetaI = sqrt( max( 0.0, 1.0 - sinThetaI * sinThetaI ) );

	var phi = 6.283185307179586 * rand.x;
	if ( p < 3u ) { phi = hairDeltaPhi( f32( p ), g.gammaO, g.gammaT ) + hairSampleTrimmed( rand.x, s ); }
	let phiI = phiO + phi;

	let lobes = hairLobes( g, v, m0, s, sinThetaO, cosThetaO, sinThetaI, cosThetaI, phi );

	var out: HairScatter;
	out.f = lobes.f;
	out.pdf = lobes.pdf;
	out.lobe = p;
	out.direction = fx * sinThetaI + fy * cosThetaI * cos( phiI ) + fz * cosThetaI * sin( phiI );
	return out;

}
`;
export const hairScatterFn = wgslFn( HAIR_SCATTER_SOURCE, [ hairGeomFn, hairLobesFn, hairDeltaPhiFn, hairSampleTrimmedFn, hairScatterStruct ] );

/**
 * Prepara il materiale: le due ruvidita' 0..1 diventano varianza e scala.
 *
 * Restituisce i tre numeri in un `vec3f` — `v`, `s`, `m0` — invece di una struct:
 * sono tre float e una struct in piu' e' una dipendenza in piu' da dichiarare in
 * ogni kernel che la tocca.
 */
export const HAIR_SETUP_SOURCE = /* wgsl */ `
fn hairSetup( roughness: f32, radialRoughness: f32, coat: f32 ) -> vec3f {

	let vRaw = clamp( roughness, 0.001, 1.0 );
	let sRaw = clamp( radialRoughness, 0.001, 1.0 );
	let m0Raw = clamp( ( 1.0 - clamp( coat, 0.0, 1.0 ) ) * vRaw, 0.001, 1.0 );

	// ── LE POTENZE SI FANNO A QUADRATI, non con pow ──
	//
	// La potenza ventesima con pow passa da un esponenziale di logaritmo e porta
	// l'errore del logaritmo moltiplicato per venti; di la' sono quadrati
	// annidati. Misurato col
	// confronto CPU/GPU: con quella forma lo scarto peggiore sul valore era 7,7e-3, a
	// quadrati e' sceso. La differenza si vede perche' la varianza finisce dentro
	// un esponenziale, dove un millesimo diventa visibile.
	let v2 = vRaw * vRaw;
	let v4 = v2 * v2;
	let v10 = v4 * v4 * v2;
	let p20 = v10 * v10;
	let vv = 0.726 * vRaw + 0.812 * v2 + 3.700 * p20;
	let m2 = m0Raw * m0Raw;
	let m4 = m2 * m2;
	let m10 = m4 * m4 * m2;
	let m20 = m10 * m10;
	let mm = 0.726 * m0Raw + 0.812 * m2 + 3.700 * m20;
	let q2 = sRaw * sRaw;
	let q4 = q2 * q2;
	let q10 = q4 * q4 * q2;
	let s22 = q10 * q10 * q2;

	return vec3f(
		vv * vv,
		( 0.265 * sRaw + 1.194 * sRaw * sRaw + 5.372 * s22 ) * 0.6266570686577501,
		mm * mm,
	);

}
`;
export const hairSetupFn = wgslFn( HAIR_SETUP_SOURCE, [] );

/**
 * Tutte le sorgenti in fila, per chi compila WGSL a mano.
 *
 * L'ordine e' quello delle dipendenze: WGSL vuole che una funzione sia
 * dichiarata prima di essere chiamata, e sbagliarlo non da' un errore leggibile.
 */
export const HAIR_BSDF_SOURCE = [
	HAIR_BSDF_STRUCTS_SOURCE,
	HAIR_SCALE_SOURCE,
	HAIR_SIGMA_SOURCE,
	HAIR_MELANIN_SOURCE,
	HAIR_FRESNEL_SOURCE,
	HAIR_BESSEL_SOURCE,
	HAIR_LOG_BESSEL_SOURCE,
	HAIR_LOGISTIC_SOURCE,
	HAIR_LOGISTIC_CDF_SOURCE,
	HAIR_TRIMMED_SOURCE,
	HAIR_SAMPLE_TRIMMED_SOURCE,
	HAIR_WRAP_SOURCE,
	HAIR_DELTA_PHI_SOURCE,
	HAIR_AZIMUTHAL_SOURCE,
	HAIR_LONGITUDINAL_SOURCE,
	HAIR_GEOM_SOURCE,
	HAIR_LOBES_SOURCE,
	HAIR_EVAL_SOURCE,
	HAIR_SCATTER_SOURCE,
	HAIR_SETUP_SOURCE,
].join( '\n' );
