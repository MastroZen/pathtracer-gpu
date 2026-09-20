// LA QUERY SUL MANTO, IN WGSL: la stessa cosa che `src/hair/curveQuery.ts` fa in Node.
//
// STA QUI E NON NELL'APPLICAZIONE perche' e' il kernel a includerla, e una
// libreria non importa da chi la usa. Il gemello in TypeScript resta di la' come
// riferimento, e `checks/e2e/_sondaCurveWgsl.mjs` li confronta: importa questo
// file da node_modules e quello da src/, e pretende gli stessi numeri.
//
// ── UNA FUNZIONE PER BLOCCO, ED E' UN VINCOLO DI TSL ──
//
// `wgslFn` analizza UNA funzione per volta: un testo con dieci funzioni e una
// struct non si puo' anteprendere a un kernel — risponde «Function is not a WGSL
// code» e il tracer muore in silenzio, che e' il modo in cui questo pezzo
// fallisce. Quindi ogni funzione e' un nodo suo, con le sue dipendenze
// dichiarate, come fa `material.wgsl.js`.
//
// Le SORGENTI restano esportate accanto ai nodi: le sonde compilano WGSL a mano,
// senza TSL, e devono poter prendere lo stesso identico testo. Una fonte sola,
// due consumatori.
//
// ── IL DATO ARRIVA PER PUNTATORE ──
//
// Nel kernel il buffer entra per interpolazione e il suo nome lo decide TSL,
// quindi un testo che lo nomina si lega a un nome che non controlla. WGSL ammette
// i puntatori nello spazio storage: chi chiama scrive `hairQuery( &buffer, ... )`,
// e il testo non presume niente.
//
// Il pacchetto, e l'aritmetica dei suoi offset, stanno in `src/hair/curvePack.ts`.
import { wgslFn } from 'three/tsl';
import { StructTypeNode } from 'three/webgpu';

/** Il colpo su un pelo. Gli stessi campi di `CurveQueryHit`, piu' `didHit`. */
export const hairHitStruct = new StructTypeNode( {
	didHit: 'bool',
	dist: 'float',
	normal: 'vec3f',
	u: 'float',
	segment: 'uint',
}, 'HairHit' );

/** Lo stesso, come testo: le sonde non hanno TSL e dichiarano la struct a mano. */
export const HAIR_HIT_SOURCE = /* wgsl */ `
struct HairHit {
	didHit: bool,
	dist: f32,
	normal: vec3f,
	u: f32,
	segment: u32,
}
`;

/**
 * Il pacchetto VUOTO: otto parole a zero.
 *
 * Uno storage buffer va SEMPRE legato, anche quando nella scena non c'e' un pelo.
 * Otto parole e non una: la traversata legge l'intestazione, e leggere fuori da un
 * array non e' un errore che qualcuno segnala — e' un numero che capita, e con un
 * numero di nodi che capita la traversata gira dentro un albero che non esiste.
 */
export const EMPTY_HAIR_DATA = new Uint32Array( 8 );

// ── LE LETTURE DAL PACCHETTO ──
//
// L'intestazione sta nelle prime otto parole e dice dove comincia ogni sezione.
// Gli offset si LEGGONO e non si ricalcolano: ricalcolarli vorrebbe dire ripetere
// qui l'aritmetica di curvePack.ts, e due conti che devono tornare uguali sono due
// conti che prima o poi divergono.
export const HAIR_WORD_SOURCE = /* wgsl */ `
fn hairWord( data: ptr<storage, array<u32>, read>, i: u32 ) -> u32 { return (*data)[ i ]; }
`;
export const hairWordFn = wgslFn( HAIR_WORD_SOURCE, [  ] );

export const HAIR_NODE_COUNT_SOURCE = /* wgsl */ `
fn hairNodeCount( data: ptr<storage, array<u32>, read> ) -> u32 { return hairWord( data, 5u ); }
`;
export const hairNodeCountFn = wgslFn( HAIR_NODE_COUNT_SOURCE, [ hairWordFn ] );

export const HAIR_BOUNDS_MIN_SOURCE = /* wgsl */ `
fn hairBoundsMin( data: ptr<storage, array<u32>, read>, node: u32 ) -> vec3f {

	let at = hairWord( data, 0u ) + node * 6u;
	return vec3f(
		bitcast<f32>( hairWord( data, at ) ),
		bitcast<f32>( hairWord( data, at + 1u ) ),
		bitcast<f32>( hairWord( data, at + 2u ) ),
	);

}
`;
export const hairBoundsMinFn = wgslFn( HAIR_BOUNDS_MIN_SOURCE, [ hairWordFn ] );

export const HAIR_BOUNDS_MAX_SOURCE = /* wgsl */ `
fn hairBoundsMax( data: ptr<storage, array<u32>, read>, node: u32 ) -> vec3f {

	let at = hairWord( data, 0u ) + node * 6u;
	return vec3f(
		bitcast<f32>( hairWord( data, at + 3u ) ),
		bitcast<f32>( hairWord( data, at + 4u ) ),
		bitcast<f32>( hairWord( data, at + 5u ) ),
	);

}
`;
export const hairBoundsMaxFn = wgslFn( HAIR_BOUNDS_MAX_SOURCE, [ hairWordFn ] );

export const HAIR_NODE_FIELD_SOURCE = /* wgsl */ `
fn hairNodeField( data: ptr<storage, array<u32>, read>, node: u32, field: u32 ) -> u32 {

	return hairWord( data, hairWord( data, 1u ) + node * 4u + field );

}
`;
export const hairNodeFieldFn = wgslFn( HAIR_NODE_FIELD_SOURCE, [ hairWordFn ] );

export const HAIR_ORDER_AT_SOURCE = /* wgsl */ `
fn hairOrderAt( data: ptr<storage, array<u32>, read>, i: u32 ) -> u32 { return hairWord( data, hairWord( data, 2u ) + i ); }
`;
export const hairOrderAtFn = wgslFn( HAIR_ORDER_AT_SOURCE, [ hairWordFn ] );

/** Il punto porta il suo spessore nella quarta componente: una lettura invece di due. */
export const HAIR_POINT_AT_SOURCE = /* wgsl */ `
fn hairPointAt( data: ptr<storage, array<u32>, read>, i: u32 ) -> vec4f {

	let at = hairWord( data, 3u ) + i * 4u;
	return vec4f(
		bitcast<f32>( hairWord( data, at ) ),
		bitcast<f32>( hairWord( data, at + 1u ) ),
		bitcast<f32>( hairWord( data, at + 2u ) ),
		bitcast<f32>( hairWord( data, at + 3u ) ),
	);

}
`;
export const hairPointAtFn = wgslFn( HAIR_POINT_AT_SOURCE, [ hairWordFn ] );

export const HAIR_SEGMENT_ENDS_SOURCE = /* wgsl */ `
fn hairSegmentEnds( data: ptr<storage, array<u32>, read>, s: u32 ) -> vec2u {

	let at = hairWord( data, 4u ) + s * 6u;
	return vec2u( hairWord( data, at ), hairWord( data, at + 1u ) );

}
`;
export const hairSegmentEndsFn = wgslFn( HAIR_SEGMENT_ENDS_SOURCE, [ hairWordFn ] );

/** Dove comincia e finisce il segmento lungo la sua CIOCCA. */
export const HAIR_SEGMENT_RANGE_SOURCE = /* wgsl */ `
fn hairSegmentRange( data: ptr<storage, array<u32>, read>, s: u32 ) -> vec2f {

	let at = hairWord( data, 4u ) + s * 6u;
	return vec2f( bitcast<f32>( hairWord( data, at + 2u ) ), bitcast<f32>( hairWord( data, at + 3u ) ) );

}
`;
export const hairSegmentRangeFn = wgslFn( HAIR_SEGMENT_RANGE_SOURCE, [ hairWordFn ] );

export const HAIR_BOX_HIT_SOURCE = /* wgsl */ `
fn hairBoxHit( data: ptr<storage, array<u32>, read>, node: u32, origin: vec3f, dir: vec3f, maxDist: f32 ) -> bool {

	let lo = hairBoundsMin( data, node );
	let hi = hairBoundsMax( data, node );
	var near = 0.0;
	var far = maxDist;
	for ( var k = 0u; k < 3u; k = k + 1u ) {

		// la divisione per una componente nulla da' infinito col segno giusto, e i
		// confronti sotto lo reggono
		let inv = 1.0 / dir[ k ];
		var a = ( lo[ k ] - origin[ k ] ) * inv;
		var b = ( hi[ k ] - origin[ k ] ) * inv;
		if ( a > b ) { let swap = a; a = b; b = swap; }
		if ( a > near ) { near = a; }
		if ( b < far ) { far = b; }
		if ( far < near ) { return false; }

	}
	return true;

}
`;
export const hairBoxHitFn = wgslFn( HAIR_BOX_HIT_SOURCE, [ hairBoundsMinFn, hairBoundsMaxFn ] );

export const HAIR_INTERSECT_SEGMENT_SOURCE = /* wgsl */ `
fn hairIntersectSegment(
	origin: vec3f, dir: vec3f,
	p0: vec3f, r0: f32,
	p1: vec3f, r1: f32,
	minDist: f32, maxDist: f32,
) -> HairHit {

	var best: HairHit;
	best.didHit = false;
	best.dist = maxDist;

	// ── L'ORIGINE SI PORTA VICINO AL PELO, O IN f32 NON RESTA NIENTE ──
	//
	// Nella quadratica compare k0 = d2 * m5 - m1 * m1, dove m5 e' la distanza al
	// quadrato dall'origine. Con la camera a quattro metri da un segmento lungo nove
	// centimetri i due termini valgono entrambi ~0,13 e si CANCELLANO: in f64 non si
	// vede, in f32 restano due cifre. Misurato col confronto CPU/GPU: fino al 24% di
	// errore relativo sulla distanza, cioe' peli messi un quarto piu' in la'.
	//
	// Si parte da dove il raggio passa piu' vicino, e lo spostamento si rimette alla
	// fine. La versione in TypeScript fa la stessa identica cosa: se una delle due
	// cambia, cambiano tutte e due.
	let mid = ( p0 + p1 ) * 0.5;
	let shift = max( 0.0, dot( mid - origin, dir ) );
	let near = origin + dir * shift;

	let ba = p1 - p0;
	let oa = near - p0;
	let ob = near - p1;
	let rr = r0 - r1;

	let m0 = dot( ba, ba );
	let m1 = dot( ba, oa );
	let m2 = dot( ba, dir );
	let m3 = dot( dir, oa );
	let m5 = dot( oa, oa );
	let m6 = dot( ob, dir );
	let m7 = dot( ob, ob );

	let d2 = m0 - rr * rr;

	// ── IL TRONCO ──
	let k2 = d2 - m2 * m2;
	let k1 = d2 * m3 - m1 * m2 + m2 * rr * r0;
	let k0 = d2 * m5 - m1 * m1 + m1 * rr * r0 * 2.0 - m0 * r0 * r0;
	let h = k1 * k1 - k0 * k2;
	if ( h >= 0.0 && abs( k2 ) > 1e-20 ) {

		let t = ( - sqrt( h ) - k1 ) / k2;
		// "y" dice dove il colpo cade lungo l'asse: fuori da [0, d2] il tronco non
		// c'e' e a rispondere sono le calotte
		let y = m1 - r0 * rr + t * m2;
		let dist = t + shift;
		if ( y > 0.0 && y < d2 && dist > minDist && dist < best.dist ) {

			best.didHit = true;
			best.dist = dist;
			best.normal = normalize( d2 * ( oa + dir * t ) - ba * y );
			best.u = clamp( y / d2, 0.0, 1.0 );

		}

	}

	// ── LE CALOTTE ──
	//
	// Servono davvero: fra un segmento e il successivo il tronco cambia direzione,
	// e senza la calotta resta una fessura sul gomito.
	let h0 = m3 * m3 - m5 + r0 * r0;
	if ( h0 > 0.0 ) {

		let t = - m3 - sqrt( h0 );
		let dist = t + shift;
		if ( dist > minDist && dist < best.dist ) {

			best.didHit = true;
			best.dist = dist;
			best.normal = normalize( ( oa + dir * t ) / max( r0, 1e-12 ) );
			best.u = 0.0;

		}

	}
	let h1 = m6 * m6 - m7 + r1 * r1;
	if ( h1 > 0.0 ) {

		let t = - m6 - sqrt( h1 );
		let dist = t + shift;
		if ( dist > minDist && dist < best.dist ) {

			best.didHit = true;
			best.dist = dist;
			best.normal = normalize( ( ob + dir * t ) / max( r1, 1e-12 ) );
			best.u = 1.0;

		}

	}

	// la normale guarda SEMPRE verso il raggio: un pelo e' sottile e lo si
	// attraversa di continuo, quindi il lato da cui si arriva cambia a ogni colpo
	if ( best.didHit && dot( best.normal, dir ) > 0.0 ) {

		best.normal = - best.normal;

	}

	return best;

}
`;
export const hairIntersectSegmentFn = wgslFn( HAIR_INTERSECT_SEGMENT_SOURCE, [ hairHitStruct ] );

/**
 * La direzione della FIBRA nel segmento.
 *
 * E' quel che un BSDF di pelo chiede al posto della normale: un pelo non ha una
 * faccia, ha un asse, e i tre lobi (R, TT, TRT) si misurano rispetto a quello. Qui
 * serve gia' prima: la tangente viaggia nel record del colpo perche' chi ombreggia
 * NON puo' rileggere il pacchetto — il suo kernel ha finito gli storage buffer.
 */
/**
 * A quale OGGETTO della scena appartiene il segmento.
 *
 * I manti di piu' volumi stanno in UN buffer — il kernel ne lega uno — quindi
 * l'indice non puo' essere un uniform: sarebbe uno per tutti, e il secondo volume
 * si vestirebbe col materiale del primo.
 */
export const HAIR_SEGMENT_OBJECT_SOURCE = /* wgsl */ `
fn hairSegmentObject( data: ptr<storage, array<u32>, read>, s: u32 ) -> u32 {

	return hairWord( data, hairWord( data, 4u ) + s * 6u + 4u );

}
`;
export const hairSegmentObjectFn = wgslFn( HAIR_SEGMENT_OBJECT_SOURCE, [ hairWordFn ] );

/**
 * Il numero della CIOCCA, fra 0 e 1: l'attributo `random` del Principled Hair.
 *
 * Da lui vengono le variazioni di colore e ruvidita' da un pelo all'altro. Sta
 * nel pacchetto perche' il kernel non ha nessun altro posto da cui prenderlo: la
 * mappa segmento -> ciocca nel buffer non c'e'.
 */
export const HAIR_SEGMENT_RANDOM_SOURCE = /* wgsl */ `
fn hairSegmentRandom( data: ptr<storage, array<u32>, read>, s: u32 ) -> f32 {

	return bitcast<f32>( hairWord( data, hairWord( data, 4u ) + s * 6u + 5u ) );

}
`;
export const hairSegmentRandomFn = wgslFn( HAIR_SEGMENT_RANDOM_SOURCE, [ hairWordFn ] );

export const HAIR_TANGENT_SOURCE = /* wgsl */ `
fn hairTangent( data: ptr<storage, array<u32>, read>, s: u32 ) -> vec3f {

	let ends = hairSegmentEnds( data, s );
	let a = hairPointAt( data, ends.x );
	let b = hairPointAt( data, ends.y );
	return normalize( b.xyz - a.xyz );

}
`;
export const hairTangentFn = wgslFn( HAIR_TANGENT_SOURCE, [ hairSegmentEndsFn, hairPointAtFn ] );

export const HAIR_QUERY_SOURCE = /* wgsl */ `
fn hairQuery( data: ptr<storage, array<u32>, read>, origin: vec3f, dir: vec3f, maxDist: f32 ) -> HairHit {

	var best: HairHit;
	best.didHit = false;
	best.dist = maxDist;
	best.segment = 0u;
	if ( hairNodeCount( data ) == 0u ) { return best; }

	var nearest = maxDist;
	var stack: array<u32, 32>;
	var depth = 0u;
	stack[ depth ] = 0u;
	depth = depth + 1u;

	while ( depth > 0u ) {

		depth = depth - 1u;
		let node = stack[ depth ];

		// la scatola si prova contro la distanza GIA' trovata: attraversarla dietro
		// al colpo che si ha gia' non serve a niente
		if ( ! hairBoxHit( data, node, origin, dir, nearest ) ) { continue; }

		if ( hairNodeField( data, node, 0u ) == 1u ) {

			let first = hairNodeField( data, node, 1u );
			let count = hairNodeField( data, node, 2u );
			for ( var i = first; i < first + count; i = i + 1u ) {

				let s = hairOrderAt( data, i );
				let ends = hairSegmentEnds( data, s );
				let a = hairPointAt( data, ends.x );
				let b = hairPointAt( data, ends.y );
				let hit = hairIntersectSegment( origin, dir, a.xyz, a.w, b.xyz, b.w, 1e-6, nearest );
				if ( hit.didHit ) {

					nearest = hit.dist;
					best = hit;
					best.segment = s;
					// l'u locale del segmento diventa quello della CIOCCA: senza,
					// l'ombreggiatura ricomincia a ogni giuntura e si vede una banda
					let range = hairSegmentRange( data, s );
					best.u = range.x + ( range.y - range.x ) * hit.u;

				}

			}

		} else if ( depth + 2u <= 32u ) {

			// trentadue e' anche la taglia della pila qui sopra, ed e' scritto in
			// cifre perche' una costante fuori dalle funzioni non ci sta: qui ogni
			// blocco e' UNA funzione e basta

			// i figli si visitano nell'ordine del raggio: trovare presto un colpo
			// vicino fa potare il resto
			let axis = hairNodeField( data, node, 2u );
			let right = hairNodeField( data, node, 1u );
			let left = node + 1u;
			var firstChild = right;
			var secondChild = left;
			if ( dir[ axis ] >= 0.0 ) { firstChild = left; secondChild = right; }
			stack[ depth ] = secondChild;
			depth = depth + 1u;
			stack[ depth ] = firstChild;
			depth = depth + 1u;

		}

	}

	return best;

}
`;
export const hairQueryFn = wgslFn( HAIR_QUERY_SOURCE, [ hairNodeCountFn, hairBoxHitFn, hairNodeFieldFn, hairOrderAtFn, hairSegmentEndsFn, hairSegmentRangeFn, hairPointAtFn, hairIntersectSegmentFn, hairHitStruct ] );

/**
 * Tutte le sorgenti in fila, per chi compila WGSL a mano.
 *
 * L'ordine e' quello delle dipendenze: WGSL vuole che una funzione sia dichiarata
 * prima di essere chiamata, e sbagliarlo non da' un errore leggibile — da' un
 * nome sconosciuto in una riga che sembra a posto.
 */
export const HAIR_SOURCE = [
	HAIR_HIT_SOURCE,
	HAIR_WORD_SOURCE,
	HAIR_NODE_COUNT_SOURCE,
	HAIR_BOUNDS_MIN_SOURCE,
	HAIR_BOUNDS_MAX_SOURCE,
	HAIR_NODE_FIELD_SOURCE,
	HAIR_ORDER_AT_SOURCE,
	HAIR_POINT_AT_SOURCE,
	HAIR_SEGMENT_ENDS_SOURCE,
	HAIR_SEGMENT_RANGE_SOURCE,
	HAIR_BOX_HIT_SOURCE,
	HAIR_INTERSECT_SEGMENT_SOURCE,
	HAIR_SEGMENT_OBJECT_SOURCE,
	HAIR_SEGMENT_RANDOM_SOURCE,
	HAIR_TANGENT_SOURCE,
	HAIR_QUERY_SOURCE,
].join( '\n' );
