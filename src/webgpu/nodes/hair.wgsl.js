// LA QUERY SUL MANTO, IN WGSL: la stessa cosa che `src/hair/curveQuery.ts` fa in Node.
//
// STA QUI E NON NELL'APPLICAZIONE perche' e' il kernel a includerla, e una
// libreria non importa da chi la usa. Il gemello in TypeScript resta di la' come
// riferimento, e `checks/e2e/_sondaCurveWgsl.mjs` li confronta: importa questo
// file da node_modules e quello da src/, e pretende gli stessi numeri.
//
// ── PERCHE' ESISTE DUE VOLTE ──
//
// Non si puo' condividere codice fra TypeScript e WGSL, quindi si condivide la
// RISPOSTA: stessi raggi, stesso manto, stessi numeri. Una trascrizione sbagliata
// non lancia — fa sparire dei peli, o li mette un millimetro piu' in la' — e
// `checks/e2e/_sondaCurveWgsl.mjs` e' il solo modo di vederlo senza guardare una
// pelliccia e sperare.
//
// Quindi questo file si legge ACCANTO a `curveIntersect.ts` e `curveQuery.ts`, e
// ogni riga di la' ha la sua riga qui. Quando una delle due cambia, cambiano
// tutte e due: e' il prezzo del doppio, e la sonda e' quel che lo rende pagabile.
//
// ── UN BUFFER SOLO, E ARRIVA PER PUNTATORE ──
//
// WebGPU garantisce otto storage buffer per stage, e il kernel di tracciamento ne
// usa gia' sei: due suoi e quattro del BVH dei triangoli. La peluria ci sta in
// UNO — `packCurves` mette tutto in un `array<u32>` con gli offset in testa, e le
// funzioni di lettura qui sotto lo aprono. I float ci entrano per
// REINTERPRETAZIONE, e i `bitcast` stanno dentro quelle funzioni: la traversata
// resta leggibile come quando i buffer erano sei.
//
// E si passa come PUNTATORE e non per nome: nel kernel il buffer arriva per
// interpolazione e il suo nome lo decide TSL, quindi un testo che lo nomina si
// lega a un nome che non controlla. WGSL ammette i puntatori nello spazio storage,
// e chi chiama scrive `hairQuery( &buffer, ... )`.

/** Il colpo, in WGSL. Gli stessi campi di `CurveQueryHit`, piu' `didHit`. */
export const hairHitStruct = /* wgsl */`
struct HairHit {
	didHit: bool,
	dist: f32,
	normal: vec3f,
	u: f32,
	segment: u32,
}
`;

/**
 * Un raggio contro un segmento: il cono raccordato da due calotte.
 *
 * Trascrizione di `intersectCurveSegment`. Le due differenze di FORMA — un
 * risultato per valore invece di `null`, e le guardie scritte a mano invece dei
 * ritorni anticipati — sono obbligate dal linguaggio, non scelte.
 */
export const hairIntersectWgsl = /* wgsl */`
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

/**
 * La traversata: trascrizione di `queryCurves`.
 *
 * La pila e' un `array` di dimensione fissa perche' sulla GPU non si alloca — ed
 * e' anche la ragione per cui la versione TypeScript ne ha una uguale invece di
 * ricorrere: le due devono traboccare allo stesso punto, o smettono di dare la
 * stessa risposta proprio nel caso raro.
 */
export const hairQueryWgsl = /* wgsl */`
const HAIR_MAX_STACK: u32 = 32u;

// ── LE LETTURE DAL PACCHETTO ──
//
// L'intestazione sta nelle prime otto parole e dice dove comincia ogni sezione.
// Gli offset si LEGGONO e non si ricalcolano: ricalcolarli vorrebbe dire ripetere
// qui l'aritmetica di curvePack.ts, e due conti che devono tornare uguali sono due
// conti che prima o poi divergono.
fn hairWord( data: ptr<storage, array<u32>, read>, i: u32 ) -> u32 { return (*data)[ i ]; }

fn hairNodeCount( data: ptr<storage, array<u32>, read> ) -> u32 { return hairWord( data, 5u ); }

fn hairBoundsMin( data: ptr<storage, array<u32>, read>, node: u32 ) -> vec3f {

	let at = hairWord( data, 0u ) + node * 6u;
	return vec3f(
		bitcast<f32>( hairWord( data, at ) ),
		bitcast<f32>( hairWord( data, at + 1u ) ),
		bitcast<f32>( hairWord( data, at + 2u ) ),
	);

}

fn hairBoundsMax( data: ptr<storage, array<u32>, read>, node: u32 ) -> vec3f {

	let at = hairWord( data, 0u ) + node * 6u;
	return vec3f(
		bitcast<f32>( hairWord( data, at + 3u ) ),
		bitcast<f32>( hairWord( data, at + 4u ) ),
		bitcast<f32>( hairWord( data, at + 5u ) ),
	);

}

fn hairNodeField( data: ptr<storage, array<u32>, read>, node: u32, field: u32 ) -> u32 {

	return hairWord( data, hairWord( data, 1u ) + node * 4u + field );

}

fn hairOrderAt( data: ptr<storage, array<u32>, read>, i: u32 ) -> u32 { return hairWord( data, hairWord( data, 2u ) + i ); }

/** Il punto porta il suo spessore nella quarta componente: una lettura invece di due. */
fn hairPointAt( data: ptr<storage, array<u32>, read>, i: u32 ) -> vec4f {

	let at = hairWord( data, 3u ) + i * 4u;
	return vec4f(
		bitcast<f32>( hairWord( data, at ) ),
		bitcast<f32>( hairWord( data, at + 1u ) ),
		bitcast<f32>( hairWord( data, at + 2u ) ),
		bitcast<f32>( hairWord( data, at + 3u ) ),
	);

}

fn hairSegmentEnds( data: ptr<storage, array<u32>, read>, s: u32 ) -> vec2u {

	let at = hairWord( data, 4u ) + s * 4u;
	return vec2u( hairWord( data, at ), hairWord( data, at + 1u ) );

}

/** Dove comincia e finisce il segmento lungo la sua CIOCCA. */
fn hairSegmentRange( data: ptr<storage, array<u32>, read>, s: u32 ) -> vec2f {

	let at = hairWord( data, 4u ) + s * 4u;
	return vec2f( bitcast<f32>( hairWord( data, at + 2u ) ), bitcast<f32>( hairWord( data, at + 3u ) ) );

}

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

		} else if ( depth + 2u <= HAIR_MAX_STACK ) {

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

/**
 * Il pacchetto VUOTO: otto parole a zero.
 *
 * Uno storage buffer va SEMPRE legato, anche quando nella scena non c'e' un pelo.
 * Otto parole e non una: la traversata legge l'intestazione, e leggere fuori da un
 * array a lunghezza nota non e' un errore che qualcuno segnala — e' un numero che
 * capita, e con un numero di nodi che capita la traversata gira dentro un albero
 * che non esiste.
 */
export const EMPTY_HAIR_DATA = new Uint32Array( 8 );
