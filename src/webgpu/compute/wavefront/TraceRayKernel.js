import { StorageBufferAttribute } from 'three/webgpu';
import { ComputeKernel } from '../ComputeKernel.js';
import { storage, globalId } from 'three/tsl';
import { proxy, wgslTagFn } from 'three-mesh-bvh/webgpu';
import { rngInit } from '../../nodes/random.wgsl.js';
import { rayQueueStruct, intersectionResultStruct } from './structs.js';
import { hairQueryFn, hairTangentFn, hairSegmentObjectFn, hairSegmentRandomFn, EMPTY_HAIR_DATA } from '../../nodes/hair.wgsl.js';

// Pure BVH traversal over the queued bounce rays: one thread per queued ray, writing a compact
// intersection result at the ray's queue index for LogicKernel to consume next frame.
export class TraceRayKernel extends ComputeKernel {

	constructor( ) {

		const params = {
			bvhData: { value: null },

			rayQueue: storage( new StorageBufferAttribute( 1, 1 ), rayQueueStruct ),
			rayIntersectionsStorage: storage( new StorageBufferAttribute( 1, 1 ), intersectionResultStruct ),

			// ── LA PELURIA, in un buffer solo ──
			//
			// WebGPU garantisce otto storage buffer per stage e qui ce ne sono gia'
			// sei: questi due e i quattro del BVH dei triangoli. Il manto ci sta in
			// uno, con gli offset in testa (l'aritmetica sta in src/hair/curvePack.ts).
			//
			// Nasce col pacchetto VUOTO e non con un buffer da un elemento: un buffer
			// va sempre legato, e la traversata legge l'intestazione prima di tutto.
			// Otto parole a zero dicono «nessun nodo» e la query torna subito.
			hairData: storage( new StorageBufferAttribute( EMPTY_HAIR_DATA, 1 ), 'uint' ).toReadOnly(),

			globalId: globalId,
		};

		const raycastOutput = proxy( 'bvhData.value.fns.raycastFirstHit.outputType', params );
		const raycastFirstHitFn = proxy( 'bvhData.value.fns.raycastFirstHit', params );

		const fn = wgslTagFn /* wgsl */`

			fn compute( globalId: vec3u ) -> void {

				let rayQueue = &${ params.rayQueue };
				let rayIntersectionsStorage = &${ params.rayIntersectionsStorage };

				let index = globalId.x;
				if ( index >= rayQueue.length ) {

					return;

				}

				let queuedRay = rayQueue.elements[ index ];
				let indexUV = vec2u( queuedRay.pixelIndex >> 16, queuedRay.pixelIndex & 0xFFFF );
				${ rngInit }( indexUV, queuedRay.seed, queuedRay.currentBounce + queuedRay.alphaDepth );

				let ray = Ray( queuedRay.origin, queuedRay.direction, queuedRay.maxDist );
				var hitResult: ${ raycastOutput };
				let hitTriangle = ${ raycastFirstHitFn }( ray, &hitResult );

				// ── LA PELURIA SI TRACCIA ACCANTO, e vince il colpo piu' vicino ──
				//
				// Due strutture invece di una mista: l'albero dei triangoli e' quello di
				// three-mesh-bvh, coi suoi nodi e il suo macchinario di istanze, e
				// infilarci dentro una primitiva nuova vorrebbe dire dipendere dalle sue
				// interiora. Due traversate e un confronto costano una manciata di
				// istruzioni e non legano niente.
				//
				// Il limite passa al manto: se un triangolo e' gia' stato colpito, un
				// pelo piu' lontano non serve a nessuno e la traversata si pota da sola.
				var limit = 1e30;
				if ( queuedRay.maxDist > 0.0 ) { limit = queuedRay.maxDist; }
				if ( hitTriangle ) { limit = min( limit, hitResult.dist ); }
				let hair = ${ hairQueryFn }( &${ params.hairData }, ray.origin, ray.direction, limit );

				if ( hair.didHit ) {

					// un colpo su un pelo non ha ne' baricentriche ne' tre indici: porta
					// il segmento e quanto si e' lontani dalla radice, e chi ombreggia si
					// costruisce la superficie da quei due
					// ── I TRE CAMPI DEL TRIANGOLO CAMBIANO MESTIERE ──
					//
					// Un colpo su un pelo non ha baricentriche ne' tre indici di vertice, e
					// al loro posto porta quel che serve a ombreggiare una FIBRA: la
					// tangente (l'asse, che per un pelo fa le veci della normale) e quanto
					// si e' lontani dalla radice.
					//
					// Riusare i campi invece di aggiungerne non e' avarizia: la tangente si
					// legge QUI perche' di la' non si puo' — MaterialKernel ha finito gli
					// storage buffer e il pacchetto non glielo si puo' legare. Il segnale e'
					// "isCurve", e senza quello questi campi si leggono come prima.
					rayIntersectionsStorage[ index ].isCurve = 1u;
					// l'oggetto arriva dal SEGMENTO e non da un uniform: i manti di piu'
					// volumi stanno in un buffer solo, e ognuno deve vestirsi col proprio
					rayIntersectionsStorage[ index ].objectIndex = i32( ${ hairSegmentObjectFn }( &${ params.hairData }, hair.segment ) );
					rayIntersectionsStorage[ index ].position = ray.origin + ray.direction * hair.dist;
					rayIntersectionsStorage[ index ].dist = hair.dist;
					rayIntersectionsStorage[ index ].normal = hair.normal;
					rayIntersectionsStorage[ index ].side = 1.0;
					// la terza parola degli indici portava zero: adesso porta il numero della
					// CIOCCA, che chi ombreggia non potrebbe leggersi da se' — il pacchetto
					// della peluria al suo kernel non e' legato
					rayIntersectionsStorage[ index ].indices = vec3u( hair.segment, bitcast<u32>( hair.u ), bitcast<u32>( ${ hairSegmentRandomFn }( &${ params.hairData }, hair.segment ) ) );
					rayIntersectionsStorage[ index ].barycoord = ${ hairTangentFn }( &${ params.hairData }, hair.segment );

				} else if ( hitTriangle ) {

					// la bandiera si RIAZZERA, e non e' pignoleria: il record di un raggio
					// si riusa di colpo in colpo, e una lasciata accesa da un pelo farebbe
					// leggere un triangolo come se fosse una curva
					rayIntersectionsStorage[ index ].isCurve = 0u;
					rayIntersectionsStorage[ index ].barycoord = hitResult.barycoord;
					rayIntersectionsStorage[ index ].objectIndex = i32( hitResult.objectIndex );
					rayIntersectionsStorage[ index ].position = ray.origin + ray.direction * hitResult.dist;
					rayIntersectionsStorage[ index ].dist = hitResult.dist;
					rayIntersectionsStorage[ index ].normal = hitResult.normal.xyz;
					rayIntersectionsStorage[ index ].side = hitResult.side;
					rayIntersectionsStorage[ index ].indices = hitResult.indices.xyz;

				} else {

					rayIntersectionsStorage[ index ].isCurve = 0u;
					rayIntersectionsStorage[ index ].objectIndex = - 1;

				}

			}
		`;

		super( fn( params ) );

		this.defineUniformAccessors( params );

	}

}
