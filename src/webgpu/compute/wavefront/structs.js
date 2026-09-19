import { StructTypeNode } from 'three/webgpu';

// StructTypeNode that force-emits dependency structs. TSL can't discover a struct referenced only via
// a type string, so we build each dependency first, which registers its definition ahead of this struct.
// TODO: remove this once TSL resolves struct dependencies from string member types itself.
class DependentStructTypeNode extends StructTypeNode {

	constructor( members, name, dependencies = [] ) {

		super( members, name );
		this.dependencies = dependencies;

	}

	setup( builder ) {

		for ( const dep of this.dependencies ) {

			dep.build( builder );

		}

		super.setup( builder );

	}

}

// Persistent per-path-slot state. One slot per in-flight path, holding everything needed to resolve
// the previous frame's trace results and stage the next bounce.
export const rayDataStruct = new StructTypeNode( {

	throughputColor: 'vec3f',
	currentBounce: 'uint',

	resultColor: 'vec4f',

	emission: 'vec3f',
	scatterPdf: 'float',

	scatterColor: 'vec3f',

	// the smallest scatter pdf seen along the path, for the glossy filter
	minPdf: 'float',

	origin: 'vec3f',

	// whether every scatter so far has been transmissive, for the background modes
	isFullyTransmissive: 'uint',

	direction: 'vec3f',
	side: 'float',

	normal: 'vec3f',
	objectIndex: 'int',

	barycoord: 'vec3f',
	pixelIndex: 'uint',

	indices: 'vec3u',
	seed: 'uint',

	lightDirection: 'vec3f',
	lightPdf: 'float',

	lightEmission: 'vec3f',
	lightDist: 'float',

	lightBsdf: 'vec3f',
	lightBsdfPdf: 'float',

	rayIntersectionIndex: 'int',
	shadowRayIntersectionIndex: 'int',
	lightType: 'int',

	// the traced segment length, for the transmission attenuation applied by MaterialKernel
	dist: 'float',

	// alpha test pass throughs, counted separately from the bounce count
	alphaDepth: 'uint',

	// the camera segment's maximum trace distance, carried across alpha pass throughs
	maxDist: 'float',

	// the path's hero wavelength, negative until its reconstruction weight is applied at the
	// first dispersive interaction
	dispersionWavelength: 'float',

	// Subsurface: the material the path is currently travelling INSIDE, or -1 outside.
	// The medium is otherwise only implied — the transmission attenuation infers it from the
	// side of the exit hit — and a walk through the volume needs to know it before the exit.
	insideMaterial: 'int',

	// Subsurface: how many steps the walk inside the volume has taken. A walk step is
	// NOT a surface bounce, so it does not spend the bounce budget and carries one of
	// its own - and that budget is the quality knob: it buys depth in a dense medium.
	subsurfaceSteps: 'uint',

	// ── QUEL CHE DEVE VIAGGIARE, E NIENTE DI PIU' ──
	//
	// Cycles decide e risolve un segmento dentro un giro solo del suo ciclo. Qui il
	// segmento NASCE a un passo e si RISOLVE al successivo, perche' in mezzo c'e' il
	// lancio del raggio, che e' un altro dispatch. Quindi quel che il peso vuole deve
	// viaggiare col cammino — ma SOLO quel che dipende dalla direzione: il canale e la
	// distanza si pescano al momento della risoluzione, dove il throughput e' quello
	// vero. Portarli invece che pescarli costa il 18% di rumore, misurato.
	//
	// "subsurfaceNormal" e' la normale ESTERNA da cui il cammino e' entrato: e' il
	// verso in cui la guida pende, e l'unica cosa che un punto dentro il volume sa su
	// dove sia la superficie.
	subsurfaceNormal: 'vec3f',
	// 1 - cos / v, lo stiramento di Dwivedi di questa direzione
	subsurfaceStretch: 'float',

	// la pdf guidata diviso quella classica, col valore della fase gia' semplificato
	subsurfacePdfFactor: 'float',
	// la guida ha estratto DAVVERO questa direzione? decide se la distanza si stira
	subsurfaceGuided: 'uint',
	// Quanti metri di materia ci sono fra l'ingresso e la parete opposta, misurati
	// lungo la normale d'ingresso. Non costa un raggio in piu': il PRIMO segmento del
	// cammino la attraversa per intero, e la sua distanza tracciata e' esattamente
	// questa. Serve a sapere se il mezzo e' otticamente SPESSO, che e' l'unico posto
	// dove la guida di Dwivedi paga.
	subsurfaceOpposite: 'float',

}, 'RayData' );

// A ray queued for BVH traversal by the trace kernels. A "maxDist" of zero traces unbounded.
export const traceQueuedRayStruct = new StructTypeNode( {

	origin: 'vec3f',
	pixelIndex: 'uint',

	direction: 'vec3f',
	currentBounce: 'uint',

	seed: 'uint',
	alphaDepth: 'uint',
	maxDist: 'float',
	_alignment0: 'uint',

}, 'TraceQueuedRay' );

// Compact trace result, written by the trace kernels at the ray's queue index and consumed by
// LogicKernel the following frame. objectIndex < 0 encodes a miss.
export const intersectionResultStruct = new StructTypeNode( {

	barycoord: 'vec3f',
	objectIndex: 'int',

	position: 'vec3f',
	dist: 'float',

	normal: 'vec3f',
	side: 'float',

	indices: 'vec3u',
	_alignment0: 'uint',

}, 'TraceResult' );

// Queue wrappers that keep an append-only length counter in a header ahead of the elements. The
// atomic variant is bound where rays are pushed with atomicAdd; the plain one where the length is
// only read or reset. getLength returns just the header since the trailing array is runtime-sized.
export const rayQueueStruct = new DependentStructTypeNode( {
	length: 'uint',
	elements: `array<${ traceQueuedRayStruct.name }>`,
}, 'RayQueue', [ traceQueuedRayStruct ] );
rayQueueStruct.getLength = () => 4;

export const rayQueueAtomicStruct = new DependentStructTypeNode( {
	length: { type: 'uint', atomic: true },
	elements: `array<${ traceQueuedRayStruct.name }>`,
}, 'RayQueue', [ traceQueuedRayStruct ] );
rayQueueAtomicStruct.getLength = () => 4;

// Round-robin queue of pixel indices waiting for a free path slot when the output resolution exceeds
// the ray data pool. The non-atomic variant is used for contention-free initialization.
export const pixelQueueStruct = new DependentStructTypeNode( {
	current: { type: 'uint', atomic: true },
	elementCount: 'uint',
	elements: 'array<atomic<u32>>',
}, 'PixelQueue' );
pixelQueueStruct.getLength = () => 2;

export const pixelQueueNonAtomicStruct = new DependentStructTypeNode( {
	current: 'uint',
	elementCount: 'uint',
	elements: 'array<u32>',
}, 'PixelQueue' );
pixelQueueNonAtomicStruct.getLength = () => 2;
