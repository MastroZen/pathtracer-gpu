// THIRD PARTY: contains code derived from Cycles (Apache-2.0), modified. Notices and license: NOTICE
import { StructTypeNode } from 'three/webgpu';

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

	// Subsurface: the material whose volume the path is WALKING, or -1 outside. Only the walk:
	// the participating media a path is inside of are the stack below, and a walk through a skin
	// inside a fog is in both
	insideMaterial: 'int',

	// Subsurface: how many steps the walk inside the volume has taken. A walk step is
	// NOT a surface bounce, so it does not spend the bounce budget and carries one of
	// its own - and that budget is the quality knob: it buys depth in a dense medium.
	subsurfaceSteps: 'uint',

	// il colpo in scena viene da un pelo e non da un triangolo: lo copia
	// LogicKernel dal risultato della tracciatura, e lo legge MaterialKernel
	isCurve: 'uint',

	// the probability the NEE choice gave the emitter table at the vertex the traced segment
	// left: LogicKernel writes it, MaterialKernel weighs the emission of the surface with it
	emitterSelectPdf: 'float',

	// 1 when LogicKernel decided the traced segment SCATTERS inside the medium the path is in,
	// at "dist" along it: MaterialKernel then samples the phase function there instead of
	// shading the surface the segment reached (Cycles shade_volume.h)
	mediumScatter: 'uint',

	// how many times the path has scattered inside a medium, against the volume bounce limit
	// of Cycles, which is a budget apart from the surface bounces
	volumeBounce: 'uint',


	// THE DIRECT LIGHT OF A MEDIUM SEGMENT, kept apart from the scatter as Cycles keeps it: the
	// shadow ray LogicKernel queued from a point of the segment, or -1, and what it brings when
	// nothing occludes it. A second channel beside the vertex NEE, resolved the next frame
	mediumShadowIndex: 'int',
	mediumDirect: 'vec3f',

	// THE MEDIUM STACK (nodes/mediumStack.wgsl.js): the media the path is inside of, one word an
	// entry - the material and the object slot, whose coordinates a heterogeneous density uses
	mediumStack: 'vec4u',
	// the weights of the entries' phases at the last scatter point, which MaterialKernel samples
	// the mixture with: LogicKernel or VolumeKernel writes them where the scatter happens
	mediumPhaseWeights: 'vec4f',

	// THE DENOISING FEATURES (Cycles film/denoising_passes.h): what the path owes the guiding passes of
	// the denoiser, packed as halves in four words because every kernel copies this record whole - as
	// three vectors and a flag it grew by twelve words and the render slowed by a tenth even with no
	// denoiser. x and y: the albedo gathered so far and the first component of the WORLD normal; z: the
	// rest of the normal; w: the share still owed (one at the camera, times the specular remainder of a
	// surface that defers the features to the next bounce, a scalar where Cycles keeps a colour) and
	// whether the path still owes any. Blended into the pixel when the path terminates.
	features: 'vec4u',

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
	// the MEDIUM STACK a shadow ray starts with, a copy of its path's (Cycles copies it into the
	// shadow state): the shadow kernel updates it at every boundary it crosses. Bounce rays leave
	// it unread
	mediumStack: 'vec4u',

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
	// ── UNA CURVA NON E' UN TRIANGOLO, e chi ombreggia deve saperlo ──
	//
	// Il colpo su un pelo non ha ne' baricentriche ne' tre indici di vertice: porta
	// un segmento e una posizione lungo di esso. MaterialKernel si costruisce la
	// superficie in un altro modo, e questo e' il segnale.
	//
	// Prende il posto della parola di riempimento che c'era: il record non cresce di
	// un byte, ed e' la ragione per cui la bandiera sta qui invece che altrove.
	isCurve: 'uint',

}, 'TraceResult' );

// A shadow ray's answer, written by TraceShadowRayKernel at the ray's queue index and read by LogicKernel the
// following frame: whether something occludes the light (objectIndex >= 0) and what the surfaces and media it
// crossed let through. It was the bounce record, 64 bytes of which these two fields were read: 16 now, for two
// records a path slot - 54 MB of pool down to 13, and a quarter of the traffic of every shadow pass
export const shadowTraceResultStruct = new StructTypeNode( {

	transmittance: 'vec3f',
	objectIndex: 'int',

}, 'ShadowTraceResult' );

// The ray and shadow ray queues are plain arrays of queued rays, with their append-only lengths in a
// separate counters buffer: since three r187 every storage buffer is bound as an array, which can't
// hold a struct ending in a runtime-sized array.
export const RAY_QUEUE_LENGTH = 0;
export const SHADOW_RAY_QUEUE_LENGTH = 1;
export const QUEUE_COUNTER_COUNT = 2;

// Round-robin queue of pixel indices waiting for a free path slot when the output resolution exceeds
// the ray data pool. Its two counters lead the same u32 array, ahead of the elements.
export const PIXEL_QUEUE_CURRENT = 0;
export const PIXEL_QUEUE_ELEMENT_COUNT = 1;
export const PIXEL_QUEUE_HEADER = 2;
