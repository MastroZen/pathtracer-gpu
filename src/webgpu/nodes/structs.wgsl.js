import { wgsl } from 'three/tsl';
import { StructTypeNode } from 'three/webgpu';

// TODO: Move to node-based constants that are embedded with wgslTagFn
export const constants = wgsl( /* wgsl */ `

	const PI: f32 = 3.141592653589793;
	const EPSILON: f32 = 1e-5;
	const MIN_ROUGHNESS: f32 = 1e-3;
	const MIN_INCIDENT_COS: f32 = 1e-3;

` );

export const scatterRecordStruct = new StructTypeNode( {
	color: 'vec3f',
	isTransmissive: 'bool',
	direction: 'vec3f',
	pdf: 'float',
}, 'ScatterRecord' );

// NOTE: all "*Map" fields are bit-packed integers of texture index & settings:
// bits 0-22: texture index
// bits 23-25: uv channel
// bits 26-27: wrapS setting
// bits 28-29: wrapT setting
// bit 30: filter setting
// bit 31: negative bit used for indicating unused texture
export const materialStruct = new StructTypeNode( {
	// offset 0
	color: 'vec3',
	map: 'int',

	// offset 4 floats
	metalness: 'float',
	metalnessMap: 'int',

	roughness: 'float',
	roughnessMap: 'int',

	// offset 8 floats
	ior: 'float',
	transmission: 'float',
	transmissionMap: 'int',

	emissiveIntensity: 'float',
	// offset 12 floats
	emissive: 'vec3',
	emissiveMap: 'int',

	// offset 16 floats
	normalMap: 'int',
	normalScale: 'vec2',

	// offset 20 floats
	clearcoat: 'float',
	clearcoatMap: 'int',
	clearcoatNormalMap: 'int',
	// offset 24 floats
	clearcoatNormalScale: 'vec2',
	clearcoatRoughness: 'float',
	clearcoatRoughnessMap: 'int',

	// offset 28 floats
	iridescenceMap: 'int',
	iridescenceThicknessMap: 'int',
	iridescence: 'float',
	iridescenceIor: 'float',
	// offset 32 floats
	iridescenceThicknessMinimum: 'float',
	iridescenceThicknessMaximum: 'float',
	diffuseRoughness: 'float',
	dispersion: 'float',

	// offset 36 floats
	specularColor: 'vec3',
	specularColorMap: 'int',

	// offset 40 floats
	specularIntensity: 'float',
	specularIntensityMap: 'int',
	thinWall: 'int', // actually a boolean

	// offset 44 floats
	attenuationColor: 'vec3',
	attenuationDistance: 'float',

	// offset 48 floats
	alphaMap: 'int',

	castShadow: 'int', // actually a boolean
	opacity: 'float',
	alphaTest: 'float',

	// offset 52 floats
	side: 'float',
	matte: 'int', // actually a boolean

	sheen: 'float',
	// offset 56 floats
	sheenColor: 'vec3',
	sheenColorMap: 'int',
	// offset 60 floats
	sheenRoughness: 'float',
	sheenRoughnessMap: 'int',

	// All those are booleans
	vertexColors: 'int',
	flatShading: 'int',
	// offset 64 floats
	transparent: 'int',
	fogVolume: 'int',
	// offset 66 floats
	anisotropy: 'float',
	anisotropyRotation: 'float',
	// offset 68 floats
	anisotropyMap: 'int',

	// offset 72 floats
	mapTransform: 'mat3',
	// offset 84 floats
	metalnessMapTransform: 'mat3',
	// offset 96 floats
	roughnessMapTransform: 'mat3',
	// offset 108 floats
	transmissionMapTransform: 'mat3',
	// offset 120 floats
	emissiveMapTransform: 'mat3',
	// offset 132 floats
	normalMapTransform: 'mat3',
	// offset 144 floats
	clearcoatMapTransform: 'mat3',
	// offset 156 floats
	clearcoatNormalMapTransform: 'mat3',
	// offset 168 floats
	clearcoatRoughnessMapTransform: 'mat3',
	// offset 180 floats
	sheenColorMapTransform: 'mat3',
	// offset 192 floats
	sheenRoughnessMapTransform: 'mat3',
	// offset 204 floats
	iridescenceMapTransform: 'mat3',
	// offset 216 floats
	iridescenceThicknessMapTransform: 'mat3',
	// offset 228 floats
	specularColorMapTransform: 'mat3',
	// offset 240 floats
	specularIntensityMapTransform: 'mat3',
	// offset 252 floats
	alphaMapTransform: 'mat3',

	// offset 264 floats
	anisotropyMapTransform: 'mat3',

	// offset 276 floats
	// Mix Shader: this material can stand for a stochastic choice between itself and
	// another one in the table. "mixWeight" is the probability of taking "mixIndex";
	// zero means there is no mix and the record is used as is. Picking one branch per
	// hit rather than evaluating both is how Cycles handles Mix Shader
	// (surface_shader_bsdf_bssrdf_pick): the estimator stays unbiased and a hit still
	// costs a single BSDF.
	mixWeight: 'float',
	mixIndex: 'int',
	// Padded by hand to a multiple of four, the way transformStruct does it: the
	// writer advances one field at a time through a buffer strided by getLength(),
	// so a struct that does not land on the stride shifts every record after the
	// first. It shows up as a mix that renders darker than either branch, which
	// looks like a physics bug and is an offset bug.
	// A wired Fac is a MASK: the index of its texture lives here. No transform beside
	// it, unlike the other maps — a baked mask has no offset or repeat, and a mat3 per
	// material to carry an identity is twelve floats each.
	mixMap: 'int',
	// Subsurface: how often a hit goes INTO the volume instead of scattering off the
	// surface. It is a closure among the others, picked the same way the mix is — which
	// is what Cycles does, and the reason this sits next to the mix fields.
	subsurfaceWeight: 'float',

	// offset 280 floats
	// The mean free path of the walk, one per channel and already multiplied by the
	// scale: it is a DISTANCE in world units, so it is the average step taken inside
	// the volume before scattering again. A radius near zero is a surface, not a
	// medium, and the kernel falls back to crossing in a straight line.
	subsurfaceRadius: 'vec3',
	// The Henyey-Greenstein g: 0 scatters in every direction, positive keeps going
	// forward. Skin is around 0.8 in Blender.
	subsurfaceAnisotropy: 'float',

	// offset 284: i PELI.
	//
	// Stanno sul materiale e non altrove perche' il kernel legge di qui: il
	// pacchetto della peluria e' legato ai kernel di tracciamento, non a quello dei
	// materiali. Il colore invece NON e' qui — e' `color`, lo stesso della
	// superficie, che per un pelo diventa assorbimento.
	//
	// Le due ruvidita' sono separate come di la': quella lungo la ciocca e quella
	// attorno alla sezione fanno due cose diverse, e una sola non basta.
	hairRoughness: 'float',
	hairRadialRoughness: 'float',
	// l'inclinazione della cuticola, in RADIANTI: e' lei a staccare il secondo
	// riflesso dal primo
	hairTilt: 'float',
	// quanto varia da una ciocca all'altra, 0..1
	hairRandomColor: 'float',
	hairRandomRoughness: 'float',
	// i due PIGMENTI, che si sommano al colore: eumelanina e feomelanina, come
	// dentro il modo a pigmenti di Cycles
	hairMelanin: 'float',
	hairRedness: 'float',
	// la vernice (quanto stringere il solo riflesso primario) e l'indice di
	// rifrazione della fibra
	hairCoat: 'float',
	hairIor: 'float',

	// ── E TRE PAROLE DI RIEMPIMENTO, che NON sono pignoleria ──
	//
	// La struct contiene dei `vec3`, quindi il suo passo e' allineato a quattro
	// float: `getLength()` risponde 296 anche se i campi sono 293. Chi scrive il
	// buffer avanza di un campo per volta, quindi senza queste tre ogni materiale
	// dopo il primo viene LETTO spostato di tre parole — e non e' un errore, e'
	// una scena intera coi materiali sbagliati.
	//
	// Misurato: aggiungendo i nove campi dei peli e fermandosi a 293, una scena
	// SENZA un pelo e' passata da 77 a 31 di luminanza media. Il totale va tenuto
	// multiplo di quattro, e il controllo qui sotto lo pretende.
	// quale MODELLO: 0 e' Chiang, 1 e' Huang. Un float perche' il record e' un
	// buffer di float, e il numero e' quello che il ponte ha gia' tradotto.
	hairModel: 'float',
	// il rapporto fra asse minore e maggiore della sezione: lo usa il solo Huang
	hairAspect: 'float',
	// COME SI DA' IL COLORE: 0 riflettanza, 1 melanina, 2 assorbimento. E' il
	// parametrization del nodo Principled Hair, e i tre modi SI ESCLUDONO — prima
	// melanina e colore si sommavano, cioe' due manopole sulla stessa cosa.
	hairParametrization: 'float',
	// ── SEI FLOAT E NON DUE vec3, ed e' una scelta con un numero accanto ──
	//
	// Un vec3 si allinea a 16 byte, quindi apre un BUCO che lo scrittore del
	// record non vede: dichiarati come vettori, la struct chiedeva 308 parole
	// mentre il writer ne scriveva 304, e il tracciatore rifiutava ogni materiale
	// (preso dalla guardia a runtime, che per questo esiste). Compensare il buco
	// vorrebbe dire tenere a mano due tabelle di offset in due file; sei float
	// hanno allineamento uno e il problema non e' rappresentabile.
	//
	// La TINTA si somma ai pigmenti, e vale solo nel modo melanina: e' il termine
	// che Cycles somma la' dentro (sigma = melanina + tinta).
	hairTintR: 'float',
	hairTintG: 'float',
	hairTintB: 'float',
	// il coefficiente di assorbimento NUDO, e vale solo nel modo assorbimento: tre
	// numeri e non un colore, perche' non sta in 0..1 (di la' arriva a 1000)
	hairAbsorptionR: 'float',
	hairAbsorptionG: 'float',
	hairAbsorptionB: 'float',
	// The density per unit area with which next event estimation picks a point of this
	// material, when its triangles are in the emitter table (emitters.js): the luminance of
	// its emission over the power of the whole table. Zero keeps its emission bsdf-sampled
	// at full weight. It took the first of the two padding words, so the record keeps its
	// size.
	emitterAreaPdf: 'float',
	// e il RIEMPIMENTO che porta il record al passo della struct: i `vec3` la
	// allineano a quattro float, quindi il totale va tenuto multiplo di quattro.
	// Il controllo a runtime nel writer conta le parole scritte e le confronta.
	_hairAlignment1: 'float',
	// total size = 304
}, 'Material' );

export const surfaceRecordStruct = new StructTypeNode( {
	// surface type
	volumeParticle: 'bool',

	// geometry
	faceNormal: 'vec3f',
	frontFace: 'bool',
	normal: 'vec3f',
	normalBasis: 'mat3x3f',
	normalInvBasis: 'mat3x3f',

	// cached properties
	eta: 'f32',
	f0: 'f32',

	// material
	roughness: 'f32',
	diffuseRoughness: 'f32',
	metalness: 'f32',
	anisotropy: 'f32',
	color: 'vec3f',
	emission: 'vec3f',
	opacity: 'f32',

	// transmission
	ior: 'f32',
	transmission: 'f32',
	thinWall: 'bool',
	attenuationColor: 'vec3f',
	attenuationDistance: ' f32',

	// clearcoat
	clearcoatNormal: 'vec3f',
	clearcoatBasis: 'mat3x3f',
	clearcoatInvBasis: 'mat3x3f',
	clearcoat: 'f32',
	clearcoatRoughness: 'f32',

	// sheen
	sheen: 'f32',
	sheenColor: 'vec3f',
	sheenRoughness: 'f32',

	// iridescence
	iridescence: 'f32',
	iridescenceIor: 'f32',
	iridescenceThickness: 'f32',

	// specular
	specularColor: 'vec3f',
	specularIntensity: 'f32',
}, 'SurfaceRecord' );

export const environmentInfoStruct = new StructTypeNode( {
	rotation: 'mat3x3f',
	intensity: 'float',
	blur: 'float',
	totalSum: 'float',
}, 'EnvironmentInfo' );

// Result of importance-sampling the environment: a world-space direction and the
// pdf (in solid-angle measure) of having drawn it. No radiance on purpose: the
// kernels read it from sampleColor, the SAME function an escaping bsdf ray reads, so
// the two strategies of the MIS see one environment - and a wrapper around
// sampleColor (a sun disc added from outside) reaches both.
export const environmentSampleStruct = new StructTypeNode( {
	direction: 'vec3f',
	pdf: 'float',
}, 'EnvironmentSample' );

// A single scene light unpacked from the lights data texture.
export const lightStruct = new StructTypeNode( {
	position: 'vec3f',
	lightType: 'int',
	color: 'vec3f',
	intensity: 'float',
	u: 'vec3f',
	v: 'vec3f',
	area: 'float',
	radius: 'float',
	decay: 'float',
	distance: 'float',
	coneCos: 'float',
	penumbraCos: 'float',
	iesProfile: 'int',
}, 'Light' );

// Result of sampling (or intersecting) a light: a world-space direction toward the light,
// the distance to it, its emitted radiance, and the pdf (solid-angle measure) of the sample.
export const lightRecordStruct = new StructTypeNode( {
	dist: 'float',
	direction: 'vec3f',
	pdf: 'float',
	emission: 'vec3f',
	lightType: 'int',
}, 'LightRecord' );

export const lobeWeightsStruct = new StructTypeNode( {
	diffuse: 'float',
	specular: 'float',
	transmission: 'float',
	clearcoat: 'float',
}, 'LobeWeights' );

export const bxdfContextStruct = new StructTypeNode( {

	V: 'vec3f', // view dir
	L: 'vec3f', // light dir
	H: 'vec3f', // half dir

	VdotH: 'float',

}, 'BxDFContext' );
