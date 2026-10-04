import { StorageBufferAttribute } from 'three/webgpu';
import { ComputeKernel } from '../ComputeKernel.js';
import { uniform, storage, globalId } from 'three/tsl';
import { proxy, proxyFn, wgslTagFn } from 'three-mesh-bvh/webgpu';
import { rayDataStruct, intersectionResultStruct } from './structs.js';
import { SVM_REGISTERS } from '../../nodes/svm.wgsl.js';
import { clampPathContributionFunc } from '../../nodes/utils.wgsl.js';
import { LIGHT_FAR_DISTANCE } from '../../nodes/lights.wgsl.js';
import { MEDIUM_MAX_STEPS, mediumRandFn, mediumPointFunc } from '../../nodes/medium.wgsl.js';
import { mediumStackFlagsFunc, mediumStackMajorantFunc, mediumStackPointFunc } from '../../nodes/mediumStack.wgsl.js';

// -- THE SEGMENTS THROUGH A HETEROGENEOUS MEDIUM, before LogicKernel --
//
// A path inside a medium whose density is a program cannot be integrated in closed form, so its
// traced segment is walked by WEIGHTED DELTA TRACKING (Cycles volume_integrate_step_scattering,
// after Kutz et al.): tentative collisions at a majorant, and at each one either a scatter or a
// null collision, chosen on the averages of the two coefficients and weighed per channel - which
// keeps the estimate unbiased when the majorant is too low. The emission is gathered at every
// tentative collision with its weight; it does not follow the density, as in Blender, where the
// emission of a Principled Volume fills the whole container until a wire shapes it.
//
// The answer is written where LogicKernel reads it: the throughput times the weight, the scatter
// flag and its distance. LogicKernel then chooses the light from the scatter point, as for a
// homogeneous medium, and integrates nothing itself. The ray data does not grow: a slot is
// already wide enough, and the paths in flight are the buffer over its size.
//
// Its own kernel and not part of LogicKernel, because it runs the program of the density: the
// interpreter costs compile time and registers, and only the scenes with such a medium pay them.
// The register file is the scene's (SVM_REGISTER_BUCKETS), and the tracer rebuilds it with
// SvmKernel's.
export class VolumeKernel extends ComputeKernel {

	constructor( registers = SVM_REGISTERS ) {

		const params = {
			bvhData: { value: null },
			rayCount: uniform( 0, 'uint' ),
			clampDirect: uniform( 0 ),
			clampIndirect: uniform( 10 ),

			rayDataStorage: storage( new StorageBufferAttribute( 1, 1 ), rayDataStruct ),
			rayIntersectionsStorage: storage( new StorageBufferAttribute( 1, 1 ), intersectionResultStruct ),

			globalId: globalId,
		};

		const materials = proxy( 'bvhData.value.storage.materials', params );
		const transforms = proxy( 'bvhData.value.storage.transforms', params );
		const svmRunFn = proxyFn( `bvhData.value.fns.svmRun${ registers }`, params );
		// the density AND the emission of a point, from one run: a wired Temperature or Emission
		// Strength is the fire, and its emission follows the field instead of filling the container
		const gridFn = proxyFn( 'bvhData.value.fns.mediumGrid', params );
		const pointFn = mediumPointFunc( materials, transforms, svmRunFn, registers, gridFn );
		// the path's STACK of media, summed at every point: a homogeneous entry beside a
		// heterogeneous one is walked with it, at its constant coefficients
		const stackFlagsFn = mediumStackFlagsFunc( materials );
		const stackPointFn = mediumStackPointFunc( materials, pointFn );
		const stackMajorantFn = mediumStackMajorantFunc( materials );

		const fn = wgslTagFn/* wgsl */`

			fn compute( rayCount: u32, clampDirect: f32, clampIndirect: f32, globalId: vec3u ) -> void {

				let index = globalId.x;
				if ( index >= rayCount ) {

					return;

				}

				let rayDataStorage = &${ params.rayDataStorage };
				let input = ${ params.rayDataStorage }[ index ];

				// a slot that traced nothing, or a path MaterialKernel ended (a zero pdf): the
				// object of the staged hit is no signal, since a camera ray born last frame has none.
				// A subsurface walk integrates its own volume, and the media around it wait
				if ( input.rayIntersectionIndex < 0 || input.scatterPdf <= 0.0 || input.insideMaterial >= 0 || input.mediumStack.x == 0xffffffffu ) {

					return;

				}
				let stack = input.mediumStack;
				if ( ( ${ stackFlagsFn }( stack ).x & 4u ) == 0u ) {

					return;

				}

				let hit = ${ params.rayIntersectionsStorage }[ u32( input.rayIntersectionIndex ) ];
				let segment = select( ${ LIGHT_FAR_DISTANCE }, hit.dist, hit.objectIndex >= 0 );
				var rng = input.seed * 747796405u + input.pixelIndex * 2891336453u
					+ ( input.currentBounce * 128u + input.alphaDepth ) * 277803737u + 0x5bd1e995u;

				// the throughput the segment starts with: LogicKernel applies the last scatter after
				let beta = input.throughputColor * input.scatterColor / input.scatterPdf;

				var weight = vec3f( 1.0 );
				var scattered = false;
				var t = 0.0;
				var gathered = vec3f( 0.0 );
				var phaseWeights = vec4f( 0.0 );
				// ONE call of the point of the stack, for the emission of a clear stretch and for the
				// walk alike: every call site is a copy of the interpreter in the compiled kernel
				let sigmaBar = ${ stackMajorantFn }( stack );
				let clear = sigmaBar <= 0.0;
				let len = min( segment, 1e4 );
				for ( var k = 0u; k < ${ MEDIUM_MAX_STEPS }u; k ++ ) {

					if ( clear ) {

						// nothing scatters or absorbs: the emission of the stretch, by one point drawn
						// uniformly along it - the integral of a field, estimated without bias
						t = len * ${ mediumRandFn }( &rng );

					} else {

						t -= log( max( 1.0 - ${ mediumRandFn }( &rng ), 1e-9 ) ) / sigmaBar;
						if ( t >= segment ) { break; }

					}

					let point = ${ stackPointFn }( stack, input.origin + input.direction * t, &phaseWeights );
					if ( clear ) {

						gathered = point[ 2 ] * len;
						break;

					}
					gathered += weight * point[ 2 ] / sigmaBar;

					let realScatter = point[ 0 ];
					let nullSigma = vec3f( sigmaBar ) - point[ 1 ];
					let scatterAverage = ( realScatter.r + realScatter.g + realScatter.b ) / 3.0;
					let nullAverage = ( abs( nullSigma.r ) + abs( nullSigma.g ) + abs( nullSigma.b ) ) / 3.0;
					let scatterP = select( 0.0, scatterAverage / ( scatterAverage + nullAverage ), scatterAverage + nullAverage > 0.0 );
					if ( ${ mediumRandFn }( &rng ) < scatterP ) {

						weight *= realScatter / ( sigmaBar * scatterP );
						scattered = true;
						break;

					}
					weight *= nullSigma / ( sigmaBar * ( 1.0 - scatterP ) );
					if ( all( abs( weight ) < vec3f( 1e-6 ) ) ) { break; }

				}

				if ( any( gathered != vec3f( 0.0 ) ) ) {

					let contribution = ${ clampPathContributionFunc }( beta * gathered, max( input.currentBounce, 1u ) - 1u, clampDirect, clampIndirect );
					rayDataStorage[ index ].resultColor = input.resultColor + vec4f( contribution, 0.0 );

				}
				rayDataStorage[ index ].throughputColor = input.throughputColor * weight;
				rayDataStorage[ index ].mediumScatter = select( 0u, 1u, scattered );
				if ( scattered ) {

					rayDataStorage[ index ].dist = t;
					// the weights of the entries' phases AT the collision, for MaterialKernel
					rayDataStorage[ index ].mediumPhaseWeights = phaseWeights;

				}

			}
		`;

		super( fn( params ) );

		this.registers = registers;
		this.defineUniformAccessors( params );

	}

}
