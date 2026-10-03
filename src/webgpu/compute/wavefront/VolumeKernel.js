import { StorageBufferAttribute } from 'three/webgpu';
import { ComputeKernel } from '../ComputeKernel.js';
import { uniform, storage, globalId } from 'three/tsl';
import { proxy, proxyFn, wgslTagFn } from 'three-mesh-bvh/webgpu';
import { rayDataStruct, intersectionResultStruct } from './structs.js';
import { SVM_REGISTERS } from '../../nodes/svm.wgsl.js';
import { clampPathContributionFunc } from '../../nodes/utils.wgsl.js';
import { LIGHT_FAR_DISTANCE } from '../../nodes/lights.wgsl.js';
import { MEDIUM_MAX_STEPS, mediumRandFn, mediumDensityFunc, mediumMajorantFunc } from '../../nodes/medium.wgsl.js';

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
		const densityFn = mediumDensityFunc( materials, transforms, svmRunFn, registers );
		const majorantFn = mediumMajorantFunc( densityFn, materials );

		const fn = wgslTagFn/* wgsl */`

			fn compute( rayCount: u32, clampDirect: f32, clampIndirect: f32, globalId: vec3u ) -> void {

				let index = globalId.x;
				if ( index >= rayCount ) {

					return;

				}

				let rayDataStorage = &${ params.rayDataStorage };
				let input = ${ params.rayDataStorage }[ index ];

				// a slot that traced nothing, or a path MaterialKernel ended (a zero pdf): the
				// object of the staged hit is no signal, since a camera ray born last frame has none
				if ( input.rayIntersectionIndex < 0 || input.scatterPdf <= 0.0 || input.insideMaterial < 0 ) {

					return;

				}
				let material = u32( input.insideMaterial );
				if ( ( ${ materials }[ material ].mediumFlags & 6u ) != 6u ) {

					return;

				}

				let hit = ${ params.rayIntersectionsStorage }[ u32( input.rayIntersectionIndex ) ];
				let segment = select( ${ LIGHT_FAR_DISTANCE }, hit.dist, hit.objectIndex >= 0 );
				let objectSlot = select( 0xffffu, u32( input.insideObject ), input.insideObject >= 0 );
				var rng = input.seed * 747796405u + input.pixelIndex * 2891336453u
					+ ( input.currentBounce * 128u + input.alphaDepth ) * 277803737u + 0x5bd1e995u;

				let sigmaS = vec3f( ${ materials }[ material ].mediumScatterR, ${ materials }[ material ].mediumScatterG, ${ materials }[ material ].mediumScatterB );
				let sigmaT = sigmaS + vec3f( ${ materials }[ material ].mediumAbsorptionR, ${ materials }[ material ].mediumAbsorptionG, ${ materials }[ material ].mediumAbsorptionB );
				let emission = vec3f( ${ materials }[ material ].mediumEmissionR, ${ materials }[ material ].mediumEmissionG, ${ materials }[ material ].mediumEmissionB );
				// the throughput the segment starts with: LogicKernel applies the last scatter after
				let beta = input.throughputColor * input.scatterColor / input.scatterPdf;
				let unitMax = max( sigmaT.r, max( sigmaT.g, sigmaT.b ) );

				var weight = vec3f( 1.0 );
				var scattered = false;
				var t = 0.0;
				var gathered = vec3f( 0.0 );
				if ( unitMax <= 0.0 ) {

					// nothing scatters or absorbs: the emission of the whole stretch
					gathered = emission * min( segment, 1e4 );

				} else {

					let sigmaBar = unitMax * ${ majorantFn }( material, objectSlot, input.origin, input.direction, min( segment, 1e4 ), &rng );
					for ( var k = 0u; k < ${ MEDIUM_MAX_STEPS }u; k ++ ) {

						t -= log( max( 1.0 - ${ mediumRandFn }( &rng ), 1e-9 ) ) / sigmaBar;
						if ( t >= segment ) { break; }

						let density = ${ densityFn }( material, objectSlot, input.origin + input.direction * t );
						gathered += weight * emission / sigmaBar;

						let realScatter = sigmaS * density;
						let nullSigma = vec3f( sigmaBar ) - sigmaT * density;
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

				}

				if ( any( gathered != vec3f( 0.0 ) ) ) {

					let contribution = ${ clampPathContributionFunc }( beta * gathered, max( input.currentBounce, 1u ) - 1u, clampDirect, clampIndirect );
					rayDataStorage[ index ].resultColor = input.resultColor + vec4f( contribution, 0.0 );

				}
				rayDataStorage[ index ].throughputColor = input.throughputColor * weight;
				rayDataStorage[ index ].mediumScatter = select( 0u, 1u, scattered );
				if ( scattered ) {

					rayDataStorage[ index ].dist = t;

				}

			}
		`;

		super( fn( params ) );

		this.registers = registers;
		this.defineUniformAccessors( params );

	}

}
