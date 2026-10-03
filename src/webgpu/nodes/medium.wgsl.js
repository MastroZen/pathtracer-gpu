import { wgslFn } from 'three/tsl';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';

// THE HETEROGENEOUS MEDIUM: a density that a program gives at every point, and the two ways of
// walking it that Cycles uses (kernel/integrator/shade_volume.h). VolumeKernel tracks the path
// segments, the shadow kernel the shadow rays. The coefficients in the record are at density one,
// and the program's value scales them.

// How far a walk goes before it stops: a bound on the work of one segment, and a bias only for a
// medium whose majorant is far below its density
export const MEDIUM_MAX_STEPS = 256;

// The density majorant of a segment, the way Cycles estimates one when nothing bounds the shader
// (volume_estimate_extrema: four stratified points). Weighted tracking stays unbiased with a
// majorant that is too low - it pays in noise - so the estimate gets a margin, and a FLOOR, since
// four points can miss a thin wisp entirely and a majorant of zero would never look again.
export const MEDIUM_MAJORANT_SAMPLES = 4;
export const MEDIUM_MAJORANT_MARGIN = 1.5;
export const MEDIUM_MAJORANT_FLOOR = 0.25;

// A generator of its own: a walk draws a variable number of values, and the reserved dimensions
// of the sequence are few. Seeded from the path, as the Huang hair BSDF does.
export const mediumRandFn = wgslFn( /* wgsl */ `

	fn mediumRand( state: ptr<function, u32> ) -> f32 {

		*state = *state * 747796405u + 2891336453u;
		var word = ( ( *state >> ( ( *state >> 28u ) + 4u ) ) ^ *state ) * 277803737u;
		word = ( word >> 22u ) ^ word;
		return f32( word >> 8u ) / 16777216.0;

	}

` );

// The density at a world point: the program of the material, run with the point's Object and
// Generated coordinates. "objectSlot" 0xffff means unknown, and the world point is used.
export const mediumDensityFunc = ( materials, transforms, svmRun, registers ) => wgslTagFn/* wgsl */`

	fn mediumDensity( material: u32, objectSlot: u32, point: vec3f ) -> f32 {

		var objectPos = point;
		if ( objectSlot != 0xffffu ) {

			objectPos = ( ${ transforms }[ objectSlot ].inverseMatrixWorld * vec4f( point, 1.0 ) ).xyz;

		}
		let spaceMin = vec3f( ${ materials }[ material ].mediumSpaceMinX, ${ materials }[ material ].mediumSpaceMinY, ${ materials }[ material ].mediumSpaceMinZ );
		let spaceSize = vec3f( ${ materials }[ material ].mediumSpaceSizeX, ${ materials }[ material ].mediumSpaceSizeY, ${ materials }[ material ].mediumSpaceSizeZ );
		let generated = ( objectPos - spaceMin ) / max( spaceSize, vec3f( 1e-6 ) );

		var regs: array<vec4f, ${ registers }>;
		_ = ${ svmRun }(
			${ materials }[ material ].mediumSvmCode, ${ materials }[ material ].mediumSvmCount, ${ materials }[ material ].mediumSvmConsts,
			vec2f( 0.0 ), mat3x3f( generated, vec3f( 0.0 ), vec3f( 0.0 ) ), mat3x3f( objectPos, vec3f( 0.0 ), vec3f( 0.0 ) ), vec3f( 0.0 ), &regs,
		);
		let outReg = ${ materials }[ material ].mediumSvmOutput;
		return select( 0.0, max( regs[ min( outReg, ${ registers - 1 }u ) ].x, 0.0 ), outReg != 255u );

	}

`;

// The majorant of the density along a segment: the largest of four stratified samples, with the
// margin and the floor above
export const mediumMajorantFunc = ( densityFn, materials ) => wgslTagFn/* wgsl */`

	fn mediumMajorant( material: u32, objectSlot: u32, origin: vec3f, direction: vec3f, len: f32, rng: ptr<function, u32> ) -> f32 {

		// the bound the host measured on a grid, when there is one: it does not undershoot where
		// the grid looked, and a weight that never goes negative is a pixel that never goes black
		let measured = ${ materials }[ material ].mediumDensityMax;
		if ( measured > 0.0 ) { return measured; }

		let offset = ${ mediumRandFn }( rng );
		var largest = 0.0;
		for ( var i = 0u; i < ${ MEDIUM_MAJORANT_SAMPLES }u; i ++ ) {

			let t = len * ( f32( i ) + offset ) / ${ MEDIUM_MAJORANT_SAMPLES }.0;
			largest = max( largest, ${ densityFn }( material, objectSlot, origin + direction * t ) );

		}
		return max( largest * ${ MEDIUM_MAJORANT_MARGIN }, ${ MEDIUM_MAJORANT_FLOOR } );

	}

`;

// THE TRANSMITTANCE of a stretch by RATIO TRACKING (Cycles volume_transmittance with null
// scattering): tentative collisions at the majorant, each multiplying by one minus the ratio of
// the true extinction to the majorant. Per channel, with the majorant of the largest; a factor
// may be negative when the majorant is too low, and the estimate stays unbiased.
export const mediumRatioTrackingFunc = ( materials, densityFn, majorantFn ) => wgslTagFn/* wgsl */`

	fn mediumRatioTracking( material: u32, objectSlot: u32, origin: vec3f, direction: vec3f, len: f32, rng: ptr<function, u32> ) -> vec3f {

		let unit = vec3f(
			${ materials }[ material ].mediumScatterR + ${ materials }[ material ].mediumAbsorptionR,
			${ materials }[ material ].mediumScatterG + ${ materials }[ material ].mediumAbsorptionG,
			${ materials }[ material ].mediumScatterB + ${ materials }[ material ].mediumAbsorptionB,
		);
		let unitMax = max( unit.r, max( unit.g, unit.b ) );
		if ( unitMax <= 0.0 ) { return vec3f( 1.0 ); }

		let sigmaBar = unitMax * ${ majorantFn }( material, objectSlot, origin, direction, min( len, 1e4 ), rng );
		var transmittance = vec3f( 1.0 );
		var t = 0.0;
		for ( var k = 0u; k < ${ MEDIUM_MAX_STEPS }u; k ++ ) {

			t -= log( max( 1.0 - ${ mediumRandFn }( rng ), 1e-9 ) ) / sigmaBar;
			if ( t >= len ) { break; }
			let density = ${ densityFn }( material, objectSlot, origin + direction * t );
			transmittance *= 1.0 - unit * density / sigmaBar;
			if ( all( abs( transmittance ) < vec3f( 1e-5 ) ) ) { break; }

		}
		return transmittance;

	}

`;

/** A material and an object slot in the one word a queued shadow ray carries: -1 is no medium. */
export const MEDIUM_NO_OBJECT = 0xffff;
