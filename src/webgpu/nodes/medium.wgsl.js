import { wgslFn } from 'three/tsl';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';

// THE HETEROGENEOUS MEDIUM: a density that a program gives at every point, and the two ways of
// walking it that Cycles uses (kernel/integrator/shade_volume.h). VolumeKernel tracks the path
// segments, the shadow kernel the shadow rays. The coefficients in the record are at density one,
// and the program's value scales them.

// How far a walk goes before it stops: a bound on the work of one segment, and a bias only for a
// medium whose majorant is far below its density
export const MEDIUM_MAX_STEPS = 256;

// The density bound of a heterogeneous medium whose record carries none: weighted tracking stays
// unbiased with a majorant that is too low - it pays in noise - and the app measures one on a grid
// whenever a program drives the density (densityBound), so this is a floor and not an estimate.
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

// THE BLACKBODY of Cycles (svm_math_blackbody_color_rec709, the tables of kernel/tables.h): a
// piecewise fit between 800 and 12000 K, linear Rec.709, NOT clamped - the Principled Volume does
// not clamp it either. The app's materials/blackbody.ts is the same table, and its test the values
export const mediumBlackbodyFn = wgslFn( /* wgsl */ `

	fn mediumBlackbody( t: f32 ) -> vec3f {

		if ( t >= 12000.0 ) { return vec3f( 0.8262954810464208, 0.9945080501520986, 1.566307710274283 ); }
		if ( t < 800.0 ) { return vec3f( 5.413294490189271, -0.20319390035873933, -0.0822535242887164 ); }
		var r = vec3f( 1.61919106e+03, -2.05010916e-03, 5.02995757e+00 );
		var g = vec3f( -4.88999748e+02, 6.04330754e-04, -7.55807526e-02 );
		var b = vec4f( 5.96945309e-11, -4.85742887e-08, -9.70622247e-05, -4.07936148e-03 );
		if ( t >= 6365.0 ) {
			r = vec3f( 3.78717450e+03, 9.35907826e-06, 3.99075871e-01 );
			g = vec3f( -5.00799571e+02, -4.59832026e-06, 1.09098763e+00 );
			b = vec4f( 6.72650283e-13, -2.73078809e-08, 4.24098264e-04, -7.52335691e-01 );
		} else if ( t >= 3315.0 ) {
			r = vec3f( 4.59509185e+03, 2.87495649e-05, 1.50345020e-01 );
			g = vec3f( -1.17554822e+03, -2.16378048e-05, 1.30408023e+00 );
			b = vec4f( -1.61997957e-13, -1.64216008e-08, 3.86216271e-04, -7.38077418e-01 );
		} else if ( t >= 1902.0 ) {
			r = vec3f( 4.67028036e+03, 2.91258199e-05, 1.26703442e-01 );
			g = vec3f( -1.42529332e+03, -4.01150431e-05, 1.43972784e+00 );
			b = vec4f( -1.97075738e-11, 1.75359352e-07, -2.50542825e-04, -2.22783266e-02 );
		} else if ( t >= 1449.0 ) {
			r = vec3f( 4.09461742e+03, -1.27446582e-04, 7.25731635e-01 );
			g = vec3f( -1.26571316e+03, 4.87340896e-06, 1.27054498e+00 );
			b = vec4f( -3.61460868e-11, 2.84822009e-07, -4.93211319e-04, 1.56723440e-01 );
		} else if ( t >= 1167.0 ) {
			r = vec3f( 3.34143193e+03, -4.86551192e-04, 1.76486769e+00 );
			g = vec3f( -1.02363977e+03, 1.20223470e-04, 9.36662319e-01 );
			b = vec4f( -1.40949732e-11, 1.89878968e-07, -3.56632824e-04, 9.10767778e-02 );
		} else if ( t >= 965.0 ) {
			r = vec3f( 2.48845471e+03, -1.11330907e-03, 3.22621544e+00 );
			g = vec3f( -7.55994277e+02, 3.16730098e-04, 4.78306139e-01 );
			b = vec4f( 2.40430366e-11, 5.55021075e-08, -1.98503712e-04, 2.89312858e-02 );
		}
		let tInv = 1.0 / t;
		return vec3f(
			r.x * tInv + r.y * t + r.z,
			g.x * tInv + g.y * t + g.z,
			( ( b.x * t + b.y ) * t + b.z ) * t + b.w,
		);

	}

` );

// THE MEDIUM AT A WORLD POINT: the emission in xyz and the density in w, from ONE run of the
// program of the material with the point's Object and Generated coordinates. A pin whose register
// is 255 keeps the value of the record - the density one, so the coefficients stand as written.
// "objectSlot" 0xffff means unknown, and the world point is used.
export const mediumPointFunc = ( materials, transforms, svmRun, registers ) => wgslTagFn/* wgsl */`

	fn mediumPoint( material: u32, objectSlot: u32, point: vec3f ) -> vec4f {

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
		let packed = ${ materials }[ material ].mediumSvmOutput;
		let densityReg = packed & 0xffu;
		let strengthReg = ( packed >> 8u ) & 0xffu;
		let colorReg = ( packed >> 16u ) & 0xffu;
		let blackbodyReg = ( packed >> 24u ) & 0xffu;
		let temperatureReg = ${ materials }[ material ].mediumSvmOutput2 & 0xffu;
		let last = ${ registers - 1 }u;

		let density = select( 1.0, max( regs[ min( densityReg, last ) ].x, 0.0 ), densityReg != 255u );
		let strength = select( ${ materials }[ material ].mediumEmissionStrength, regs[ min( strengthReg, last ) ].x, strengthReg != 255u );
		let color = select(
			vec3f( ${ materials }[ material ].mediumEmissionColorR, ${ materials }[ material ].mediumEmissionColorG, ${ materials }[ material ].mediumEmissionColorB ),
			regs[ min( colorReg, last ) ].xyz, colorReg != 255u,
		);
		let blackbody = select( ${ materials }[ material ].mediumBlackbodyIntensity, regs[ min( blackbodyReg, last ) ].x, blackbodyReg != 255u );
		let temperature = max( select( ${ materials }[ material ].mediumTemperature, regs[ min( temperatureReg, last ) ].x, temperatureReg != 255u ), 0.0 );

		// the emission of svm_node_principled_volume: strength times colour, plus the blackbody when
		// its intensity is on - sigma times mix( 1, T^4, intensity ) times the colour of T, tinted
		var emission = select( vec3f( 0.0 ), strength * color, strength > 0.0 );
		if ( blackbody > 0.0 ) {

			let t2 = temperature * temperature;
			let power = ${ 5.670373e-8 * 1e-6 / Math.PI } * mix( 1.0, t2 * t2, blackbody );
			let tint = vec3f( ${ materials }[ material ].mediumBlackbodyTintR, ${ materials }[ material ].mediumBlackbodyTintG, ${ materials }[ material ].mediumBlackbodyTintB );
			emission += tint * power * ${ mediumBlackbodyFn }( temperature );

		}
		return vec4f( emission, density );

	}

`;
