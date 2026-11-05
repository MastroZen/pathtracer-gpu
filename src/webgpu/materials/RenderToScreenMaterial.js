import { MeshBasicNodeMaterial, NoToneMapping, StorageTexture, CustomBlending, OneFactor, OneMinusSrcAlphaFactor } from 'three/webgpu';
import { uv, varying, texture, vec4, toneMapping, uniform, wgslFn, materialOpacity } from 'three/tsl';
import { wgslTagFn } from 'three-mesh-bvh/webgpu';

// TODO: we could fall back to hardware-based filtering if available but it has to be specifically
// requested and available on the renderer which we don't have access to immediately. It's possible this
// can be detected during build time via a custom node to adjust the sample approach used?
const sampleTexelFn = wgslFn( /* wgsl */`
	fn sampleTexel( tex: texture_2d<f32>, coord: vec2f ) -> vec4f {

		// Manual bilinear filtering using textureLoad to support filterable float32 textures
		// on all devices
		let size = vec2f( textureDimensions( tex, 0 ) );

		// the path tracer stores its color rows top down
		let pxCoord = vec2f( coord.x, 1.0 - coord.y ) * size - 0.5;
		let px = vec2i( floor( pxCoord ) );
		let fr = fract( pxCoord );

		// get the four sibling samples
		let s00 = textureLoad( tex, clamp( px,                 vec2i( 0 ), vec2i( size ) - 1 ), 0 );
		let s10 = textureLoad( tex, clamp( px + vec2i( 1, 0 ), vec2i( 0 ), vec2i( size ) - 1 ), 0 );
		let s01 = textureLoad( tex, clamp( px + vec2i( 0, 1 ), vec2i( 0 ), vec2i( size ) - 1 ), 0 );
		let s11 = textureLoad( tex, clamp( px + vec2i( 1, 1 ), vec2i( 0 ), vec2i( size ) - 1 ), 0 );

		// interpolate on the x axis
		let y0 = mix( s00, s10, fr.x );
		let y1 = mix( s01, s11, fr.x );

		// then y
		return mix( y0, y1, fr.y );

	}
` );

// Material to apply tone mapping _before_ applying alpha blending
export class RenderToScreenNodeMaterial extends MeshBasicNodeMaterial {

	get texture() {

		return this._texNode.value;

	}

	set texture( v ) {

		this._texNode.value = v;

	}

	get fromTexture() {

		return this._fromTexNode.value;

	}

	set fromTexture( v ) {

		this._fromTexNode.value = v;

	}

	get toneMapping() {

		return this._toneMapping.toneMapping;

	}

	set toneMapping( v ) {

		this._toneMapping.setToneMapping( v );

	}

	get transition() {

		return this._transitionUniform.value;

	}

	set transition( v ) {

		this._transitionUniform.value = v;

	}

	get exposure() {

		return this._toneMapping.exposureNode.value;

	}

	set exposure( v ) {

		this._toneMapping.exposureNode.value = v;

	}

	constructor( params ) {

		super();

		const texNode = texture( new StorageTexture() );
		this._texNode = texNode;

		const fromTexNode = texture( new StorageTexture() );
		this._fromTexNode = fromTexNode;

		const texUV = varying( uv() );

		const transitionUniform = uniform( 1.0 );
		this._transitionUniform = transitionUniform;

		// NOTE: varyings cannot be referenced directly and must be passed as arguments
		const getFadedColorFn = wgslTagFn/* wgsl */`
			fn fade( uv: vec2f ) -> vec4f {

				if ( ${ transitionUniform } <= 0.0 ) {

					return ${ sampleTexelFn }( ${ fromTexNode }, uv );

				} else if ( ${ transitionUniform } >= 1.0 ) {

					return ${ sampleTexelFn }( ${ texNode }, uv );

				} else {

					let col0 = ${ sampleTexelFn }( ${ fromTexNode }, uv );
					let col1 = ${ sampleTexelFn }( ${ texNode }, uv );
					return mix( col0, col1, ${ transitionUniform } );

				}

			}
		`;

		const toneMappingNode = toneMapping( NoToneMapping, 1.0, getFadedColorFn( texUV ) );
		this._toneMapping = toneMappingNode;

		// the accumulation is PREMULTIPLIED, as in Cycles, so it is blended as such: color + destination * ( 1 - alpha ).
		// The normal blending multiplied the color by the alpha a second time - light in front of a transparent
		// background (an emitting medium before the void, alpha zero) came out black, and the edge of a scattering
		// shell at alpha 0.1 ten times too dark. The fade opacity scales color and alpha together, as a premultiplied
		// color wants
		this.transparent = true;
		this.blending = CustomBlending;
		this.blendSrc = OneFactor;
		this.blendDst = OneMinusSrcAlphaFactor;
		this.blendSrcAlpha = OneFactor;
		this.blendDstAlpha = OneMinusSrcAlphaFactor;
		this.colorNode = vec4( toneMappingNode.rgb.mul( materialOpacity ), toneMappingNode.a );

		this.setValues( params );

	}

}
