import { ColorManagement, FloatType, RGBAFormat } from 'three';
import { RedIntegerFormat, StorageTexture, UnsignedIntType } from 'three/webgpu';
import { StorageBufferAttribute } from 'three/webgpu';
import { ZeroOutKernel } from './compute/ZeroOutKernel.js';
import { EMPTY_HAIR_DATA } from './nodes/hair.wgsl.js';
import { SUBSURFACE_MAX_STEPS } from './nodes/material.wgsl.js';

export class PathTracerBackend {

	constructor( renderer ) {

		this.renderer = renderer;
		this.maxBounces = 15;

		// how much work a single update dispatches, in path slots. Lower values keep frames
		// responsive on slower hardware; the unit is resolution independent.
		this.frameBudget = 250000;
		this.maxTransparentBounces = 15;

		// Subsurface: how many steps the walk inside a volume may take before the path is
		// dropped. It is the quality knob of the random walk - a dense medium needs many
		// short steps to reach the other side, and stopping early loses that light.
		this.maxSubsurfaceSteps = SUBSURFACE_MAX_STEPS;

		// ── LA PELURIA ──
		//
		// Un pacchetto solo, gia' impacchettato da chi lo genera (src/hair/curvePack.ts
		// dell'applicazione): albero, punti e segmenti in un "Uint32Array". Il tracer non
		// lo costruisce e non lo interpreta — lo lega e basta, ed e' il motivo per cui il
		// generatore puo' stare fuori da qui.
		//
		// Nasce VUOTO: uno storage buffer va sempre legato, e otto parole a zero dicono
		// alla traversata «nessun nodo».
		this.hairAttribute = new StorageBufferAttribute( EMPTY_HAIR_DATA, 1 );
		this.lowResMode = false;

		// stop taking samples once a pixel reaches this count. zero means no limit.
		this.maxSamples = 0;

		this._renderTask = null;

		this.outputTarget = new StorageTexture( 1, 1, );
		this.outputTarget.format = RGBAFormat;
		this.outputTarget.type = FloatType;
		this.outputTarget.colorSpace = ColorManagement.workingColorSpace;
		this.outputTarget.name = 'Output #0';
		this.outputTarget.generateMipmaps = false;

		this.prevOutputTarget = new StorageTexture( 1, 1, );
		this.prevOutputTarget.format = RGBAFormat;
		this.prevOutputTarget.type = FloatType;
		this.prevOutputTarget.colorSpace = ColorManagement.workingColorSpace;
		this.prevOutputTarget.name = 'Output #1';
		this.prevOutputTarget.generateMipmaps = false;

		this.sampleCountTarget = new StorageTexture( 1, 1, );
		this.sampleCountTarget.format = RedIntegerFormat;
		this.sampleCountTarget.type = UnsignedIntType;
		this.sampleCountTarget.name = 'Sample Count';
		this.sampleCountTarget.generateMipmaps = false;

		this.sampleCountClearKernel = new ZeroOutKernel().setWorkgroupSize( 8, 8, 1 );
		this.outputTargetClearKernel = new ZeroOutKernel().setWorkgroupSize( 8, 8, 1 );

	}

	setRandom( random ) {

	}

	setBVHData( data ) {

	}

	/**
	 * Lega un manto di peli al tracciatore.
	 *
	 * Un pacchetto solo per tutta la scena: il kernel lega un buffer, e ogni
	 * segmento dice a quale oggetto appartiene.
	 *
	 * @param {Uint32Array|null} packed - il pacchetto delle curve, o null per toglierlo.
	 */
	setFur( packed ) {

		const data = packed === null ? EMPTY_HAIR_DATA : packed;
		const previous = this.hairAttribute;

		// ── THE SAME SIZE MEANS THE SAME BUFFER ──
		//
		// A new attribute on every re-bake means a new GPU buffer AND a new host
		// array, and the old pair survives one generation: measured on a lawn of
		// 250k strands, the JS heap settles at TWO packets (+91 MB each) instead
		// of one. Writing in place keeps a single buffer on both sides, and the
		// coat is re-baked on every scene rebuild, so the size rarely changes.
		if ( previous && previous.array.length === data.length ) {

			previous.array.set( data );
			previous.needsUpdate = true;
			return;

		}

		this.hairAttribute = new StorageBufferAttribute( data, 1 );
		// a different size cannot reuse the buffer, so the old one is released
		// here: nothing else owns it, and a dropped reference is not a freed
		// GPU allocation.
		previous?.dispose?.();

	}

	setTextures( textures ) {

	}

	setMaterial( material ) {

	}

	setFilterGlossy( value ) {

	}

	setClamping( _direct, _indirect ) {

	}

	setEnvironment(
		envMap,
		envMapIntensity,
		envMapRotation,
	) {

	}

	setLights( lights ) {

	}

	setBackground(
		background,
		backgroundIntensity,
		backgroundRotation,
		backgroundBlurriness,
	) {

	}

	// Rebuild the render task after a shader-generating function changes.
	rebuild() {

		this._renderTask = null;

	}

	setSize( w, h ) {

		w = Math.ceil( w );
		h = Math.ceil( h );

		const { width, height } = this.outputTarget;
		if ( width === w && height === h ) {

			return false;

		}

		this.outputTarget.dispose();
		this.prevOutputTarget.dispose();
		this.sampleCountTarget.dispose();

		this.outputTarget = this.outputTarget.clone();
		this.prevOutputTarget = this.outputTarget.clone();
		this.sampleCountTarget = this.sampleCountTarget.clone();

		this.outputTarget.setSize( w, h );
		this.prevOutputTarget.setSize( w, h );
		this.sampleCountTarget.setSize( w, h );

		this.reset();
		return true;

	}

	getSize( target ) {

		const { width, height } = this.outputTarget;
		target.x = width;
		target.y = height;

		return target;

	}

	update() {

		const { renderer } = this;
		if ( ! renderer.initialized ) {

			return;

		}

		if ( ! this._renderTask ) {

			this._renderTask = this.createRenderTask();

		}

		this._renderTask.next();

	}

	*createRenderTask() {

	}

	// Measures the current per pixel sample counts. Backends that need GPU work to answer this do
	// it here, so it only happens when asked for rather than every round.
	async getSampleCountsAsync() {

		return { min: 0, max: 0, avg: 0 };

	}

	resetSeed() {}

	reset() {

		const {
			renderer,
			outputTargetClearKernel,
			sampleCountClearKernel,

			outputTarget,
			prevOutputTarget,
			sampleCountTarget,
		} = this;

		if ( ! renderer.initialized ) {

			return;

		}

		const { width, height } = outputTarget;
		const dispatchSize = outputTargetClearKernel.getDispatchSize( width, height );

		outputTargetClearKernel.target = outputTarget;
		renderer.compute( outputTargetClearKernel.kernel, dispatchSize );

		outputTargetClearKernel.target = prevOutputTarget;
		renderer.compute( outputTargetClearKernel.kernel, dispatchSize );

		sampleCountClearKernel.target = sampleCountTarget;
		renderer.compute( sampleCountClearKernel.kernel, dispatchSize );

		this._renderTask = null;

	}

	dispose() {

		this.outputTarget.dispose();
		this.prevOutputTarget.dispose();
		this.sampleCountTarget.dispose();

		this._renderTask = null;

	}

}
