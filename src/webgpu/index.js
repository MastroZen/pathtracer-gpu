export * from './WebGPUPathTracer.js';
export * from './BlurredEnvMapGenerator.js';
export * from './constants.js';
export * from './materials/RenderToScreenMaterial.js';
export * from './denoise/OIDNDenoiser.js';
export * from './upscale/FSRUpscaler.js';

// the camera with depth of field, from the same module the shim below extends: importing it
// through another entry point could hand out a second copy of the class, without the shim
export { PhysicalCamera } from '../objects/PhysicalCamera.js';

// extend the cameras to avoid adding WebGPU imports to the WebGLPathTracer
import './shims/EquirectCameraShim.js';
import './shims/PhysicalCameraShim.js';
import './shims/ArrayCameraShim.js';
