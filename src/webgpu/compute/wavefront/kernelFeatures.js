// THE KERNEL FEATURES, as Cycles has them (kernel/types.h: KERNEL_FEATURE_HAIR, _VOLUME, _SUBSURFACE):
// what the scene can ask of MaterialKernel and LogicKernel. The code of a feature the scene does not have is not
// generated at all. Its branches were never taken anyway - a hit is a curve only with a coat bound,
// a path is inside a medium only when a material has one - so the image is the same; what changes is
// the compile time, which FXC (the HLSL compiler under D3D) pays for every line it inlines.
//
// "hair" is a coat bound to the trace kernels, "huang" a material that asks for the Huang model,
// "medium" a material with a participating medium, "subsurface" one with a subsurface weight.
export const NO_KERNEL_FEATURES = Object.freeze( { hair: false, huang: false, medium: false, subsurface: false } );

export function kernelFeaturesKey( features ) {

	return [ features.hair, features.hair && features.huang, features.medium, features.subsurface ].map( v => ( v ? '1' : '0' ) ).join( '' );

}
