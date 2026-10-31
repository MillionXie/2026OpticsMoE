"""Explicit SDXL size-conditioning DEV ablation, separate from pixel resolution.

Historical defaults come from the checkpoint. An override is development only;
it is not a claim that a mismatch caused an observed semantic failure.
"""


def descriptor(state,resolution,override=None,scope='dev',expanded=False,timing=False):
    trained=int(state.get('SDXL_conditioning_size',resolution))
    if override is not None:
        if override not in[512,1024]or scope!='dev'or not expanded or timing:
            raise ValueError('Size-conditioning probe requires explicit expanded DEV, no TEST/timing')
    actual=trained if override is None else override
    return dict(training_conditioning_side=trained,inference_conditioning_side=actual,
                output_pixel_side=resolution,explicit_DEV_override=override is not None,
                size_conditioning_is_not_pixel_resolution=True)
