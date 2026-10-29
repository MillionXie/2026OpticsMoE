"""Explicit higher-order sampler ablation; no weight or architecture change.

https://huggingface.co/docs/diffusers/api/schedulers/multistep_dpm_solver
This is inference solver selection, not learned tail distillation or an optical
speed result. Both electronic and LG controls must use the same setting.
"""


def descriptor(name):
    if name=='original':return None
    if name not in ['dpmpp2m-karras','dpmpp2m-karras-teacher30']:raise ValueError('Unsupported sampler')
    result=dict(kind='DPMpp_2M_Karras_v1',scheduler='DPMSolverMultistepScheduler',algorithm_type='dpmsolver++',
        solver_order=2,solver_type='midpoint',use_karras_sigmas=True,thresholding=False,lower_order_final=True,
        final_sigmas_type='zero',timestep_spacing='linspace',steps_offset=0,CFG=7.5,weights_unchanged=True,
        not_an_optical_only_speed_gain=True)
    if name.endswith('-teacher30'):
        result.update(kind='DPMpp_2M_Karras_teacher30_bounds_v1',
            noise_bounds='native_Euler30_first_and_last_nonzero_sigma_from_same_trained_betas',
            reference_steps=30,unit_noise_preserved=True)
    return result


def install(pipe,name):
    choice=descriptor(name)
    if choice is None:return None
    from diffusers import DPMSolverMultistepScheduler
    old=pipe.scheduler.config
    if old.prediction_type!='epsilon':raise ValueError('This candidate is bound to the existing epsilon-trained UNet')
    overrides={k:choice[k] for k in ['algorithm_type','solver_order','solver_type','use_karras_sigmas','thresholding',
        'lower_order_final','final_sigmas_type','timestep_spacing','steps_offset']}
    pipe.scheduler=DPMSolverMultistepScheduler.from_config(old,**overrides)
    if name.endswith('-teacher30'):
        from diffusers import EulerDiscreteScheduler
        teacher=EulerDiscreteScheduler.from_config(old);teacher.set_timesteps(30)
        # The installed scheduler's Karras converter reads these explicit config
        # bounds. Clipping lambda alone does NOT limit its Karras sigma range.
        # Original Euler30 starts at t958 for this asset, while default Karras
        # starts at t999. This ablation matches the accepted teacher's bounds.
        pipe.scheduler.register_to_config(sigma_max=float(teacher.sigmas[0]),sigma_min=float(teacher.sigmas[-2]))
    for key in ['num_train_timesteps','beta_start','beta_end','beta_schedule','prediction_type']:
        assert pipe.scheduler.config[key]==old[key],'Changed trained noise protocol'
    return choice
