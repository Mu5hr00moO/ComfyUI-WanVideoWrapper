import torch
from .fm_solvers import (FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps)
from .fm_solvers_unipc import FlowUniPCMultistepScheduler
from .basic_flowmatch import FlowMatchScheduler
from .flowmatch_pusa import FlowMatchSchedulerPusa
from .flowmatch_res_multistep import FlowMatchSchedulerResMultistep
from .scheduling_flow_match_lcm import FlowMatchLCMScheduler
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, DEISMultistepScheduler
from .fm_solvers_euler import EulerScheduler

import numpy as np
from ...utils import log

scheduler_list = [
    "unipc", "unipc/beta",
    "dpm++", "dpm++/beta",
    "dpm++_sde", "dpm++_sde/beta",
    "euler", "euler/beta",
    "deis",
    "lcm", "lcm/beta",
    "res_multistep",
    "flowmatch_causvid",
    "flowmatch_distill",
    "flowmatch_pusa",
    "lightning_euler", "lightning_euler/beta", "lightning_euler/beta57",
    "multitalk"
]

def get_scheduler(scheduler, steps, shift, device, transformer_dim, flowedit_args, denoise_strength, sigmas=None):
    timesteps = None
    if 'unipc' in scheduler:
        sample_scheduler = FlowUniPCMultistepScheduler(shift=shift)
        if sigmas is None:
            sample_scheduler.set_timesteps(steps, device=device, shift=shift, use_beta_sigmas=('beta' in scheduler))
        else:
            sample_scheduler.sigmas = sigmas.to(device)
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64).to(device)
            sample_scheduler.num_inference_steps = len(sample_scheduler.timesteps)

    elif scheduler in ['euler/beta', 'euler']:
        sample_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift, use_beta_sigmas=(scheduler == 'euler/beta'))
        if flowedit_args: #seems to work better
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=get_sampling_sigmas(steps, shift))
        else:
            sample_scheduler.set_timesteps(steps, device=device, sigmas=sigmas[:-1].tolist() if sigmas is not None else None)
    elif 'dpm' in scheduler:
        if 'sde' in scheduler:
            algorithm_type = "sde-dpmsolver++"
        else:
            algorithm_type = "dpmsolver++"
        sample_scheduler = FlowDPMSolverMultistepScheduler(shift=shift, algorithm_type=algorithm_type)
        if sigmas is None:
            sample_scheduler.set_timesteps(steps, device=device, use_beta_sigmas=('beta' in scheduler))
        else:
            sample_scheduler.sigmas = sigmas.to(device)
            sample_scheduler.timesteps = (sample_scheduler.sigmas[:-1] * 1000).to(torch.int64).to(device)
            sample_scheduler.num_inference_steps = len(sample_scheduler.timesteps)
    elif scheduler == 'deis':
        sample_scheduler = DEISMultistepScheduler(use_flow_sigmas=True, prediction_type="flow_prediction", flow_shift=shift)
        sample_scheduler.set_timesteps(steps, device=device)
        sample_scheduler.sigmas[-1] = 1e-6
    elif 'lcm' in scheduler:
        sample_scheduler = FlowMatchLCMScheduler(shift=shift, use_beta_sigmas=(scheduler == 'lcm/beta'))
        sample_scheduler.set_timesteps(steps, device=device, sigmas=sigmas[:-1].tolist() if sigmas is not None else None)
    elif 'flowmatch_causvid' in scheduler:
        if sigmas is not None:
            raise NotImplementedError("This scheduler does not support custom sigmas")
        if transformer_dim == 5120:
            denoising_list = [999, 934, 862, 756, 603, 410, 250, 140, 74]
        else:
            if steps != 4:
                raise ValueError("CausVid 1.3B schedule is only for 4 steps")
            denoising_list = [1000, 750, 500, 250]
        sample_scheduler = FlowMatchScheduler(num_inference_steps=steps, shift=shift, sigma_min=0, extra_one_step=True)
        sample_scheduler.timesteps = torch.tensor(denoising_list)[:steps].to(device)
        sample_scheduler.sigmas = torch.cat([sample_scheduler.timesteps / 1000, torch.tensor([0.0], device=device)])
    elif 'flowmatch_distill' in scheduler:
        if sigmas is not None:
            raise NotImplementedError("This scheduler does not support custom sigmas")
        sample_scheduler = FlowMatchScheduler(
            shift=shift, sigma_min=0.0, extra_one_step=True
        )
        sample_scheduler.set_timesteps(1000, training=True)
    
        denoising_step_list = torch.tensor([999, 750, 500, 250] , dtype=torch.long)
        temp_timesteps = torch.cat((sample_scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
        denoising_step_list = temp_timesteps[1000 - denoising_step_list]
        #print("denoising_step_list: ", denoising_step_list)
        
        if steps != 4:
            raise ValueError("This scheduler is only for 4 steps")
        
        sample_scheduler.timesteps = denoising_step_list[:steps].clone().detach().to(device)
        sample_scheduler.sigmas = torch.cat([sample_scheduler.timesteps / 1000, torch.tensor([0.0], device=device)])
    elif 'flowmatch_pusa' in scheduler:
        sample_scheduler = FlowMatchSchedulerPusa(
            shift=shift, sigma_min=0.0, extra_one_step=True
        )
        sample_scheduler.set_timesteps(steps, denoising_strength=denoise_strength, shift=shift, sigmas=sigmas[:-1].tolist() if sigmas is not None else None)
    elif 'lightning_euler' in scheduler:
        if sigmas is not None:
            raise NotImplementedError("This scheduler does not support custom sigmas")

        # --- resolve Beta shaping + (alpha, beta) for 'beta57' ---
        use_beta = (scheduler == 'lightning_euler/beta') or (scheduler == 'lightning_euler/beta57')
        alpha_val, beta_val = (0.6, 0.6)
        if scheduler == 'lightning_euler/beta57':
            alpha_val, beta_val = (0.5, 0.7)

        sample_scheduler = EulerScheduler(
            num_train_timesteps=1000,
            shift=shift,
            device=device,
            use_beta_sigmas=use_beta,
            # --- pass alpha/beta (no-op for plain lighting_euler) ---
            alpha=alpha_val,
            beta=beta_val,
        )

        sample_scheduler.set_timesteps(num_inference_steps=steps, device=device)
        timesteps = sample_scheduler.timesteps[:-1].clone()
    elif scheduler == 'res_multistep':
        sample_scheduler = FlowMatchSchedulerResMultistep(shift=shift)
        sample_scheduler.set_timesteps(steps, denoising_strength=denoise_strength, sigmas=sigmas[:-1].tolist() if sigmas is not None else None)
    if timesteps is None:
        timesteps = sample_scheduler.timesteps
    return sample_scheduler, timesteps
