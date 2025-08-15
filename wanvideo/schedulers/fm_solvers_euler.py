#Based on https://github.com/ModelTC/Wan2.2-Lightning/tree/main/wan/utils
import numpy as np
import torch
# pyright: ignore
from diffusers import FlowMatchEulerDiscreteScheduler  # pyright: ignore
from torch import Tensor
from diffusers.utils import is_scipy_available
import scipy.stats  # Added import for scipy.stats.beta

def unsqueeze_to_ndim(in_tensor: Tensor, tgt_n_dim: int):
    if in_tensor.ndim > tgt_n_dim:
        return in_tensor
    if in_tensor.ndim < tgt_n_dim:
        in_tensor = in_tensor[(...,) + (None,) * (tgt_n_dim - in_tensor.ndim)]
    return in_tensor

def get_timesteps(num_steps, max_steps: int = 1000):
    return np.linspace(max_steps, 0, num_steps + 1, dtype=np.float32)

def timestep_shift(timesteps, shift: float = 1.0):
    return shift * timesteps / (1 + (shift - 1) * timesteps)

class EulerScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(
        self,
        num_train_timesteps: int,
        shift: float = 1.0,
        use_beta_sigmas: bool = False,
        device: torch.device | str = "cuda",
        alpha: float = 0.6,
        beta: float = 0.6,
        **kwargs
    ) -> None:
        if use_beta_sigmas and not is_scipy_available():
            raise ImportError("Make sure to install scipy if you want to use beta sigmas.")
        super().__init__(num_train_timesteps=num_train_timesteps, shift=shift, **kwargs)
        self.init_noise_sigma = 1.0
        self.num_train_timesteps = num_train_timesteps
        self._shift = shift
        self.use_beta_sigmas = use_beta_sigmas
        self.device = device

        # --- persist params for rebuilds (e.g., set_shift) ---
        self.num_inference_steps: int | None = None
        self.beta_alpha: float = alpha
        self.beta_beta: float = beta

        self.set_timesteps(num_inference_steps=num_train_timesteps)
        pass

    def set_shift(self, shift: float = 1.0):
        # --- recompute sigmas preserving Beta shaping if enabled ---
        self._shift = shift

        # Base timesteps grid (N+1) from original range
        timesteps = self.timesteps_ori if hasattr(self, "timesteps_ori") else torch.from_numpy(
            get_timesteps(num_steps=self.num_inference_steps or self.num_train_timesteps,
                          max_steps=self.num_train_timesteps)
        ).to(dtype=torch.float32, device=self.device)

        # Linear sigmas on [0,1] with shift
        sigmas = (timesteps / self.num_train_timesteps).to(dtype=torch.float32)
        sigmas = timestep_shift(sigmas, shift=self._shift)

        if self.use_beta_sigmas:
            # Exclude terminal zero for Beta shaping; rebuild with stored alpha/beta and step count
            in_sigmas = sigmas[:-1].detach().cpu().numpy()
            steps = self.num_inference_steps if self.num_inference_steps is not None else (len(timesteps) - 1)
            sigmas_np = self._convert_to_beta(in_sigmas, steps, alpha=self.beta_alpha, beta=self.beta_beta)
            sigmas_t = torch.from_numpy(sigmas_np).to(dtype=torch.float32, device=self.device)
            # Append terminal zero for step() reading (i, i+1)
            self.sigmas = torch.cat([sigmas_t, torch.zeros(1, dtype=sigmas_t.dtype, device=sigmas_t.device)])
        else:
            self.sigmas = sigmas.to(dtype=torch.float32, device=self.device)

        # Update timesteps to match sigmas (N+1 elements)
        self.timesteps = self.sigmas * self.num_train_timesteps

        self._step_index = None
        self._begin_index = None

    def set_timesteps(
        self, num_inference_steps: int, device: torch.device | str | int | None = None
    ):
        # --- remember count for later re-builds ---
        self.num_inference_steps = int(num_inference_steps)

        timesteps = get_timesteps(
            num_steps=num_inference_steps, max_steps=self.num_train_timesteps
        )
        self.timesteps = torch.from_numpy(timesteps).to(
            dtype=torch.float32, device=device or self.device
        )
        self.timesteps_ori = self.timesteps.clone()

        # Calculate sigmas based on timesteps
        sigmas = self.timesteps_ori / self.num_train_timesteps

        # Apply shift to sigmas
        sigmas = timestep_shift(sigmas, shift=self._shift)

        # If use_beta_sigmas=True, apply direct Beta conversion (continuous, numerically safe)
        if self.use_beta_sigmas:
            in_sigmas = sigmas[:-1]  # Exclude the terminal zero for beta conversion
            sigmas_np = self._convert_to_beta(in_sigmas.cpu().numpy(), num_inference_steps,
                                              alpha=self.beta_alpha, beta=self.beta_beta)
            self.sigmas = torch.from_numpy(sigmas_np).to(
                dtype=torch.float32, device=device or self.device
            )
            # Append the terminal zero
            self.sigmas = torch.cat([self.sigmas, torch.zeros(1, dtype=self.sigmas.dtype, device=self.sigmas.device)])
        else:
            self.sigmas = sigmas.to(
                dtype=torch.float32, device=device or self.device
            )

        # Update timesteps based on new sigmas
        self.timesteps = self.sigmas * self.num_train_timesteps

        self._step_index = None
        self._begin_index = None

    def _convert_to_beta(
        self, in_sigmas: np.ndarray, num_inference_steps: int, alpha: float = 0.6, beta: float = 0.6
    ) -> np.ndarray:
        """Convert sigmas to a continuous Beta-shaped schedule (Direct Beta), numerically safe.
        Based on 'Beta Sampling is All You Need' [arXiv:2407.12173] (Lee et. al, 2024)."""
        # Use config values if available, otherwise fall back to in_sigmas
        sigma_min = getattr(self.config, "sigma_min", in_sigmas[-1]) if hasattr(self, "config") else in_sigmas[-1]
        sigma_max = getattr(self.config, "sigma_max", in_sigmas[0]) if hasattr(self, "config") else in_sigmas[0]
        assert sigma_max >= sigma_min, "expected sigma_max >= sigma_min"
        assert num_inference_steps >= 1, "num_inference_steps must be >= 1"
        assert alpha > 0.0 and beta > 0.0, "alpha and beta must be > 0"

        eps = 1e-6  # avoid exact endpoints {0,1} to prevent NaN/Inf in some SciPy builds
        t = np.linspace(eps, 1.0 - eps, num_inference_steps, endpoint=True)  # length N
        u = 1.0 - t
        p = scipy.stats.beta.ppf(u, alpha, beta)  # p in (0,1)

        sigmas = sigma_min + p * (sigma_max - sigma_min)
        sigmas = sigmas.astype(np.float32)
        return sigmas

    def step(
        self,
        model_output: torch.FloatTensor,
        timestep: float | torch.FloatTensor,
        sample: torch.FloatTensor,
        **kwargs,
    ) -> tuple:
        if (
            isinstance(timestep, int)
            or isinstance(timestep, torch.IntTensor)
            or isinstance(timestep, torch.LongTensor)
        ):
            raise ValueError(
                (
                    "Passing integer indices (e.g. from `enumerate(timesteps)`) as timesteps to"
                    " `EulerDiscreteScheduler.step()` is not supported. Make sure to pass"
                    " one of the `scheduler.timesteps` as a timestep."
                ),
            )

        if self.step_index is None:
            self._init_step_index(timestep)
        sample = sample.to(torch.float32)  # pyright: ignore
        sigma = unsqueeze_to_ndim(self.sigmas[self.step_index], sample.ndim).to(sample.device)
        sigma_next = unsqueeze_to_ndim(self.sigmas[self.step_index + 1], sample.ndim).to(sample.device)
        x_t_next = sample + (sigma_next - sigma) * model_output
        self._step_index += 1
        return x_t_next
