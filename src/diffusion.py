"""Diffusion process utils."""

from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """Linear beta schedule.

    Args:
        timesteps: Number of timesteps.

    Returns:
        Beta coefficients with linear schedule.
    """
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(a, t, x_shape):
    """Gathers array elements for timesteps t.

    Args:
        a: Input array.
        t: Timesteps.
        x_shape: Output shape.

    Returns:
        Array of elements at timesteps t reshaped as x_shape.
    """
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())

    # [batch_size] -> [batch_size, h, w, c]
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


class GaussianDiffusion:
    """Class to model Gaussian Diffusion process.

    Args:
        noise_schedule: Function that returns beta coefficients.
        timesteps: How many diffusion steps to do.
    """

    def __init__(self, noise_schedule: Callable, timesteps: int) -> None:
        """Inits Gaussian diffusion."""
        self.betas = noise_schedule(timesteps=timesteps)
        self.timesteps = timesteps

        self.alphas = 1.0 - self.betas

        # alpha_bar = [alpha_bar_0, alpha_bar_1, ..., alpha_bar_T]
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

        # alpha_bar_t-1 = [1.0, alpha_bar_0, ..., alpha_bar_T-1]
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # sqrt(alpha_bar) and sqrt(1 - alpha_bar)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # posterior variance aka tilde_beta_t
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None):
        """Forward deiffusion process sample.

        Args:
            x0: Initial image: [b, c, h w].
            t: Timesteps batch: [b, 1].
            eps: Sampled noise. Defaults to None.

        Returns:
            Noised images x_t [b, c, h w].
        """
        if eps is None:
            eps = torch.randn_like(x0)

        # Get sqrt(alpha_bar_t) and sqrt(1 - alpha_bar_t) for every timestep t.
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)

        return sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * eps

    @torch.no_grad()
    def p_sample(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, t_index: int) -> torch.Tensor:
        """Reverse diffusion process sample.

        Args:
            model: Model to generate eps_theta(x_t, t).
            x: x_t: [b, c, h, w].
            t: Timestep.
            t_index: Timestep index.

        Returns:
            x_t-1: [b, c, h, w].
        """
        betas_t = extract(self.betas, t, x.shape)

        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # mu_theta(x_t, t)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

        if t_index == 0:
            return model_mean
        else:
            # posterior variance aka tilde_beta_t
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, shape: Tuple[int, int, int, int]) -> torch.Tensor:
        """Reverse diffusion process sample from x_t to x_0.

        Args:
            model: Model to generate eps_theta(x_t, t).
            shape: Output shape.

        Returns:
            x_0 sample.
        """
        device = next(model.parameters()).device

        b = shape[0]

        # Start from pure noise (for each example in the batch)
        img = torch.randn(shape, device=device)
        imgs = []

        for i in tqdm(
            reversed(range(0, self.timesteps)),
            desc="sampling loop time step",
            total=self.timesteps,
        ):
            img = self.p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
            imgs.append(img.cpu())

        return imgs

    def sample(
        self,
        model: nn.Module,
        image_size: int,
        batch_size: int = 16,
        channels: int = 3,
    ):
        """Reverse diffusion process sample from x_t to x_0.

        Args:
            model: Model to generate eps_theta(x_t, t).
            image_size: Output image size.
            batch_size: How many images to generate. Defaults to 16.
            channels: Image num channels. Defaults to 3.

        Returns:
            x_0 samples [b, c, h, w].
        """
        return self.p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))
