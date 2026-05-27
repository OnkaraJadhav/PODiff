from __future__ import annotations
from dataclasses import dataclass
import torch
from .schedule import cosine_beta_schedule, linear_beta_schedule

@dataclass
class DDPMConstants:
    betas: torch.Tensor
    alphas: torch.Tensor
    abar: torch.Tensor
    abar_prev: torch.Tensor
    sqrt_abar: torch.Tensor
    sqrt_1mabar: torch.Tensor
    post_var: torch.Tensor

def make_ddpm_constants(timesteps: int, schedule: str = "cosine", device: torch.device | None = None) -> DDPMConstants:
    if schedule == "cosine":
        betas = cosine_beta_schedule(timesteps)
    elif schedule == "linear":
        betas = linear_beta_schedule(timesteps)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")

    if device is not None:
        betas = betas.to(device)

    alphas = 1.0 - betas
    abar = torch.cumprod(alphas, dim=0)
    abar_prev = torch.cat([torch.tensor([1.0], device=betas.device), abar[:-1]], dim=0)
    sqrt_abar = torch.sqrt(abar)
    sqrt_1mabar = torch.sqrt(1.0 - abar)
    post_var = betas * (1.0 - abar_prev) / (1.0 - abar)
    return DDPMConstants(betas, alphas, abar, abar_prev, sqrt_abar, sqrt_1mabar, post_var)

def q_sample(x0: torch.Tensor, t: torch.Tensor, const: DDPMConstants, noise: torch.Tensor | None = None) -> torch.Tensor:
    if noise is None:
        noise = torch.randn_like(x0)
    return const.sqrt_abar[t].view(-1,1,1) * x0 + const.sqrt_1mabar[t].view(-1,1,1) * noise

@torch.no_grad()
def p_sample(model, xt: torch.Tensor, t: torch.Tensor, const: DDPMConstants, cond) -> torch.Tensor:
    betas_t = const.betas[t].view(-1,1,1)
    sqrt_1m = const.sqrt_1mabar[t].view(-1,1,1)
    sqrt_recip_alpha = torch.sqrt(1.0 / const.alphas[t]).view(-1,1,1)

    eps = model(xt, t, cond)
    mean = sqrt_recip_alpha * (xt - betas_t * eps / sqrt_1m)

    noise = torch.randn_like(xt)
    var = const.post_var[t].view(-1,1,1)
    mask = (t != 0).float().view(-1,1,1)
    return mean + mask * torch.sqrt(var) * noise

@torch.no_grad()
def p_sample_loop(model, shape, const: DDPMConstants, cond, device: torch.device) -> torch.Tensor:
    xt = torch.randn(shape, device=device)
    for i in reversed(range(const.betas.shape[0])):
        t = torch.full((shape[0],), i, device=device, dtype=torch.long)
        xt = p_sample(model, xt, t, const, cond)
    return xt
