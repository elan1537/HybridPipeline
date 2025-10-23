"""Depth normalization utilities."""

import torch
import math


def depth_to_log_uint8(depth: torch.Tensor, alpha: torch.Tensor,
                       near: float, far: float,
                       alpha_cutoff: float = 0.5,
                       scale: float = 0.80823) -> torch.Tensor:
    """
    Convert depth to log-normalized uint8.

    From backend-feedforward-20251021/server.py depth_to_8bit_one_channel().
    """
    d = depth.clone()
    d[alpha < alpha_cutoff] = far

    z = d.clamp(min=near, max=far)
    log_ratio = torch.log(z / near)
    log_den = math.log(far / near)

    d_norm = (log_ratio / log_den) * scale
    d_uint8 = (d_norm * 255.0).clamp(0, 255).to(torch.uint8)

    return d_uint8


def depth_to_log_float16(depth: torch.Tensor, alpha: torch.Tensor,
                         near: float, far: float) -> torch.Tensor:
    """
    Convert depth to log-normalized float16 [0, 1].

    For JPEG depth encoding (backend-feedforward-20251021/server.py).
    """
    d_uint8 = depth_to_log_uint8(depth, alpha, near, far)
    d_float = d_uint8.float() / 255.0
    d_f16 = d_float.to(torch.float16)
    return d_f16


def depth_to_ndc(depth: torch.Tensor, alpha: torch.Tensor,
                 near: float, far: float,
                 alpha_cutoff: float = 0.5) -> torch.Tensor:
    """
    Convert depth to WebGL NDC range [-1, 1].

    From legacy/server.py NDC transformation.
    """
    d = depth.clone()
    d[alpha < alpha_cutoff] = torch.nan

    term_A = (far + near) / (far - near)
    term_B_num = 2 * far * near
    denominator = (far - near) * d

    ndc = term_A - (term_B_num / denominator)
    ndc = torch.clamp(ndc, -1.0, 1.0)

    return ndc


def depth_to_rgb(depth: torch.Tensor, alpha: torch.Tensor,
                 near: float, far: float) -> torch.Tensor:
    """
    Convert depth to RGB uint8 for visualization.

    Returns (H, W, 3) uint8.
    """
    if depth.ndim == 3:
        depth = depth.squeeze(0)

    d_uint8 = depth_to_log_uint8(depth, alpha, near, far)
    d_rgb = d_uint8.unsqueeze(-1).expand(-1, -1, 3).contiguous()

    return d_rgb
