"""Color space conversion utilities."""

import torch


def rgb_to_nv12(rgb: torch.Tensor) -> torch.Tensor:
    """
    Convert RGB to NV12 format (BT.601).

    From backend-feedforward-20251021/server.py convert_rgb_to_nv12().

    Args:
        rgb: (H, W, 3) uint8

    Returns:
        nv12: (H + H/2, W) uint8
    """
    H, W = rgb.shape[:2]

    r = rgb[..., 0].float()
    g = rgb[..., 1].float()
    b = rgb[..., 2].float()

    # BT.601 conversion
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128

    y = torch.clamp(y, 0, 255).to(torch.uint8)
    u = torch.clamp(u, 0, 255).to(torch.uint8)
    v = torch.clamp(v, 0, 255).to(torch.uint8)

    y_plane = y.contiguous()

    # 4:2:0 subsampling
    u_sub = u.view(H // 2, 2, W // 2, 2).float().mean(dim=(1, 3)).to(torch.uint8)
    v_sub = v.view(H // 2, 2, W // 2, 2).float().mean(dim=(1, 3)).to(torch.uint8)

    # Interleave UV
    uv_plane = torch.zeros(H // 2, W, dtype=torch.uint8, device=rgb.device)
    uv_plane[:, 0::2] = u_sub
    uv_plane[:, 1::2] = v_sub

    nv12 = torch.cat([y_plane, uv_plane], dim=0)
    return nv12
