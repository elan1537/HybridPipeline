# ws_handler.py
import struct
import torch
import torch.nn.functional as F
import kornia

def parse_payload(raw: bytes, device="cuda"):
    vals = struct.unpack("<15f", raw) 
    
    eye = torch.tensor([[vals[0], -vals[1], vals[2]]], dtype=torch.float32, device=device)
    target = torch.tensor([[vals[3], -vals[4], vals[5]]], dtype=torch.float32, device=device)
    up = torch.tensor([[0., 1., 0]], dtype=torch.float32, device=device)

    zaxis = F.normalize(target - eye, dim=-1)
    xaxis = F.normalize(torch.cross(up, zaxis, dim=-1), dim=-1)
    yaxis = torch.cross(zaxis, xaxis, dim=-1)

    R_w2c = torch.stack([xaxis.squeeze(0), yaxis.squeeze(0), zaxis.squeeze(0)], dim=0).unsqueeze(0)

    t = -R_w2c @ eye.unsqueeze(-1)
    view = kornia.geometry.conversions.Rt_to_matrix4x4(R_w2c, t)

    intrinsics_vals = vals[6:]
    intrinsics = torch.tensor([[
        [intrinsics_vals[0], intrinsics_vals[1], intrinsics_vals[2]],
        [intrinsics_vals[3], intrinsics_vals[4], intrinsics_vals[5]],
        [intrinsics_vals[6], intrinsics_vals[7], intrinsics_vals[8]] 
    ]], device=device)
    
    # print(intrinsics)
    # Return the transformed matrices
    return view, intrinsics
