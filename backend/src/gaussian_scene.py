#gaussian_scene.py
from gsplat import rasterization, rasterization_2dgs
from gaussian_renderer import GaussianModel, render
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, ModelHiddenParams
import json
from utils.general_utils import safe_state
from scene import Scene
from scene.cameras import MiniCam, Camera
from plyfile import PlyData, PlyElement
import torch
import numpy as np
import math


class GaussianScene:
    def __init__(self, model_path, apply_x_flip=False):
        data = PlyData.read(model_path)
        self.vertex_data = data["vertex"]

        means, quats, scales, ops, colors, shs = self.load_from_plydata()

        if apply_x_flip:
            print("Applying X-axis flip to Gaussian data...")
            means[:, 0] *= -1
            print(f" Flipped means X coordinate.")

            quats[:, 2] *= -1  # Flip qy
            quats[:, 3] *= -1  # Flip qz

            norms = np.linalg.norm(quats, axis=1, keepdims=True)

            zero_norms_mask = norms < 1e-8
            norms[zero_norms_mask] = 1.0 # 0/0 방지 (0 쿼터니언은 그대로 둠)
            quats /= norms
            print(f" Flipped quaternion y and z components and renormalized.")

        self.means = means
        self.quats = quats
        self.scales = scales
        self.ops = ops
        self.shs = shs
        self.colors = colors

        # GPU 업로드 상태 플래그
        self.on_gpu = False

        print("Init okay")
        print(f"Loaded {self.means.shape[0]} Gaussians.")


    def load_from_plydata(self):
        vertex_data = self.vertex_data

        positions = np.vstack(
            [vertex_data["x"], vertex_data["y"], vertex_data["z"]]
        ).T.astype(np.float32)

        rotations = np.vstack(
            [
                vertex_data["rot_0"],
                vertex_data["rot_1"],
                vertex_data["rot_2"],
                vertex_data["rot_3"],
            ]
        ).T.astype(np.float32)

        log_scales = np.vstack(
            [vertex_data["scale_0"], vertex_data["scale_1"], vertex_data["scale_2"]]
        ).T.astype(np.float32)

        opacities = np.array([vertex_data["opacity"]]).T.astype(np.float32)

        colors = np.array(
            [vertex_data["f_dc_0"], vertex_data["f_dc_1"], vertex_data["f_dc_2"]]
        ).T.astype(np.float32)
        shs = np.expand_dims(colors.copy(), axis=1)
        
        for i in range(15):
            r_channel = f"f_rest_{i}"
            g_channel = f"f_rest_{15 + i}"
            b_channel = f"f_rest_{30 + i}"

            col = np.vstack(
                [
                    vertex_data[r_channel],
                    vertex_data[g_channel],
                    vertex_data[b_channel],
                ]
            ).T
            col = np.expand_dims(col, axis=1)
            shs = np.concatenate([shs, col], axis=1)

        return [positions, rotations, log_scales, opacities, colors, shs]

    def save_scene_to_ply(self, output_path):
        # vertex 데이터 준비
        dtype = [
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),
            ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
            ('opacity', 'f4'),
            ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4')
        ]
        
        # 고차원 색상 계수 필드 추가
        for i in range(15):
            dtype.extend([
                (f'f_rest_{i}', 'f4'),
                (f'f_rest_{15 + i}', 'f4'),
                (f'f_rest_{30 + i}', 'f4')
            ])
            
        vertex_data = np.zeros(self.means.shape[0], dtype=dtype)

        # 기본 속성 할당
        vertex_data['x'] = self.means[:, 0]
        vertex_data['y'] = self.means[:, 1]
        vertex_data['z'] = self.means[:, 2]
        vertex_data['rot_0'] = self.quats[:, 0]
        vertex_data['rot_1'] = self.quats[:, 1]
        vertex_data['rot_2'] = self.quats[:, 2]
        vertex_data['rot_3'] = self.quats[:, 3]
        vertex_data['scale_0'] = self.scales[:, 0]
        vertex_data['scale_1'] = self.scales[:, 1]
        vertex_data['scale_2'] = self.scales[:, 2]
        vertex_data['opacity'] = self.ops.reshape(-1)
        vertex_data['f_dc_0'] = self.shs[:, 0, 0]
        vertex_data['f_dc_1'] = self.shs[:, 0, 1]
        vertex_data['f_dc_2'] = self.shs[:, 0, 2]

        # 고차원 색상 계수 추가
        for i in range(15):
            vertex_data[f'f_rest_{i}'] = self.shs[:, i, 0]
            vertex_data[f'f_rest_{15 + i}'] = self.shs[:, i, 1]
            vertex_data[f'f_rest_{30 + i}'] = self.shs[:, i, 2]

        # PlyElement 생성
        vertex_element = PlyElement.describe(vertex_data, 'vertex')

        # PLY 파일 저장
        PlyData([vertex_element], text=False).write(output_path)
        
    def upload_to_gpu(self):
        """ Gaussian 파라미터를 GPU로 이동시키고 필요한 변환(스케일, 불투명도)을 적용합니다. """
        if self.on_gpu:
            return

        self.means_gpu = torch.from_numpy(self.means).cuda()
        self.quats_gpu = torch.from_numpy(self.quats).cuda()
        # 스케일은 GPU로 이동 후 exp 적용
        self.scales_gpu = torch.exp(torch.from_numpy(self.scales).cuda())
        # 불투명도는 GPU로 이동 후 sigmoid 적용
        self.ops_gpu = torch.sigmoid(torch.from_numpy(self.ops.reshape(-1)).cuda())
        self.shs_gpu = torch.from_numpy(self.shs).cuda()
        self.colors_gpu = torch.from_numpy(self.colors).cuda()

        self.on_gpu = True
        print("Gaussian data uploaded to GPU.")

    def scale(self, scale_factor: float):
        """ Gaussian의 위치와 스케일을 조정합니다. """
        if self.on_gpu:
            print("Warning: Cannot scale after GPU upload. Scale before upload_to_gpu().")
            return
            
        # 위치 스케일링
        self.means *= scale_factor
        
        # 스케일 로그값 조정 (exp(log_scale) * scale_factor = exp(log_scale + log(scale_factor)))
        self.scales += np.log(scale_factor)
        
        print(f"Applied scale factor {scale_factor} to Gaussian scene.")

    def render(self, viewmats, ks, width, height, near, far):
        viewmats = viewmats.cuda()
        ks = ks.cuda()

        return rasterization(
            means=self.means_gpu,
            quats=self.quats_gpu,
            scales=self.scales_gpu,
            opacities=self.ops_gpu,
            colors=self.shs_gpu,
            viewmats=viewmats,
            Ks=ks,
            width=width,
            height=height,
            near_plane=near,
            far_plane=far,
            packed=False,
            radius_clip=0.1,
            sh_degree=2,
            eps2d=0.3,
            render_mode="RGB+D",
            rasterize_mode="antialiased",
            camera_model="pinhole"
        )
    
    def render_2dgs(self, 
                    viewmats, 
                    ks, 
                    width, height, 
                    near, far, 
                    backgrounds=torch.zeros((1, 3), dtype=torch.float32, device="cuda")):
        viewmats = viewmats.cuda()
        ks = ks.cuda()
    
        return rasterization_2dgs(
            means=self.means_gpu,
            quats=self.quats_gpu,
            scales=self.scales_gpu,
            opacities=self.ops_gpu,
            colors=self.shs_gpu,
            viewmats=viewmats,
            Ks=ks,
            width=width,
            height=height,
            near_plane=near,
            far_plane=far,
            sh_degree=3,
            backgrounds=backgrounds,
            render_mode="RGB+ED",
            depth_mode="expected"
        )



class Gaussian4DScene:
    scene_cam: MiniCam

    def __init__(self):
        parser = ArgumentParser(description="Testing script parameters")
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        hyperparam = ModelHiddenParams(parser)
        parser.add_argument("--iteration", default=-1, type=int)
        parser.add_argument("--skip_train", action="store_true")
        parser.add_argument("--skip_test", action="store_true")
        parser.add_argument("--quiet", action="store_true")
        parser.add_argument("--skip_video", action="store_true")
        parser.add_argument("--configs", type=str)

        args = parser.parse_args(args=['--model_path', 'flame_steak/', 
                                       '--skip_train', 
                                       '--configs',  'arguments/multipleview/flame_steak.py'])

        with open("../flame_steak_cp/cfg_args.json", "r") as f:
            cfg_args = json.load(f)

        args.__dict__ = cfg_args

        self.dataset = model.extract(args)
        self.hp = hyperparam.extract(args) 
        self.iteration=args.iteration

        self.gaussians = GaussianModel(self.dataset.sh_degree, self.hp)
        self.scene = Scene(self.dataset, self.gaussians, load_iteration=self.iteration, shuffle=False)
        self.bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")

        print(f"Loaded {self.gaussians.get_xyz.shape[0]} Gaussians.")

        self.cam_type = self.scene.dataset_type
        self.pipeline = pipeline

        temp_cam = self.scene.getVideoCameras()[0]

        self.scene_cam = MiniCam(
            temp_cam.image_width,
            temp_cam.image_height,
            temp_cam.FoVy,
            temp_cam.FoVx,
            temp_cam.znear,
            temp_cam.zfar,
            temp_cam.world_view_transform,
            temp_cam.full_proj_transform,
            0.0,
        )

        print(self.scene_cam)

        self.time_rel = lambda x: x / 10

    def render_4dgs(self, viewmat: torch.Tensor, projmat: torch.Tensor, width: float, height: float, near: float, far: float, scene_time: float):
        self.scene_cam.world_view_transform = viewmat
        self.scene_cam.full_proj_transform = (viewmat.unsqueeze(0).bmm(projmat.unsqueeze(0))).squeeze(0)

        fov_deg = 80
        fov_rad = math.radians(fov_deg)
        aspect = width / height

        self.scene_cam.image_width = width
        self.scene_cam.image_height = height
        self.scene_cam.near = near
        self.scene_cam.far = far
        self.scene_cam.FoVy = fov_rad
        self.scene_cam.FoVx = 2 * math.atan(math.tan(fov_rad / 2) * aspect)
        self.scene_cam.time = scene_time

        result = render(self.scene_cam, self.gaussians, self.pipeline, self.background, 1.0, cam_type=self.cam_type)
        return result["render"], result["depth"], result["alpha"]


if __name__ == "__main__":
    import cv2
    scene_4d = Gaussian4DScene()

    temp_cam: Camera = scene_4d.scene.getVideoCameras()[0]

    print(temp_cam.world_view_transform)
    print(temp_cam.projection_matrix)

    view = temp_cam.world_view_transform.squeeze().cuda()
    # proj = torch.tensor(
    #     [1.0358232159930985, 0, 0, 0, 
    #      0, 1.19175359259421, 0, 0, 
    #      0, 0, -1.0408163265306123 * -1, -1 * -1, 
    #      0, 0, -2.0408163265306123, 0], device="cuda")
    # proj = proj.reshape(4, 4).squeeze(0)
    # print(proj)


    # rendered_color, rendered_depth, rendered_alpha = scene_4d.render_4dgs(viewmat=view, 
    #                                                       projmat=proj, 
    #                                                       width=1712, height=1488, 
    #                                                       near=1, far=20, scene_time=0)

    rendered_color, rendered_depth, rendered_alpha = scene_4d.render_4dgs(viewmat=view, 
                                                        projmat=temp_cam.projection_matrix.squeeze().cuda(), 
                                                        width=temp_cam.image_width, height=temp_cam.image_height, 
                                                        near=temp_cam.znear, far=temp_cam.zfar, scene_time=0)

    
    print(rendered_alpha.squeeze().min(), rendered_alpha.squeeze().max())
    scaled_depth = (rendered_depth - 0.1) / (100 - 0.1)
    depth_img = scaled_depth.mul(255).squeeze().to(torch.uint8).cpu().detach().numpy()
    img = rendered_color.permute(1, 2, 0).mul(255).to(torch.uint8).cpu().detach().numpy()
    alpha_img = rendered_alpha.mul(255).squeeze().to(torch.uint8).cpu().detach().numpy()
    
    cv2.imwrite("rendered_color.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite("depth_img.jpg", depth_img)
    cv2.imwrite("alpha_img.jpg", alpha_img)