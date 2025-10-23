"""
Optimization and pipeline parameters for 3DGStream rendering.
"""


class SimplePipelineParams:
    """Pipeline parameters for Gaussian rendering."""

    def __init__(self):
        self.debug = False
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.bwd_depth = True


class OptimizationParams:
    """Optimization parameters for 3DGStream training."""

    def __init__(self):
        self.iterations = 10
        self.iterations_s2 = 20
        self.first_load_iteration = 15000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.depth_smooth = 0.0
        self.ntc_lr = None
        self.lambda_dxyz = 0.0
        self.lambda_drot = 0.0
        # Densification parameters
        self.densification_interval = 20
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 130
        self.densify_until_iter = 15000
        self.densify_grad_threshold = 0.00015
        self.ntc_conf_path = ""
        self.ntc_path = ""
        self.batch_size = 1
        self.spawn_type = "spawn"
        self.s2_type = "spawn"
        self.s2_adding = True
        self.num_of_split = 1
        self.num_of_spawn = 1
        self.std_scale = 2
        self.min_opacity = 0.01
        self.rotate_sh = True
        self.only_mlp = False