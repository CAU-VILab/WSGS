import copy
import torch
import numpy as np
import random
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from skimage.metrics import structural_similarity as compare_ssim

from easyvolcap.engine import cfg
from easyvolcap.engine import SAMPLERS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.models.networks.noop_network import NoopNetwork
from easyvolcap.models.samplers.refractive_gaussian2d_sampler import RefractiveGaussian2DSampler

from easyvolcap.utils.sh_utils import *
from easyvolcap.utils.console_utils import *
from easyvolcap.utils.math_utils import normalize
from easyvolcap.utils.grid_utils import sample_points_subgrid
from easyvolcap.utils.colmap_utils import load_sfm_ply, save_sfm_ply
from easyvolcap.utils.net_utils import freeze_module, make_params, make_buffer
from easyvolcap.utils.refractive_gaussian2d_utils import GaussianModel, render, prepare_gaussian_camera, get_step_lr_func
from easyvolcap.utils.data_utils import load_pts, export_pts, to_x, to_cuda, to_cpu, to_tensor, remove_batch
from easyvolcap.utils.depth_utils import normalize_depth

from typing import List, Tuple
import torchvision.utils as vutils
from skimage.metrics import structural_similarity as ssim
import os
import cv2
import kornia
@SAMPLERS.register_module()
class WSGSSampler(RefractiveGaussian2DSampler):
    def __init__(self,
                 # Legacy APIs
                 network: NoopNetwork = None,  # ignore this 

                 # 3DGS-DR related configs
                 sh_start_iter: int = 10000,
                 densify_until_iter: int = 30000,
                 init_densification_interval: int = 100,
                 norm_densification_interval: int = 500, #TBD 500
                 normal_prop_until_iter: int = 24000,
                 normal_prop_interval: int = 1000,
                 opacity_lr0_interval: int = 200,
                 opacity_lr: float = 0.05,
                 color_sabotage_until_iter: int = 24000,
                 color_sabotage_interval: int = 1000,
                 reset_specular_all: bool = False,
                 # + refraction
                 reset_refraction_all: bool = False,

                 # Gaussian configs
                 env_preload_gs: str = '',
                 env_bounds: List[List[float]] = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                 
                 refracted_preload_gs: str = '',
                 refracted_bounds: List[List[float]] = [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
                 # SHs configs
                 env_sh_deg: int = 3,
                 env_init_sh_deg: int = 0,
                 env_sh_start_iter: int = 0,
                 env_sh_update_iter: int = 1000,

                 refracted_sh_deg: int = 3,
                 refracted_init_sh_deg: int = 0,
                 refracted_sh_start_iter: int = 0,
                 refracted_sh_update_iter: int = 1000,
                 # Opacity and scale configs
                 env_init_occ: float = 0.1,
                 refracted_init_occ: float = 0.1,

                 # Densify & pruning configs
                 env_densify_from_iter: int = 500,
                 env_densify_until_iter: int = 15000,
                 env_densification_interval: int = 100,
                 env_opacity_reset_interval: int = 3000,
                 env_densify_grad_threshold: float = 0.0002,
                 env_min_opacity: float = 0.05,
                 env_densify_size_threshold: float = 0.01,  # alias for `percent_dense` as in the original code, https://github.com/hbb1/2d-gaussian-splatting/blob/6d249deeec734ad07760496fc32be3b91ac236fc/scene/gaussian_model.py#L378
                 env_prune_large_gs: bool = True,
                 env_prune_visibility: bool = False,
                 env_max_scene_threshold: float = 0.1,  # default 0.1, same as the original 2DGS
                 env_max_screen_threshold: float = None,  # not used in the original 3DGS/2DGS, they wrote a bug, though `max_screen_threshold=20` 
                 env_min_weight_threshold: float = None, 

                 refracted_densify_from_iter: int = 500,
                 refracted_densify_until_iter: int = 15000,
                 refracted_densification_interval: int = 100,
                 refracted_opacity_reset_interval: int = 3000,
                 refracted_densify_grad_threshold: float = 0.0002,
                 refracted_min_opacity: float = 0.05, #TBD 0.05
                 refracted_densify_size_threshold: float = 0.01,  # alias for `percent_dense` as in the original code, https://github.com/hbb1/2d-gaussian-splatting/blob/6d249deeec734ad07760496fc32be3b91ac236fc/scene/gaussian_model.py#L378
                 refracted_prune_large_gs: bool = True,
                 refracted_prune_visibility: bool = False,
                 refracted_max_scene_threshold: float = 0.1,  # default 0.1, same as the original 2DGS
                 refracted_max_screen_threshold: float = None,  # not used in the original 3DGS/2DGS, they wrote a bug, though `max_screen_threshold=20` 
                 refracted_min_weight_threshold: float = None, 
                 # EasyVolcap additional densify & pruning tricks
                 env_screen_until_iter: int = int(4000 / 60 * cfg.runner_cfg.epochs),
                 env_split_screen_threshold: float = None, 
                 env_min_gradient: float = None, 

                 refracted_screen_until_iter: int = int(4000 / 60 * cfg.runner_cfg.epochs),
                 refracted_split_screen_threshold: float = None, 
                 refracted_min_gradient: float = None, 
                 # Rendering configs
                 env_white_bg: bool = False,  # always set to False !!!
                 env_bg_brightness: float = 0.0,  # used in the original renderer
                 
                 refracted_white_bg: bool = False,  # always set to False !!!
                 refracted_bg_brightness: float = 0.0,  # used in the original renderer

                 # Reflection related parameters
                 render_reflection: bool = True,  # default is True here
                 render_reflection_start_iter: int = 3000,  # need a initial geometry to model reflection
                 # Refraction related parameters
                 render_refraction: bool = True,  # default is True here
                 render_refraction_start_iter: int = 3000,  # need a initial geometry to model refraction

                 detach: bool = False,  # detach the reflected/refracted rays for training the reflection model

                 # Ray tracing configs
                 use_optix_tracing: bool = True,
                 use_base_tracing: bool = False,
                 tracing_backend: str = 'cpp',
                 env_max_gs: float = 3e5,  # control the maximum number of gaussians 5e5 TBD
                 env_max_gs_threshold: float = 0.9,  # percentage of the visibility pruning
                 refracted_max_gs: float = 5e5,  # control the maximum number of gaussians, or GPU explode #TBD 7e5
                 refracted_max_gs_threshold: float = 0.9,  # percentage of the visibility pruning
                 prune_visibility: bool = True,  # whether to prune the gaussians based on accumulated weights
                 max_trace_depth: int = 0,
                 specular_threshold: float = 0.0,  # specular threshold for reflection rendering
                 n_sample_dirs: int = 1, # number of sampled reflected directions
                 specular_filtering_start_iter: int = -1,  # start to filter pixels with large specular values
                 specular_filtering_percent: float = 0.75,  # percentage of pixels to be filtered
                 acc_filtering_start_iter: int = -1,  # start to filter pixels with large accumulated weights
                 multi_sampling_start_iter: int = -1,  # start to use multi-sample for reflection rendering

                 # water plane related parameters
                 initial_plane_normal: list = [0.0,-1.0,0.0],
                 initial_plane_offset: float = 0.01,
                 plane_offset_freeze_iter: int = 20000,

                 env_xyz_lr_scheduler: dotdict = None,
                 refracted_xyz_lr_scheduler: dotdict = None,
                 
                 plane_offset_lr_scheduler: dotdict = None,


                 # Default parameters for Gaussian2DSampler
                 **kwargs,
                 ):
        # Inherit from the default `VolumetricVideoDataset`
        call_from_cfg(super().__init__,
                      kwargs,
                      network=network,
                      sh_start_iter=sh_start_iter,
                      densify_until_iter=densify_until_iter,
                      render_reflection=render_reflection,
                      render_refraction=render_refraction,
                      use_optix_tracing=use_optix_tracing,
                      tracing_backend=tracing_backend,
                      prune_visibility=prune_visibility,
                      max_trace_depth=max_trace_depth,
                      specular_threshold=specular_threshold)

        # 3DGS-DR related configs
        self.init_densification_interval = init_densification_interval
        self.norm_densification_interval = norm_densification_interval
        self.normal_prop_until_iter = normal_prop_until_iter
        self.normal_prop_interval = normal_prop_interval
        self.opacity_lr0_interval = opacity_lr0_interval
        self.opacity_lr = opacity_lr
        self.color_sabotage_until_iter = color_sabotage_until_iter
        self.color_sabotage_interval = color_sabotage_interval
        self.reset_specular_all = reset_specular_all
        self.reset_refraction_all = reset_refraction_all

        # Reflection & Refraction related parameters
        self.use_base_tracing = use_base_tracing
        self.render_reflection_start_iter = render_reflection_start_iter
        self.render_refraction_start_iter = render_refraction_start_iter
        self.n_sample_dirs = n_sample_dirs
        self.detach = detach
        self.specular_filtering_start_iter = specular_filtering_start_iter
        self.specular_filtering_percent = specular_filtering_percent
        self.acc_filtering_start_iter = acc_filtering_start_iter
        self.multi_sampling_start_iter = multi_sampling_start_iter

        # Environment&refracted Gaussian related parameters
        self.env_preload_gs = env_preload_gs
        self.env_bounds = env_bounds
        self.refracted_preload_gs = refracted_preload_gs
        self.refracted_bounds = refracted_bounds

        # Environment& SH related parameters
        self.env_sh_deg = env_sh_deg
        self.env_init_sh_deg = env_init_sh_deg
        self.env_sh_start_iter = env_sh_start_iter
        self.env_sh_update_iter = env_sh_update_iter

        self.refracted_sh_deg = refracted_sh_deg
        self.refracted_init_sh_deg = refracted_init_sh_deg
        self.refracted_sh_start_iter = refracted_sh_start_iter
        self.refracted_sh_update_iter = refracted_sh_update_iter

        # Environment&refracted opacity and scale parameters
        self.env_init_occ = env_init_occ
        self.refracted_init_occ = refracted_init_occ

        # Densify & pruning parameters
        self.env_densify_from_iter = env_densify_from_iter
        self.env_densify_until_iter = env_densify_until_iter
        self.env_densification_interval = env_densification_interval
        self.env_opacity_reset_interval = env_opacity_reset_interval
        self.env_densify_grad_threshold = env_densify_grad_threshold
        self.env_min_opacity = env_min_opacity
        self.env_densify_size_threshold = env_densify_size_threshold
        self.env_prune_large_gs = env_prune_large_gs
        self.env_prune_visibility = env_prune_visibility
        self.env_max_scene_threshold = env_max_scene_threshold
        self.env_max_screen_threshold = env_max_screen_threshold
        self.env_min_weight_threshold = env_min_weight_threshold

        self.refracted_densify_from_iter = refracted_densify_from_iter
        self.refracted_densify_until_iter = refracted_densify_until_iter
        self.refracted_densification_interval = refracted_densification_interval
        self.refracted_opacity_reset_interval = refracted_opacity_reset_interval
        self.refracted_densify_grad_threshold = refracted_densify_grad_threshold
        self.refracted_min_opacity = refracted_min_opacity
        self.refracted_densify_size_threshold = refracted_densify_size_threshold
        self.refracted_prune_large_gs = refracted_prune_large_gs
        self.refracted_prune_visibility = refracted_prune_visibility
        self.refracted_max_scene_threshold = refracted_max_scene_threshold
        self.refracted_max_screen_threshold = refracted_max_screen_threshold
        self.refracted_min_weight_threshold = refracted_min_weight_threshold

        # EasyVolcap additional densify & pruning tricks
        self.env_screen_until_iter = env_screen_until_iter
        self.env_split_screen_threshold = env_split_screen_threshold
        self.env_min_gradient = env_min_gradient
        self.env_max_gs = env_max_gs
        self.env_max_gs_threshold = env_max_gs_threshold

        self.refracted_screen_until_iter = refracted_screen_until_iter
        self.refracted_split_screen_threshold = refracted_split_screen_threshold
        self.refracted_min_gradient = refracted_min_gradient
        self.refracted_max_gs = refracted_max_gs
        self.refracted_max_gs_threshold = refracted_max_gs_threshold

        # Store the last output for updating the gaussians
        self.last_output_env = None
        self.last_output_env_plane = None
        self.last_output_refracted = None

        #
        self.plane_offset_freeze_iter = plane_offset_freeze_iter  # freeze the plane offset after this iteration
        self.plane_offset_lr_scheduler = plane_offset_lr_scheduler

        # plane_offset용 scheduler 생성
        if plane_offset_lr_scheduler is not None:
            self.plane_offset_scheduler = get_step_lr_func(**plane_offset_lr_scheduler)
            log(magenta(f'[INIT] Using plane_offset learning rate scheduler, lr_init: {plane_offset_lr_scheduler["lr_init"]}, lr_final: {plane_offset_lr_scheduler["lr_final"]}'))
        else:
            self.plane_offset_scheduler = None

        self.env_xyz_lr_scheduler = env_xyz_lr_scheduler
        self.refracted_xyz_lr_scheduler = refracted_xyz_lr_scheduler

        xyz_env, colors_env = self.init_env_points(self.env_preload_gs)
        xyz_refracted, colors_refracted = self.init_refracted_points(self.refracted_preload_gs)
        

        #Trainable water plane parameters
        pn = nn.Parameter(torch.Tensor(initial_plane_normal), requires_grad=False) # Fixed normal

        # For a plane like y=d, 'd' is the offset. Initial value from the y-component of old pp.
        self.plane_normal = make_buffer(pn) 
        self._plane_offset = make_params(nn.Parameter(torch.tensor([initial_plane_offset], dtype=torch.float32)))

        self.plane_offset_history = []
        self.plane_offset_moving_avg_window = 10  # 10개 iteration의 이동 평균 사용

        # Create environment Gaussians and refracted Gaussians
        self.env = GaussianModel(
            xyz=xyz_env,
            colors=colors_env,
            init_occ=self.env_init_occ,
            init_scale=None,
            sh_degree=self.env_sh_deg,
            init_sh_degree=self.env_init_sh_deg,
            spatial_scale=self.spatial_scale,
            xyz_lr_scheduler=self.env_xyz_lr_scheduler,
            render_reflection=False,
            max_gs=self.env_max_gs,
            max_gs_threshold=self.env_max_gs_threshold
        )
        
        #TBD
        self.refracted = GaussianModel(                         
            xyz=xyz_refracted,
            colors=colors_refracted,
            init_occ=self.refracted_init_occ,
            init_scale=None,
            sh_degree=self.refracted_sh_deg,
            init_sh_degree=self.refracted_init_sh_deg,
            spatial_scale=self.spatial_scale,
            xyz_lr_scheduler=self.refracted_xyz_lr_scheduler,
            render_reflection=False,
            max_gs=self.refracted_max_gs,
            max_gs_threshold=self.refracted_max_gs_threshold
        )



        # Update `self.pipe`
        self.pipe.convert_SHs_python = True  # enable SH -> RGB conversion in Python
        if self.use_base_tracing: self.pipe.convert_SHs_python = False
        self.pipe_env = copy.deepcopy(self.pipe)
        self.pipe_env.convert_SHs_python = False

        self.pipe_refracted = copy.deepcopy(self.pipe)
        self.pipe_refracted.convert_SHs_python = False        

        # Rendering configs of environment&refracted Gaussian
        self.env_white_bg = env_white_bg
        self.env_bg_brightness = 1. if env_white_bg else env_bg_brightness
        self.env_bg_channel = 3
        self.env_bg_color = make_buffer(torch.Tensor([self.env_bg_brightness] * self.env_bg_channel))

        self.refracted_white_bg = refracted_white_bg
        self.refracted_bg_brightness = 1. if refracted_white_bg else refracted_bg_brightness
        self.refracted_bg_channel = 3
        self.refracted_bg_color = make_buffer(torch.Tensor([self.refracted_bg_brightness] * self.refracted_bg_channel))

        # Time statistics
        self.times = []

    def init_env_points(self, ply_file: str = None, S: int = 32, N: int = 5):
        # Try to load the ply file
        try:
            xyz, rgb = load_sfm_ply(ply_file)  # (P, 3), (P, 3)
            log(yellow(f"Loaded the reflection point cloud from {ply_file}."))
            xyz = torch.as_tensor(xyz, dtype=torch.float)
            rgb = torch.as_tensor(rgb, dtype=torch.float)  # already normalized to [0, 1]
        # If the file does not exist, generate random points and save them
        except:
            log(yellow(f"Failed to load the reflection point cloud from {ply_file}, generating random points."))
            xyz = sample_points_subgrid(torch.as_tensor(self.env_bounds), S, N).float()  # (P, 3)
            rgb = torch.rand(xyz.shape, dtype=torch.float) / 255.0  # (P, 3)
            save_sfm_ply(ply_file, xyz.numpy(), rgb.numpy() * 255.0)

        return xyz, rgb

    def init_refracted_points(self, ply_file: str = None, S: int = 32, N: int = 5): #TBD : min(z)~max(z) 범위로 yaml 지정하도록 코드 변경.
        #Same function as init_env_points now.
        # Try to load the ply file
        try:
            xyz, rgb = load_sfm_ply(ply_file)  # (P, 3), (P, 3)
            log(yellow(f"Loaded the refracted point cloud from {ply_file}."))
            xyz = torch.as_tensor(xyz, dtype=torch.float)
            rgb = torch.as_tensor(rgb, dtype=torch.float)  # already normalized to [0, 1]
        # If the file does not exist, generate random points and save them
        except:
            log(yellow(f"Failed to load the refracted point cloud from {ply_file}, generating random points."))
            xyz = sample_points_subgrid(torch.as_tensor(self.refracted_bounds), S, N).float()  # (P, 3)
            rgb = torch.rand(xyz.shape, dtype=torch.float) / 255.0  # (P, 3)
            save_sfm_ply(ply_file, xyz.numpy(), rgb.numpy() * 255.0)

        return xyz, rgb


    def update_plane_offset_learning_rate(self, iter: int, optimizer: Adam):
        """plane_offset의 learning rate 업데이트"""
        if self.plane_offset_scheduler is not None and iter < self.plane_offset_freeze_iter:
            new_lr = self.plane_offset_scheduler(iter)
            for param_group in optimizer.param_groups:
                if 'sampler._plane_offset' in param_group.get('name', ''):
                    old_lr = param_group['lr']
                    param_group['lr'] = float(new_lr)
                    if iter % 10 == 0:
                        print(new_lr)
                    break


    @torch.no_grad()
    def update_dif_gaussians(self, batch: dotdict):
        if not self.training: return

        # Update the densification interval
        if batch.meta.iter < self.render_reflection_start_iter: self.densification_interval = self.init_densification_interval
        elif batch.meta.iter < self.normal_prop_until_iter: self.densification_interval = self.norm_densification_interval #TBD
        else: self.densification_interval = self.init_densification_interval

        # Prepare global variables
        iter: int = batch.meta.iter  # controls whether we're to update in this iteration
        output = self.last_output  # contains necessary information for updating gaussians
        optimizer: Adam = cfg.runner.optimizer

        # plane_offset 학습 중단 (특정 iteration 이후) TBD
        if iter < self.plane_offset_freeze_iter:
            self.update_plane_offset_learning_rate(iter, optimizer)

            # # --- 이동 평균을 이용한 plane_offset 업데이트 ---
            # # 현재 값 (이전 optimizer.step()의 결과)을 기록
            # current_offset = self._plane_offset.data.clone()
            # self.plane_offset_history.append(current_offset)

            # # 윈도우 크기 유지
            # if len(self.plane_offset_history) > self.plane_offset_moving_avg_window:
            #     self.plane_offset_history.pop(0)

            # # 이동 평균 계산 및 적용
            # if self.plane_offset_history:
            #     moving_avg = torch.stack(self.plane_offset_history).mean()
            #     print("moving : ", moving_avg.item(), current_offset.item())
            #     # 파라미터 값을 이동 평균으로 직접 설정
            #     self._plane_offset.data.fill_(moving_avg)


        if iter == self.plane_offset_freeze_iter:
            for param_group in optimizer.param_groups:
                if 'sampler._plane_offset' in param_group.get('name', ''):
                    param_group['lr'] = 0.0
                    log(yellow_slim(f'[PLANE OFFSET FREEZE] Learning rate set to 0 at iter {iter}'))
                    break



        # Log the total number of gaussians
        scalar_stats = batch.output.get('scalar_stats', dotdict())
        scalar_stats.num_pts = self.pcd.number
        batch.output.scalar_stats = scalar_stats
        # Log the last opacity reset iteration
        batch.output.last_opacity_reset_iter = self.opacity_reset_interval * (iter // self.opacity_reset_interval)

        # Update the learning rate
        self.pcd.update_learning_rate(iter.item(), optimizer, prefix='sampler.pcd.')

        # Increase the levels of SHs every `self.sh_update_iter=1000` iterations until a maximum degree
        if iter > 0 and iter < self.densify_until_iter and iter % self.sh_update_iter == 0 and self.sh_start_iter is not None and iter > self.sh_start_iter:
            changed = self.pcd.oneupSHdegree()
            if changed: log(yellow_slim(f'[ONEUP SH DEGREE] sh_deg: {self.pcd.active_sh_degree.item()}'))

        #TBD Remove refraction above the camera position. Water plane can't be above the camera.
        #if iter > self.render_refraction_start_iter and iter % 50 == 0: 
        #    self.pcd.set_refraction_for_upper_gaussians_by_position(self.plane_normal, batch, optimizer, prefix='sampler.pcd.')


        opacity_reset_flag = False
        # Update only the rendered frame
        if iter > 0 and iter < self.densify_until_iter and output is not None:
            # Update all rendered gaussians in the batch
            pcd: GaussianModel = self.pcd

            # Preparing gaussian status for update
            visibility_filter = output.visibility_filter
            viewspace_point_tensor = output.viewspace_points  # no indexing, otherwise no grad # !: BATCH
            if output.viewspace_points.grad is None: return  # previous rendering was an evaluation
            if 'weight_accumulate' not in output: pcd.add_densification_stats(viewspace_point_tensor, visibility_filter)
            else: pcd.add_densification_stats(viewspace_point_tensor, visibility_filter, output.weight_accumulate)

            # Update gaussian splatting radii for update
            if not self.use_optix_tracing:
                radii = output.radii
                pcd.max_radii2D[visibility_filter] = torch.max(pcd.max_radii2D[visibility_filter], radii[visibility_filter])

            # Perform densification and pruning
            if iter > self.densify_from_iter and iter % self.densification_interval == 0:
                log(yellow_slim(f'Start updating gaussians of step: {iter:06d}'))
                # Iteration-related densification and pruning parameters
                split_screen_threshold = self.split_screen_threshold if iter < self.screen_until_iter else None
                max_screen_threshold = self.max_screen_threshold if iter > self.opacity_reset_interval else None
                # Perform actual densification and pruning
                pcd.densify_and_prune(
                    self.min_opacity,
                    self.min_gradient,
                    self.densify_grad_threshold,
                    self.densify_size_threshold,
                    split_screen_threshold,
                    self.max_scene_threshold,
                    max_screen_threshold,
                    self.min_weight_threshold,
                    self.prune_visibility,
                    optimizer,
                    self.prune_large_gs,
                    prefix='sampler.pcd.'
                )
                log(yellow_slim('Densification and pruning done! ' +
                                f'min opacity: {pcd.get_opacity.min().item():.4f} ' +
                                f'max opacity: {pcd.get_opacity.max().item():.4f} ' +
                                f'number of points: {pcd.get_xyz.shape[0]}'))

            #opacity_reset_flag = False
            # Perform opacity reset
            if iter % self.opacity_reset_interval == 0:
                # Reset the opacity of the gaussians to 0.01 (default)
                pcd.reset_opacity(optimizer=optimizer, prefix='sampler.pcd.')
                log(yellow_slim('Resetting opacity done! ' +
                                f'min opacity: {pcd.get_opacity.min().item():.4f} ' +
                                f'max opacity: {pcd.get_opacity.max().item():.4f}'))
                opacity_reset_flag = True

                if iter > self.opacity_reset_interval and iter > self.render_reflection_start_iter:
                    # Reset the specular of the gaussians to 0.001 (default)
                    pcd.reset_specular(
                        reset_specular=self.init_specular,
                        reset_specular_all=self.reset_specular_all,
                        optimizer=optimizer,
                        prefix='sampler.pcd.'
                    )
                    log(yellow_slim('Resetting specular done! ' +
                                    f'min specular: {pcd.get_specular.min().item():.4f} ' +
                                    f'max specular: {pcd.get_specular.max().item():.4f}'))
                
                # if iter > self.opacity_reset_interval and iter > self.render_refraction_start_iter: #TBD
                #     # Reset the refraction of the gaussians to default
                #     pcd.reset_refraction(
                #         reset_refraction=self.init_refraction,
                #         reset_refraction_all=self.reset_refraction_all,
                #         optimizer=optimizer,
                #         prefix='sampler.pcd.'
                #     )
                #     log(yellow_slim('Resetting refraction done! ' +
                #                     f'min refraction: {pcd.get_refraction.min().item():.4f} ' +
                #                     f'max refraction: {pcd.get_refraction.max().item():.4f}'))

            if self.opacity_lr0_interval > 0 and iter % self.opacity_lr0_interval == 0 and self.render_reflection_start_iter < iter <= self.normal_prop_until_iter:
                pcd.update_learning_rate_by_name(
                    name='_opacity',
                    lr=self.opacity_lr,
                    optimizer=optimizer,
                    prefix='sampler.pcd.'
                )

        #can works after base gaussian densification
        if iter > 0 and output is not None and self.render_reflection_start_iter < iter <= self.refracted_densify_until_iter \
            and not opacity_reset_flag:
            pcd: GaussianModel = self.pcd
            # black color sabotage
            if iter % self.color_sabotage_interval == 0:
                #if iter > self.plane_offset_freeze_iter: #zero_out 시에 plane 아래 가우시안들은 전부 영향을 받도록 refraction reset
                self.pcd.reset_refraction_for_lower_gaussians_by_plane(self.plane_normal, self._plane_offset, self.init_refraction, optimizer, prefix='sampler.pcd.') 
                pcd.zero_out_color_black(optimizer=optimizer, prefix='sampler.pcd.')

            #if iter > self.plane_offset_freeze_iter and iter % 50 == 0: #remove refraction(set 1) for gaussians above the plane
            #    self.pcd.set_refraction_for_upper_gaussians_by_plane(self.plane_normal, self._plane_offset, optimizer, prefix='sampler.pcd.')    
            
            
            # if self.render_reflection_start_iter < iter <= self.color_sabotage_until_iter and iter % self.color_sabotage_interval == 0 and not opacity_reset_flag:
            #     #pcd.distort_color(optimizer=optimizer, prefix='sampler.pcd.')
            #     #pcd.distort_color_refraction(optimizer=optimizer, prefix='sampler.pcd.')
            #     pcd.zero_out_color_black(optimizer=optimizer, prefix='sampler.pcd.')
            
            #    pcd.enlarge_refraction(optimizer=optimizer, prefix='sampler.pcd.') #Not original code. dismiss this.

            # Normal Propagation
            # if self.render_reflection_start_iter < iter <= self.normal_prop_until_iter and iter % self.normal_prop_interval == 0 and not opacity_reset_flag:
            #     # Reset the opacity of the gaussians to 0.9 (default)
            #     pcd.enlarge_opacity(optimizer=optimizer, prefix='sampler.pcd.')
            #     pcd.enlarge_scaling(optimizer=optimizer, prefix='sampler.pcd.')
            #     if self.opacity_lr0_interval > 0 and iter != self.normal_prop_until_iter:
            #         pcd.update_learning_rate_by_name(
            #             name='_opacity',
            #             lr=0.0,
            #             optimizer=optimizer,
            #             prefix='sampler.pcd.'
            #         )



    @torch.no_grad()
    def update_env_gaussians(self, batch: dotdict):
        if not self.training: return

        # Log the total number of gaussians
        scalar_stats = batch.output.get('scalar_stats', dotdict())
        scalar_stats.env_num_pts = self.env.number
        batch.output.scalar_stats = scalar_stats

        # Prepare global variables
        iter: int = batch.meta.iter  # controls whether we're to update in this iteration
        output = self.last_output_env  # contains necessary information for updating gaussians
        optimizer: Adam = cfg.runner.optimizer
        # Return if we're not in the update iteration
        if iter <= self.render_reflection_start_iter: return

        # Update the learning rate
        self.env.update_learning_rate(iter.item(), optimizer, prefix='sampler.env.')

        # Increase the levels of SHs every `self.sh_update_iter=1000` iterations until a maximum degree
        if iter > 0 and iter < self.env_densify_until_iter and iter % self.env_sh_update_iter == 0 and self.env_sh_start_iter is not None and iter > self.env_sh_start_iter:
            changed = self.env.oneupSHdegree()
            if changed: log(green_slim(f'[ONEUP SH DEGREE] sh_deg: {self.env.active_sh_degree.item()}'))

        # Update only the rendered frame
        if iter > 0 and iter < self.env_densify_until_iter and output is not None:
            # Update all rendered gaussians in the batch
            env: GaussianModel = self.env

            # Process geometric reflection output (self.last_output_env)
            output_geom = self.last_output_env
            if output_geom is not None and \
               hasattr(output_geom, 'viewspace_points') and \
               output_geom.viewspace_points is not None and \
               output_geom.viewspace_points.grad is not None:
                
                visibility_filter_geom = output_geom.visibility_filter
                viewspace_point_tensor_geom = output_geom.viewspace_points
                if 'weight_accumulate' not in output_geom:
                    env.add_densification_stats(viewspace_point_tensor_geom, visibility_filter_geom)
                else:
                    env.add_densification_stats(viewspace_point_tensor_geom, visibility_filter_geom, output_geom.weight_accumulate)

            # Process planar reflection output (self.last_output_env_plane)
            output_plane = self.last_output_env_plane
            if output_plane is not None and \
               hasattr(output_plane, 'viewspace_points') and \
               output_plane.viewspace_points is not None and \
               output_plane.viewspace_points.grad is not None:

                visibility_filter_plane = output_plane.visibility_filter
                viewspace_point_tensor_plane = output_plane.viewspace_points
                if 'weight_accumulate' not in output_plane:
                    env.add_densification_stats(viewspace_point_tensor_plane, visibility_filter_plane)
                else:
                    env.add_densification_stats(viewspace_point_tensor_plane, visibility_filter_plane, output_plane.weight_accumulate)

            # Perform densification and pruning
            if iter > self.env_densify_from_iter and iter % self.env_densification_interval == 0:
                log(green_slim(f'Start updating reflection gaussians of step: {iter:06d}'))
                # Iteration-related densification and pruning parameters
                env_split_screen_threshold = self.env_split_screen_threshold if iter < self.env_screen_until_iter else None
                env_max_screen_threshold = self.env_max_screen_threshold if iter > self.env_opacity_reset_interval else None
                # Perform actual densification and pruning
                env.densify_and_prune(
                    self.env_min_opacity,
                    self.env_min_gradient,
                    self.env_densify_grad_threshold,
                    self.env_densify_size_threshold,
                    env_split_screen_threshold,
                    self.env_max_scene_threshold,
                    env_max_screen_threshold,
                    self.env_min_weight_threshold,
                    self.env_prune_visibility,
                    optimizer,
                    self.env_prune_large_gs,
                    prefix='sampler.env.'
                )
                log(green_slim('Densification and pruning done! ' +
                                f'min opacity: {env.get_opacity.min().item():.4f} ' +
                                f'max opacity: {env.get_opacity.max().item():.4f} ' +
                                f'number of points: {env.get_xyz.shape[0]}'))

            # Perform opacity reset
            if iter % self.env_opacity_reset_interval == 0:
                env.reset_opacity(optimizer=optimizer, prefix='sampler.env.')
                log(green_slim('Resetting opacity done! ' +
                                f'min opacity: {env.get_opacity.min().item():.4f} ' +
                                f'max opacity: {env.get_opacity.max().item():.4f}'))


    @torch.no_grad()
    def update_refracted_gaussians(self, batch: dotdict): #TBD
        if not self.training: return

        # Log the total number of gaussians
        scalar_stats = batch.output.get('scalar_stats', dotdict())
        scalar_stats.refracted_num_pts = self.refracted.number
        batch.output.scalar_stats = scalar_stats

        # Prepare global variables
        iter: int = batch.meta.iter  # controls whether we're to update in this iteration
        output = self.last_output_refracted  # contains necessary information for updating gaussians
        optimizer: Adam = cfg.runner.optimizer
        # Return if we're not in the update iteration
        if iter <= self.render_refraction_start_iter: return

        # Update the learning rate
        self.refracted.update_learning_rate(iter.item(), optimizer, prefix='sampler.refracted.')

        # Increase the levels of SHs every `self.sh_update_iter=1000` iterations until a maximum degree
        if iter > 0 and iter < self.refracted_densify_until_iter and iter % self.refracted_sh_update_iter == 0 and self.refracted_sh_start_iter is not None and iter > self.refracted_sh_start_iter:
            changed = self.refracted.oneupSHdegree()
            if changed: log(green_slim(f'[ONEUP SH DEGREE] sh_deg: {self.refracted.active_sh_degree.item()}'))

        # Update only the rendered frame
        if iter > 0 and iter < self.refracted_densify_until_iter and output is not None:
            # Update all rendered gaussians in the batch
            refracted: GaussianModel = self.refracted

            # Preparing gaussian status for update
            visibility_filter = output.visibility_filter
            viewspace_point_tensor = output.viewspace_points  # no indexing, otherwise no grad # !: BATCH
            if output.viewspace_points.grad is None: return  # previous rendering was an evaluation
            if 'weight_accumulate' not in output: refracted.add_densification_stats(viewspace_point_tensor, visibility_filter)
            else: refracted.add_densification_stats(viewspace_point_tensor, visibility_filter, output.weight_accumulate)

            # Perform densification and pruning
            if iter > self.refracted_densify_from_iter and iter % self.refracted_densification_interval == 0:
                log(green_slim(f'Start updating refracted gaussians of step: {iter:06d}'))

                #TBD opacity 0으로 해서 densification에서 제거되게.
                if iter <= self.plane_offset_freeze_iter:
                    # plane_offset_freeze_iter 이전: 카메라 기준
                    self.refracted.zero_opacity_above_camera(
                        self.plane_normal, 
                        batch,
                        optimizer, 
                        prefix='sampler.refracted.'
                    )
                else:
                    # plane_offset_freeze_iter 이후: 평면 기준
                    self.refracted.zero_opacity_above_plane(
                        self.plane_normal, 
                        self._plane_offset, 
                        optimizer, 
                        prefix='sampler.refracted.'
                    )
                # Iteration-related densification and pruning parameters
                refracted_split_screen_threshold = self.refracted_split_screen_threshold if iter < self.refracted_screen_until_iter else None
                refracted_max_screen_threshold = self.refracted_max_screen_threshold if iter > self.refracted_opacity_reset_interval else None
                # Perform actual densification and pruning
                refracted.densify_and_prune(
                    self.refracted_min_opacity,
                    self.refracted_min_gradient,
                    self.refracted_densify_grad_threshold,
                    self.refracted_densify_size_threshold,
                    refracted_split_screen_threshold,
                    self.refracted_max_scene_threshold,
                    refracted_max_screen_threshold,
                    self.refracted_min_weight_threshold,
                    self.refracted_prune_visibility,
                    optimizer,
                    self.refracted_prune_large_gs,
                    prefix='sampler.refracted.'
                )

                log(green_slim('Densification and pruning done! ' +
                                f'min opacity: {refracted.get_opacity.min().item():.4f} ' +
                                f'max opacity: {refracted.get_opacity.max().item():.4f} ' +
                                f'number of points: {refracted.get_xyz.shape[0]}'))

            # Perform opacity reset
            if iter % self.refracted_opacity_reset_interval == 0:
                refracted.reset_opacity(optimizer=optimizer, prefix='sampler.refracted.')
                log(green_slim('Resetting opacity done! ' +
                                f'min opacity: {refracted.get_opacity.min().item():.4f} ' +
                                f'max opacity: {refracted.get_opacity.max().item():.4f}'))


        # if iter % 100 == 0:
        #     refracted: GaussianModel = self.refracted
        #     print("Only Prune")
        #     refracted.densify_and_prune(
        #         min_opacity=0.05,  # 원하는 opacity 임계값 설정
        #         min_gradient=None,
        #         densify_grad_threshold=float('inf'),
        #         densify_size_threshold=0.0,
        #         split_screen_threshold=None,  
        #         max_scene_threshold=None, 
        #         max_screen_threshold=None, 
        #         min_weight_threshold=None, 
        #         prune_visibility=False,
        #         prune_large_gs=False, 
        #         optimizer=optimizer,
        #         prefix='sampler.refracted.'
        #     )


    def store_gaussian_output(self, output: dotdict, batch: dotdict):
        # Post-process mainly does two things: add the batch dimension and reshape to the desired (B, P, C) shape
        # Visualization and supervision results processing
        output.acc_map       = output.rend_alpha[None ].permute(0, 2, 3, 1).reshape(1, -1, output.rend_alpha.shape[0] )  # (B, H * W, 1)
        output.dpt_map       = output.surf_depth[None ].permute(0, 2, 3, 1).reshape(1, -1, output.surf_depth.shape[0] )  # (B, H * W, 1)
        output.norm_map      = output.rend_normal[None].permute(0, 2, 3, 1).reshape(1, -1, output.rend_normal.shape[0])  # (B, H * W, 3)
        # Supervision results processing
        output.dist_map      = output.rend_dist[None  ].permute(0, 2, 3, 1).reshape(1, -1, output.rend_dist.shape[0]  )  # (B, H * W, 3)
        output.surf_norm_map = output.surf_normal[None].permute(0, 2, 3, 1).reshape(1, -1, output.surf_normal.shape[0])  
        output.bg_color      = torch.full_like(output.norm_map, self.bg_brightness)  # only for training and comparing with gt

        # Reflection & Refraction related processing 
        if self.render_reflection and 'specular' in output:
            output.spec_map  = output.specular[None].permute(0, 2, 3, 1).reshape(1, -1, output.specular.shape[0])  # (B, H * W, 3)
        if self.render_refraction and 'refraction' in output: #TBD
            output.refraction_map = output.refraction[None].permute(0, 2, 3, 1).reshape(1, -1, output.refraction.shape[0])  # (B, H * W, 1)

        # RGB color related processing
        rgb = output.render[None].permute(0, 2, 3, 1).reshape(1, -1, output.render.shape[0])  # (B, H * W, 3)
        output.rgb_map = rgb

        # Don't forget the iteration number for later supervision retrieval
        output.iter = batch.meta.iter
        return output



    def store_dif_gaussian_output(self, middle: dotdict, batch: dotdict, 
                                ray_o: torch.Tensor = None, ray_d: torch.Tensor = None):
        middle = self.store_gaussian_output(middle, batch)

        output = dotdict()
        # Store the output for supervision and visualization
        output.acc_map       = middle.acc_map         # (B, P, 1)
        output.dpt_map       = middle.dpt_map         # (B, P, 1)
        output.norm_map      = middle.norm_map        # (B, P, 3)
        output.dist_map      = middle.dist_map        # (B, P, 1)
        output.surf_norm_map = middle.surf_norm_map   # (B, P, 3)
        output.bg_color      = torch.full_like(output.norm_map, self.bg_brightness)
        
        # Reflectance related outputs
        if self.render_reflection and 'specular' in middle:
            output.spec_map  = middle.spec_map        # (B, P, 1)
        
        # Refraction related outputs with upper region processing
        if self.render_refraction and 'refraction' in middle:
            output.refraction_map = middle.refraction_map     # (B, P, 1)
        
        # The diffuse RGB output
        output.dif_rgb_map   = middle.rgb_map.clone()  # (B, P, 3)
        output.rgb_map       = middle.rgb_map          # (B, P, 3)

        # Don't forget the iteration number for later supervision retrieval
        output.iter = batch.meta.iter
        return output

    def get_reflect_rays(self, ray_o: torch.Tensor, ray_d: torch.Tensor, coords: torch.Tensor,
                         output: dotdict, batch: dotdict):
        # Compute the reflected rays direction, -d+d' = -2(d·n)n -> d' = d - 2(d·n)n
        norm = normalize(output.norm_map)  # (B, P, 3)
        ref_d = ray_d - 2 * torch.sum(ray_d * norm, dim=-1, keepdim=True) * norm  # (B, P, 3)

        # Compute the surface coordinate as the intersection point
        ref_o = ray_o + ray_d * output.dpt_map  # (B, P, 3)

        # Store the reflected rays for later supervision
        output.ref_o = ref_o  # (B, P, 3)
        output.ref_d = ref_d  # (B, P, 3)

        H, W = batch.meta.H[0].item(), batch.meta.W[0].item()

        # This branch is for compatibility with the original code
        ref_o = ref_o.reshape(H, W, 3)  # (H, W, 3)
        ref_d = ref_d.reshape(H, W, 3)  # (H, W, 3)

        if self.detach: return ref_o.detach(), ref_d.detach()
        else: return ref_o, ref_d


    def get_refract_rays(self, ray_o: torch.Tensor, ray_d: torch.Tensor, coords: torch.Tensor,
                        output: dotdict, batch: dotdict, ior_in: float = 1.000293, ior_out: float = 1.333): #TBD

        assert ior_out >= ior_in, "Total Inter Reflection is not implemented: require ior_in >= ior_out"
        norm = normalize(output.norm_map)  # (B, P, 3)
        ray_d_norm = normalize(ray_d)  # (B, P, 3) For refraction, we need to normalize the ray direction
        
        # cos(theta_i) = -dot(ray_d, norm)
        cos_i = -torch.sum(ray_d_norm * norm, dim=-1, keepdim=True)  # (B, P, 1)

        eta = ior_in / ior_out # η = ior_in / ior_out
        sin_t2 = eta**2 * (1 - cos_i**2) # sin²(theta_t) = η² * (1 - cos²(theta_i))
        cos_t = torch.sqrt(torch.clamp(1 - sin_t2, min=0.0)) # cos(theta_t) = sqrt( max(0, 1 - sin_t2) )
        
        # 굴절 광선 방향 계산 (Snell의 법칙):
        refract_d = eta * ray_d_norm + (eta * cos_i - cos_t) * norm  # d_t = η * ray_d + (η * cos_i - cos_t) * norm        ,(B, P, 3)
        
        refract_o = ray_o + ray_d * output.dpt_map  # (B, P, 3)

        # 결과를 output에 저장 (반후 supervision 등에서 사용)
        output.refr_o = refract_o
        output.refr_d = refract_d

        H, W = batch.meta.H[0].item(), batch.meta.W[0].item()

        refract_o = refract_o.reshape(H, W, 3)
        refract_d = refract_d.reshape(H, W, 3)

        if self.detach: return refract_o.detach(), refract_d.detach()
        else: return refract_o, refract_d



    def store_env_gaussian_output(self, middle: dotdict, output: dotdict, batch: dotdict):
        # Reshape and permute the middle output
        middle = self.store_gaussian_output(middle, batch)

        # Update the RGB output with the reflection
        output.rgb_map = (1 - output.spec_map) * output.rgb_map + output.spec_map * middle.rgb_map
        output.ref_rgb_map = middle.rgb_map  # (B, P, 3)

        # Store the environment Gaussian output for supervision
        output.env_opacity = self.env.get_opacity  # (P, 1)
        return output





######################################################################################
    def get_refract_rays_plane(self, ray_o: torch.Tensor, ray_d: torch.Tensor, coords: torch.Tensor,
                               output: dotdict, batch: dotdict, ior_in: float = 1.000293, ior_out: float = 1.333): #TBD
        assert ior_out >= ior_in, "Total Inter Reflection is not implemented: require ior_in >= ior_out"
        # Use plane normal instead of output.norm_map; normalize it to unit length.
        n = normalize(self.plane_normal)  # (3,)
        ray_d_norm = normalize(ray_d)

        # cos(theta_i) = -dot(ray_d, n)
        cos_i = -torch.sum(ray_d_norm * n, dim=-1, keepdim=True)  # (B, P, 1)

        eta = ior_in / ior_out  # η = ior_in / ior_out
        sin_t2 = eta**2 * (1 - cos_i**2)  # sin²(theta_t)
        cos_t = torch.sqrt(torch.clamp(1 - sin_t2, min=0.0))  # cos(theta_t)
        
        # 굴절 광선 방향 계산 (Snell의 법칙):
        refract_d = eta * ray_d_norm + (eta * cos_i - cos_t) * n  # (B, P, 3)
        
        # 평면 깊이 맵(output.plane_dpt_map)을 사용하여 평면 상의 교차점을 계산
        #refract_o = ray_o + ray_d * output.plane_dpt_map  # (B, P, 3)
        
        # 유효하지 않은 경우 ray 방향을 0으로 만들어 무효화 TBD
        valid_intersection = output.plane_dpt_map > 1e-6  # (B, P, 1)
        refract_d = torch.where(valid_intersection, refract_d, torch.zeros_like(refract_d))
        
        # 굴절 시작점
        refract_o = torch.where(
            valid_intersection,
            ray_o + ray_d * output.plane_dpt_map,  # 평면 교점에서 시작
            ray_o  # 무효한 경우 원점에서 시작 (방향이 0이므로 문제없음)
       )


        # 결과를 output에 저장 (반후 supervision 등에서 사용)
        output.refr_o = refract_o
        output.refr_d = refract_d

        H, W = batch.meta.H[0].item(), batch.meta.W[0].item()

        refract_o = refract_o.reshape(H, W, 3)
        refract_d = refract_d.reshape(H, W, 3)
        
        if self.detach: return refract_o.detach(), refract_d.detach()
        else: return refract_o, refract_d

    def compute_plane_depth_map(self,
                                ray_o: torch.Tensor,  # (B, P, 3)
                                ray_d: torch.Tensor   # (B, P, 3)
                                ) -> torch.Tensor:     # returns (B, P, 1)
        # unit-norm
        n = normalize(self.plane_normal)         # (3,)
        # P = self.plane_point.view(1, 1, 3)     # Old way using 3D point
        
        # New way: plane equation n_x*x + n_y*y + n_z*z = plane_offset (if n is [0,1,0], then y = plane_offset)
        # t = (plane_offset - ray_o_dot_n) / ray_d_dot_n
        # P_dot_n represents the constant in the plane equation n·X = C.
        # If n = [0,1,0], and plane is y = d, then C = d. self.plane_offset stores this 'd'.
        # If n is a general unit vector, P_on_plane = self.plane_offset * n. Then P_on_plane.dot(n) = self.plane_offset.
        p_dot_n = self._plane_offset 

        ray_o_dot_n = (ray_o * n).sum(-1, keepdim=True) # (B,P,1)
        denom = (ray_d * n).sum(-1, keepdim=True)       # (B,P,1)
        
        t = (p_dot_n - ray_o_dot_n) / (denom + 1e-8)
        
        return t.clamp(min=0)                             # (B,P,1)

    def get_reflect_rays_plane(self, ray_o: torch.Tensor, ray_d: torch.Tensor, coords: torch.Tensor,
                          output: dotdict, batch: dotdict):
        """평면에 대한 반사 광선 계산"""
        # 평면 법선 사용
        n = normalize(self.plane_normal)  # (3,)
        ray_d_norm = normalize(ray_d)     # (B, P, 3)
        
        # 평면에서의 반사: d' = d - 2(d·n)n
        ref_d_plane = ray_d_norm - 2 * torch.sum(ray_d_norm * n, dim=-1, keepdim=True) * n  # (B, P, 3)
        
        # 평면과의 교점을 반사 시작점으로 사용
        #ref_o_plane = ray_o + ray_d * output.plane_dpt_map  # (B, P, 3)
        
        # 유효하지 않은 경우 ray 방향을 0으로 만들어 무효화 TBD
        valid_intersection = output.plane_dpt_map > 1e-6  # (B, P, 1)
        ref_d_plane = torch.where(valid_intersection, ref_d_plane, torch.zeros_like(ref_d_plane))
        
        # 반사 시작점
        ref_o_plane = torch.where(
            valid_intersection,
            ray_o + ray_d * output.plane_dpt_map,
            ray_o  # 무효한 경우 원점에서 시작 (방향이 0이므로 문제없음)
        )



        # 결과 저장
        output.ref_o_plane = ref_o_plane
        output.ref_d_plane = ref_d_plane

        H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
        ref_o_plane = ref_o_plane.reshape(H, W, 3)
        ref_d_plane = ref_d_plane.reshape(H, W, 3)
        
        if self.detach: 
            return ref_o_plane.detach(), ref_d_plane.detach()
        else: 
            return ref_o_plane, ref_d_plane


######################################################################################
    
    def store_env_refracted_gaussian_output(self, env_middle: dotdict, env_plane_middle: dotdict, refracted_middle: dotdict, 
                                        output: dotdict, batch: dotdict, ray_o: torch.Tensor, ray_d: torch.Tensor):
            # Reshape and permute the middle output
            env_middle = self.store_gaussian_output(env_middle, batch)
            env_plane_middle = self.store_gaussian_output(env_plane_middle, batch)
            refracted_middle = self.store_gaussian_output(refracted_middle, batch)
            
            
            # Fresnel 계수 계산 - ray_o, ray_d를 파라미터로 전달받도록 수정 예정
            #fresnel_coef = self.compute_fresnel_coefficient(ray_o, ray_d, output, ior_out=1.5) # (B, P, 1) #no material info, fix ior_out=1.5
            fresnel_coef_plane = self.compute_fresnel_coefficient_plane(ray_o, ray_d) # (B, P, 1)
            
            #C_in = F * C_refl + (1 - F) * C_refr 
            water_image = fresnel_coef_plane * env_plane_middle.rgb_map + (1 - fresnel_coef_plane) * refracted_middle.rgb_map  # (B, P, 3)            
            #water_image = refracted_middle.rgb_map
            # water_image = (1 - fresnel_coef_plane) * refracted_middle.rgb_map  # (B, P, 3) - for debugging. remove water. not used now

            #C_out = C_base + F * C_refl
            #out_image = output.rgb_map + fresnel_coef * env_middle.rgb_map  # (B, P, 3) why not (1-F)*C_base + F*C_refl?
            #out_image = (1 - fresnel_coef) * output.rgb_map + fresnel_coef * env_middle.rgb_map
            out_image = output.rgb_map #TBD
            #C_out Specular version 
            #out_image = (1 - output.spec_map) * output.rgb_map + output.spec_map * env_middle.rgb_map  # (B, P, 3)

            #C_final = (1 - t) * C_in + t * C_out
            final_image = (1 - output.refraction_map) * water_image + output.refraction_map * out_image  # (B, P, 3)

            output.rgb_map = final_image  # (B, P, 3)

            # 디버깅용 출력들
            output.ref_rgb_map = env_middle.rgb_map
            output.ref_plane_rgb_map = env_plane_middle.rgb_map
            output.refracted_rgb_map = refracted_middle.rgb_map
            #output.fresnel_coef = fresnel_coef
            output.fresnel_coef_plane = fresnel_coef_plane
            #output.water_tint = water_tint
            
            # Store the environment&refracted Gaussian output for supervision
            output.env_opacity = self.env.get_opacity
            output.refracted_opacity = self.refracted.get_opacity
            
            return output


    def ablation_no_soft_mask(self, env_middle: dotdict, env_plane_middle: dotdict, refracted_middle: dotdict, 
                                        output: dotdict, batch: dotdict, ray_o: torch.Tensor, ray_d: torch.Tensor):
            # Reshape and permute the middle output
            env_middle = self.store_gaussian_output(env_middle, batch)
            env_plane_middle = self.store_gaussian_output(env_plane_middle, batch)
            refracted_middle = self.store_gaussian_output(refracted_middle, batch)
            
            #fresnel_coef = self.compute_fresnel_coefficient(ray_o, ray_d, output, ior_out=1.5) # (B, P, 1) #no material info, fix ior_out=1.5
            fresnel_coef_plane = self.compute_fresnel_coefficient_plane(ray_o, ray_d) # (B, P, 1)
            
            #C_in = F * C_refl + (1 - F) * C_refr 
            water_image = fresnel_coef_plane * env_plane_middle.rgb_map + (1 - fresnel_coef_plane) * refracted_middle.rgb_map  # (B, P, 3)            
            #water_image = refracted_middle.rgb_map
            # water_image = (1 - fresnel_coef_plane) * refracted_middle.rgb_map  # (B, P, 3) - for debugging. remove water

            out_image = output.rgb_map #TBD

            final_image = water_image + out_image  # (B, P, 3)

            output.rgb_map = final_image  # (B, P, 3)

            # 디버깅용 출력들
            output.ref_rgb_map = env_middle.rgb_map
            output.ref_plane_rgb_map = env_plane_middle.rgb_map
            output.refracted_rgb_map = refracted_middle.rgb_map
            #output.fresnel_coef = fresnel_coef
            output.fresnel_coef_plane = fresnel_coef_plane
            
            # Store the environment&refracted Gaussian output for supervision
            output.env_opacity = self.env.get_opacity
            output.refracted_opacity = self.refracted.get_opacity
            
            return output
#######################################################################################

    def compute_fresnel_coefficient(self, ray_o: torch.Tensor, ray_d: torch.Tensor, 
                                output: dotdict, ior_in: float = 1.000293, ior_out: float = 1.333) -> torch.Tensor:
        """Fresnel 방정식으로 반사/굴절 비율 계산"""
        norm = normalize(output.norm_map)  # (B, P, 3) - output 파라미터 추가 필요
        ray_d_norm = normalize(ray_d)  # (B, P, 3)
        
        # 입사각 계산
        cos_i = torch.abs(torch.sum(ray_d_norm * norm, dim=-1, keepdim=True))  # (B, P, 1)
        cos_i = torch.clamp(cos_i, 0.0, 1.0)
        
        # Schlick approximation
        F0 = ((ior_in - ior_out) / (ior_in + ior_out)) ** 2  # 약 0.02 for air-water
        fresnel = F0 + (1 - F0) * (1 - cos_i) ** 5
        
        return fresnel


    def compute_fresnel_coefficient_plane(self, ray_o: torch.Tensor, ray_d: torch.Tensor, 
                                ior_in: float = 1.000293, ior_out: float = 1.333) -> torch.Tensor: # 1.000293 1.333
        """Fresnel 방정식으로 반사/굴절 비율 계산"""
        n = normalize(self.plane_normal)  # (3,)
        ray_d_norm = normalize(ray_d)  # (B, P, 3)
        
        # 입사각 계산
        cos_i = torch.abs(torch.sum(ray_d_norm * n, dim=-1, keepdim=True))  # (B, P, 1)
        cos_i = torch.clamp(cos_i, 0.0, 1.0)
        
        # Schlick approximation
        F0 = ((ior_in - ior_out) / (ior_in + ior_out)) ** 2  # 약 0.02 for air-water
        fresnel = F0 + (1 - F0) * (1 - cos_i) ** 5
        
        return fresnel

######################################################################################

    def save_refracted_gaussians_to_ply(self, ply_file: str):
        """
        Save the refracted Gaussian xyz to a .ply file with their actual colors.
        Gaussians with opacity <= 0.05 are colored red.
        """
        xyz = self.refracted.get_xyz.detach().cpu().numpy()  # Detach and convert to numpy array
        
        # 가우시안의 opacity 정보 가져오기
        opacity = self.refracted.get_opacity.detach().cpu().numpy()  # (N, 1)
        
        # 가우시안의 실제 색상 정보 가져오기
        # SH coefficients에서 색상 추출 (DC component만 사용)
        features_dc = self.refracted.get_features[:, :1, :].detach().cpu().numpy()  # (N, 1, 3)
        
        # SH DC component를 RGB로 변환
        # DC component는 SH의 첫 번째 계수이므로 0.28209479177387814를 곱해줍니다
        SH_C0 = 0.28209479177387814
        rgb = features_dc[:, 0, :] * SH_C0 + 0.5  # (N, 3)
        
        # RGB 값을 [0, 1] 범위로 클램핑
        rgb = np.clip(rgb, 0.0, 1.0)
        
        # [0, 255] 범위로 변환
        colors = (rgb * 255).astype(np.uint8)  # (N, 3)
        
        # opacity가 0.05 이하인 가우시안을 빨간색으로 칠하기
        low_opacity_mask = opacity[:, 0] <= 0.05
        colors[low_opacity_mask] = [255, 0, 0]  # 빨간색 [R, G, B]
        
        # 통계 정보
        num_low_opacity = np.sum(low_opacity_mask)
        total_points = len(xyz)
        
        save_sfm_ply(ply_file, xyz, colors)
        log(green(f"Saved refracted Gaussians to {ply_file} with their actual colors."))
        log(green(f"Total points: {total_points}, Low opacity (≤0.05): {num_low_opacity} ({num_low_opacity/total_points*100:.1f}%)"))

    def save_pcd_gaussians_to_ply(self, ply_file: str):
        """
        Save the pcd Gaussian xyz to a .ply file with their actual colors.
        Gaussians with opacity <= 0.05 are colored red.
        """
        xyz = self.pcd.get_xyz.detach().cpu().numpy()  # Detach and convert to numpy array
        
        # 가우시안의 opacity 정보 가져오기
        opacity = self.pcd.get_opacity.detach().cpu().numpy()  # (N, 1)
        
        # 가우시안의 실제 색상 정보 가져오기
        # SH coefficients에서 색상 추출 (DC component만 사용)
        features_dc = self.pcd.get_features[:, :1, :].detach().cpu().numpy()  # (N, 1, 3)
        
        # SH DC component를 RGB로 변환
        # DC component는 SH의 첫 번째 계수이므로 0.28209479177387814를 곱해줍니다
        SH_C0 = 0.28209479177387814
        rgb = features_dc[:, 0, :] * SH_C0 + 0.5  # (N, 3)
        
        # RGB 값을 [0, 1] 범위로 클램핑
        rgb = np.clip(rgb, 0.0, 1.0)
        
        # [0, 255] 범위로 변환
        colors = (rgb * 255).astype(np.uint8)  # (N, 3)
        
        # opacity가 0.05 이하인 가우시안을 빨간색으로 칠하기
        low_opacity_mask = opacity[:, 0] <= 0.01
        colors[low_opacity_mask] = [255, 0, 0]  # 빨간색 [R, G, B]
        
        # 통계 정보
        num_low_opacity = np.sum(low_opacity_mask)
        total_points = len(xyz)
        
        save_sfm_ply(ply_file, xyz, colors)
        log(yellow(f"Saved pcd Gaussians to {ply_file} with their actual colors."))
        log(yellow(f"Total points: {total_points}, Low opacity (≤0.05): {num_low_opacity} ({num_low_opacity/total_points*100:.1f}%)"))



    def forward(self, batch: dotdict):
        import time
        start_time = time.time()
        step_times = {}
        
        # Update diffuse Gaussians: densification & pruning
        step_start = time.time()
        self.update_dif_gaussians(batch)
        step_times['update_dif_gaussians'] = time.time() - step_start
        
        # Update environment Gaussians: densification & pruning
        step_start = time.time()
        self.update_env_gaussians(batch)
        step_times['update_env_gaussians'] = time.time() - step_start
        
        # Update refracted Gaussians: densification & pruning
        step_start = time.time()
        self.update_refracted_gaussians(batch)  #TBD
        step_times['update_refracted_gaussians'] = time.time() - step_start

        # Prepare the camera transformation for Gaussian
        step_start = time.time()
        viewpoint_camera = to_x(prepare_gaussian_camera(batch), torch.float)
        step_times['prepare_camera'] = time.time() - step_start

        # Compute the camera ray origins and directions, and reflected rays
        step_start = time.time()
        ray_o, ray_d, coords, _, _, _ = self.get_camera_rays(
            batch,
            n_rays=self.n_rays,
            patch_size=self.patch_size
        )
        step_times['get_camera_rays'] = time.time() - step_start

        #batch.ray_o = ray_o #for camera upper refraction filtering
        #print(torch.sum(ray_d**2, dim=-1)) #TBD: ray_d가 정규화되어 있는지 확인하기 위한 디버깅 코드
        # Shape things
        H, W = batch.meta.H[0].item(), batch.meta.W[0].item()

        # Invoke hardware ray tracer
        step_start = time.time()
        if self.use_base_tracing:
            if self.tracing_backend == 'cpp':
                dif_output = self.diffop.render_gaussians(
                    viewpoint_camera,
                    ray_o.reshape(H, W, 3),
                    ray_d.reshape(H, W, 3),
                    self.pcd,
                    self.pipe,
                    self.bg_color,
                    0,
                    self.specular_threshold,
                    scaling_modifier=self.scale_mod,
                    override_color=None,
                    batch=batch
                )
            else:
                raise ValueError(f'Unknown tracing backend: {self.tracing_backend}')
        # Rasterize diffuse Gaussians to image, obtain their radii (on screen)
        else:
            dif_output = self.render_gaussians(
                viewpoint_camera,
                self.pcd,
                self.pipe,
                self.bg_color,
                self.scale_mod,
                override_color=None
            )
        step_times['dif_rendering'] = time.time() - step_start

        # Retain diffuse Gaussian gradients after updates
        # Skip saving the output if not in training mode to avoid unexpected densification skipping caused by `None` gradient
        if self.training: self.last_output = dif_output
        
        # Prepare output for supervision and visualization
        step_start = time.time()
        output = self.store_dif_gaussian_output(dif_output, batch, ray_o, ray_d) #TBD
        step_times['store_dif_output'] = time.time() - step_start
        
        #Add plane depth map to the output #TBD
        step_start = time.time()
        plane_dpt_map = self.compute_plane_depth_map(ray_o, ray_d) # (B, P, 1)
        output.plane_dpt_map = plane_dpt_map # (B, P, 1)
        step_times['compute_plane_depth'] = time.time() - step_start

        #Reflection Gaussian Part
        if batch.meta.iter >= self.render_reflection_start_iter:
            # Compute the reflected rays origins and directions
            step_start = time.time()
            ref_o, ref_d = self.get_reflect_rays(ray_o, ray_d, coords, output, batch)
            step_times['get_reflect_rays'] = time.time() - step_start
            
            #Invoke hardware ray tracer
            step_start = time.time()
            if self.tracing_backend == 'cpp':
                env_output = self.diffop.render_gaussians(
                    viewpoint_camera,
                    ref_o,
                    ref_d,
                    self.env,
                    self.pipe_env,
                    self.env_bg_color,
                    0,
                    start_from_first=False,
                    scaling_modifier=self.scale_mod,
                    override_color=None,
                    batch=batch
                )
            step_times['env_rendering'] = time.time() - step_start

            step_start = time.time()
            ref_o_plane, ref_d_plane = self.get_reflect_rays_plane(ray_o, ray_d, coords, output, batch)
            step_times['get_reflect_rays_plane'] = time.time() - step_start
            
            # Invoke hardware ray tracer for plane-based reflection
            step_start = time.time()
            if self.tracing_backend == 'cpp':
                env_output_plane = self.diffop.render_gaussians(
                    viewpoint_camera,
                    ref_o_plane,
                    ref_d_plane,
                    self.env,
                    self.pipe_env,
                    self.env_bg_color,
                    0,
                    start_from_first=False,
                    scaling_modifier=self.scale_mod,
                    override_color=None,
                    batch=batch
                )
            else:
                raise ValueError(f'Unknown tracing backend: {self.tracing_backend}')
            step_times['env_plane_rendering'] = time.time() - step_start

            # Retain gradients after updates
            # Skip saving the output if not in training mode to avoid unexpected densification skipping caused by `None` gradient
            if self.training: 
                self.last_output_env = env_output
                self.last_output_env_plane = env_output_plane

            # Prepare output for supervision and visualization. Use reflection only output before the refraction starts
            if batch.meta.iter >= self.render_reflection_start_iter and batch.meta.iter < self.render_refraction_start_iter:
                step_start = time.time()
                output = self.store_env_gaussian_output(env_output, output, batch)
                step_times['store_env_output'] = time.time() - step_start

        # #Refracted Gaussian Part #TBD. 
        if batch.meta.iter >= self.render_refraction_start_iter:
            # Compute the reflected rays origins and directions
            step_start = time.time()
            refr_o, refr_d = self.get_refract_rays_plane(ray_o, ray_d, coords, output, batch)
            step_times['get_refract_rays'] = time.time() - step_start
            
            # Invoke hardware ray tracer
            step_start = time.time()
            if self.tracing_backend == 'cpp':
                refracted_output = self.diffop_refracted.render_gaussians(
                    viewpoint_camera,
                    refr_o,
                    refr_d,
                    self.refracted,
                    self.pipe_refracted,
                    self.refracted_bg_color,
                    0,
                    start_from_first=False,
                    scaling_modifier=self.scale_mod,
                    override_color=None,
                    batch=batch
                )
            else:
                raise ValueError(f'Unknown tracing backend: {self.tracing_backend}')
            step_times['refracted_rendering'] = time.time() - step_start

            # Retain gradients after updates
            # Skip saving the output if not in training mode to avoid unexpected densification skipping caused by `None` gradient
            if self.training: self.last_output_refracted = refracted_output

            # Prepare output for supervision and visualization. Use reflection&refraction output after the refraction starts
            if batch.meta.iter >= self.render_reflection_start_iter and batch.meta.iter >= self.render_refraction_start_iter:
                step_start = time.time()
                output = self.store_env_refracted_gaussian_output(env_output, env_output_plane, refracted_output, output, batch, ray_o, ray_d) 
                #output = self.ablation_no_soft_mask(env_output, env_output_plane, refracted_output, output, batch, ray_o, ray_d) #ABLATION
                step_times['store_env_refracted_output'] = time.time() - step_start

        # Final output preparation
        step_start = time.time()
        if batch.meta.iter % 10 == 0:
            print(self._plane_offset)
            #print(self.plane_normal) #TBD: 디버깅용 출력
            #print(ray_o[0][0][1])
        
        # Update the output to the batch
        # 물 후보 영역 검출 및 depth_map 업데이트 TBD
        if hasattr(output, 'plane_dpt_map'):
            initial_mask = (output.dpt_map > output.plane_dpt_map).float()  # (B, P, 1)
            valid_plane_intersection = (output.plane_dpt_map > 1e-6).float()  # (B, P, 1)
            initial_mask = initial_mask * valid_plane_intersection
            # --- 기존 로직 끝 ---

            # --- 새로운 마스크 정제 로직 시작 (OpenCV 버전) ---
            B, P, _ = initial_mask.shape
            H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
            
            # 1. 마스크를 이미지 형태로 복원하고 CPU의 NumPy 배열로 변환
            #    (일반적으로 배치 크기 B=1이라고 가정)
            mask_tensor_cpu = initial_mask.reshape(B, H, W)[0].cpu()
            bw_mask = (mask_tensor_cpu.numpy() * 255).astype(np.uint8)

            # 2. OpenCV를 사용하여 연결 요소 라벨링
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw_mask, connectivity=8)

            # 3. 경계에 닿는 컴포넌트들의 레이블 수집
            border_labels = set()
            border_labels.update(labels[0, :])      # Top
            border_labels.update(labels[-1, :])     # Bottom
            border_labels.update(labels[:, 0])      # Left
            border_labels.update(labels[:, -1])     # Right

            # 4. 경계에 닿지 않는 "둘러싸인" 컴포넌트 중 가장 큰 것 찾기
            largest_enclosed_label = -1
            max_area = -1
            
            # 0번 레이블은 배경이므로 1부터 시작
            for i in range(1, num_labels):
                if i not in border_labels:
                    area = stats[i, cv2.CC_STAT_AREA]
                    if area > max_area:
                        max_area = area
                        largest_enclosed_label = i
            
            # 5. 가장 큰 컴포넌트만으로 최종 마스크 생성
            final_mask_np = np.zeros_like(bw_mask)
            if largest_enclosed_label != -1:
                final_mask_np[labels == largest_enclosed_label] = 255

            # 6. 후처리: Morphological Closing으로 작은 구멍 메우기
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
            final_mask_np = cv2.morphologyEx(final_mask_np, cv2.MORPH_CLOSE, kernel)

            # 7. 최종 마스크를 다시 GPU의 PyTorch 텐서로 변환하여 덮어쓰기
            final_mask_tensor = torch.from_numpy(final_mask_np).to(initial_mask.device).float() / 255.0
            water_candidate_mask = final_mask_tensor.reshape(B, H, W, 1).reshape(B, P, 1)
            # --- 새로운 마스크 정제 로직 끝 ---

            output.spec_map = water_candidate_mask

        batch.output.update(output)
        step_times['final_output_update'] = time.time() - step_start
        
        # 시간 측정 결과 출력 (매 100 iteration마다 또는 조건에 따라)
        total_time = time.time() - start_time
        step_times['total_forward'] = total_time
        
        # 시간 출력 조건 (필요에 따라 수정)
        if batch.meta.iter % 100 == 0 or total_time > 1.0:  # 100 iter마다 또는 1초 이상 걸릴 때
            print(f"\n[TIMING] Iteration {batch.meta.iter}: Total Forward Time: {total_time:.4f}s")
            print("=" * 60)
            
            # 시간순으로 정렬하여 출력
            sorted_times = sorted(step_times.items(), key=lambda x: x[1], reverse=True)
            for step_name, step_time in sorted_times:
                percentage = (step_time / total_time) * 100
                print(f"{step_name:25s}: {step_time:.4f}s ({percentage:5.1f}%)")
            print("=" * 60)
            
            # 가장 오래 걸리는 3개 단계 강조
            print(f"TOP 3 BOTTLENECKS:")
            for i, (step_name, step_time) in enumerate(sorted_times[:3]):
                percentage = (step_time / total_time) * 100
                print(f"  {i+1}. {step_name}: {step_time:.4f}s ({percentage:.1f}%)")
            print()
        if batch.meta.iter % 100 == 0:
            self.save_refracted_gaussians_to_ply(f"./refr_pcd/refracted_pcd_{batch.meta.iter}.ply")
            self.save_pcd_gaussians_to_ply(f"./base_pcd/base_pcd_{batch.meta.iter}.ply")
