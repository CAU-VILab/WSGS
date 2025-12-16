import torch
from torch import nn
from torch.nn import functional as F

from easyvolcap.engine import cfg
from easyvolcap.engine import SUPERVISORS
from easyvolcap.engine.registry import call_from_cfg
from easyvolcap.models.supervisors.volumetric_video_supervisor import VolumetricVideoSupervisor

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.math_utils import normalize
from easyvolcap.utils.data_utils import save_image
from easyvolcap.utils.depth_utils import normalize_depth
from easyvolcap.utils.loss_utils import l1, l2, l1_reg, l2_reg, cos, mIoU_loss, mse


@SUPERVISORS.register_module()
class WSGSSupervisor(VolumetricVideoSupervisor):
    def __init__(self,
                 network: nn.Module,

                 norm_loss_weight: float = 0.0,
                 norm_loss_weight_final: float = None,
                 norm_loss_start_iter: int = 7000,
                 norm_loss_until_iter: int = None,
                 use_acc_scale_norm_loss: bool = False,
                 use_dpt_scale_norm_loss: bool = False,
                 max_dpt_scale_percet: bool = False,
                 use_spec_scale_norm_loss: bool = False,
                 use_spec_scale_norm_loss_start_iter: int = 7000,
                 use_spec_scale_norm_loss_until_iter: int = None,

                 gs_norm_loss_weight: float = 0.0,
                 gs_norm_loss_weight_final: float = None,
                 gs_dist_loss_weight: float = 0.0,
                 gs_norm_loss_start_iter: int = 7000,
                 gs_norm_loss_until_iter: int = None,
                 use_acc_scale_gs_norm_loss: bool = False,
                 use_dpt_scale_gs_norm_loss: bool = False,
                 use_spec_scale_gs_norm_loss: bool = False,
                 use_spec_scale_gs_norm_loss_start_iter: int = 7000,
                 use_spec_scale_gs_norm_loss_until_iter: int = None,
                 gs_dist_loss_start_iter: int = 3000,
                 gs_dist_loss_until_iter: int = None,

                 env_opacity_loss_weight: float = 0.0,
                 env_opacity_loss_type: str = 'sparse',
                 env_opacity_loss_start_iter: int = 0,

                 # Mask mIoU loss
                 msk_loss_weight: float = 0.0,  # mask mIoU loss
                 msk_loss_start_iter: int = 7000,
                 msk_loss_until_iter: int = None,

                 # Normal smoothness loss
                 norm_smooth_loss_weight: float = 0.0,
                 norm_smooth_loss_start_iter: int = 7000,
                 norm_smooth_loss_until_iter: int = None,
                 use_edge_aware_smooth: bool = True,
                 use_dpt_scale_norm_smooth_loss: bool = True,

                 # Residual normal loss
                 res_norm_loss_weight: float = 0.001,

                 # Specular loss
                 specular_loss_weight: float = 0.0,
                 specular_loss_start_iter: int = 7000,
                 specular_loss_until_iter: int = 9000,
                 specular_target: float = 0.8,
                 min_specular_percent: float = 0.5,

                 # Reflection color loss
                 ref_rgb_loss_weight: float = 0.0,
                 ref_rgb_loss_start_iter: int = 7000,
                 ref_rgb_loss_until_iter: int = 9000,

                 # pseudo_mask loss
                 pseudo_mask_loss_weight: float = 0.0,
                 pseudo_mask_loss_start_iter: int = 3000,
                 pseudo_mask_loss_until_iter: int = None,
                 **kwargs,
                 ):
        call_from_cfg(super().__init__, kwargs, network=network)

        # Normal loss
        self.norm_loss_weight = norm_loss_weight
        self.norm_loss_weight_final = norm_loss_weight_final
        self.norm_loss_start_iter = norm_loss_start_iter
        self.norm_loss_until_iter = norm_loss_until_iter
        self.use_acc_scale_norm_loss = use_acc_scale_norm_loss
        self.use_dpt_scale_norm_loss = use_dpt_scale_norm_loss
        self.max_dpt_scale_percet = max_dpt_scale_percet
        self.use_spec_scale_norm_loss = use_spec_scale_norm_loss
        self.use_spec_scale_norm_loss_start_iter = use_spec_scale_norm_loss_start_iter
        self.use_spec_scale_norm_loss_until_iter = use_spec_scale_norm_loss_until_iter

        self.gs_norm_loss_weight = gs_norm_loss_weight
        self.gs_norm_loss_weight_final = gs_norm_loss_weight_final
        self.gs_dist_loss_weight = gs_dist_loss_weight
        self.gs_norm_loss_start_iter = gs_norm_loss_start_iter
        self.gs_norm_loss_until_iter = gs_norm_loss_until_iter
        self.use_acc_scale_gs_norm_loss = use_acc_scale_gs_norm_loss
        self.use_dpt_scale_gs_norm_loss = use_dpt_scale_gs_norm_loss
        self.use_spec_scale_gs_norm_loss = use_spec_scale_gs_norm_loss
        self.use_spec_scale_gs_norm_loss_start_iter = use_spec_scale_gs_norm_loss_start_iter
        self.use_spec_scale_gs_norm_loss_until_iter = use_spec_scale_gs_norm_loss_until_iter
        self.gs_dist_loss_start_iter = gs_dist_loss_start_iter
        self.gs_dist_loss_until_iter = gs_dist_loss_until_iter

        self.env_opacity_loss_weight = env_opacity_loss_weight
        self.env_opacity_loss_type = env_opacity_loss_type
        self.env_opacity_loss_start_iter = env_opacity_loss_start_iter

        # Mask mIoU loss
        self.msk_loss_weight = msk_loss_weight
        self.msk_loss_start_iter = msk_loss_start_iter
        self.msk_loss_until_iter = msk_loss_until_iter

        # Smooth loss
        self.norm_smooth_loss_weight = norm_smooth_loss_weight
        self.norm_smooth_loss_start_iter = norm_smooth_loss_start_iter
        self.norm_smooth_loss_until_iter = norm_smooth_loss_until_iter
        self.use_edge_aware_smooth = use_edge_aware_smooth
        self.use_dpt_scale_norm_smooth_loss = use_dpt_scale_norm_smooth_loss

        # Residual normal loss
        self.res_norm_loss_weight = res_norm_loss_weight

        # Specular loss
        self.specular_loss_weight = specular_loss_weight
        self.specular_loss_start_iter = specular_loss_start_iter
        self.specular_loss_until_iter = specular_loss_until_iter
        self.specular_target = specular_target
        self.min_specular_percent = min_specular_percent

        # Reflection color loss
        self.ref_rgb_loss_weight = ref_rgb_loss_weight
        self.ref_rgb_loss_start_iter = ref_rgb_loss_start_iter
        self.ref_rgb_loss_until_iter = ref_rgb_loss_until_iter

        # pseudo_mask loss parameters
        self.pseudo_mask_loss_weight = pseudo_mask_loss_weight
        self.pseudo_mask_loss_start_iter = pseudo_mask_loss_start_iter
        self.pseudo_mask_loss_until_iter = pseudo_mask_loss_until_iter

        # Compute the total number of iterations
        self.total_iter = cfg.runner_cfg.epochs * cfg.runner_cfg.ep_iter


    def reshape_output_to_image(self, output_map: torch.Tensor, batch: dotdict):
        """
        Reshape flattened output to image format
        Args:
            output_map: (B, P, 3) or (B, 1, P, 3) flattened pixels
            batch: batch containing meta information
        Returns:
            reshaped image: (B, H, W, 3)
        """
        # Get image dimensions
        if hasattr(batch, 'meta') and 'H' in batch.meta and 'W' in batch.meta:
            H, W = batch.meta.H[0].item(), batch.meta.W[0].item()
        elif hasattr(batch, 'H') and hasattr(batch, 'W'):
            H, W = batch.H, batch.W
        else:
            # Try to infer from output shape
            if output_map.dim() == 3:  # (B, P, 3)
                P = output_map.shape[1]
            elif output_map.dim() == 4 and output_map.shape[1] == 1:  # (B, 1, P, 3)
                P = output_map.shape[2]
            else:
                raise ValueError(f"Cannot handle output_map shape: {output_map.shape}")
            
            H = W = int(P ** 0.5)
            if H * W != P:
                raise ValueError(f"Cannot infer square H, W from P={P}")
        
        # Reshape based on input format
        if output_map.dim() == 3:  # (B, P, 3)
            B, P, C = output_map.shape
            return output_map.view(B, H, W, C)
        elif output_map.dim() == 4 and output_map.shape[1] == 1:  # (B, 1, P, 3)
            B, _, P, C = output_map.shape
            return output_map.view(B, H, W, C)
        else:
            return output_map


    def compute_loss(self, output: dotdict, batch: dotdict, loss: torch.Tensor, scalar_stats: dotdict, image_stats: dotdict):
        if 'env_opacity' in output and self.env_opacity_loss_weight > 0:
            if output.iter >= self.env_opacity_loss_start_iter:
                if self.env_opacity_loss_type == 'sparse':
                    epsilon = 1e-3
                    v = torch.clamp(output.env_opacity, epsilon, 1 - epsilon)
                    env_opacity_loss = torch.mean(torch.log(v) + torch.log(1 - v))
                elif self.env_opacity_loss_type == 'l1':
                    env_opacity_loss = l1_reg(1 - output.env_opacity)
                scalar_stats.env_opacity_loss = env_opacity_loss
                loss += self.env_opacity_loss_weight * env_opacity_loss

        if 'norm_map' in output and 'norm' in batch and self.norm_loss_weight > 0:
            if output.iter >= self.norm_loss_start_iter and \
                (self.norm_loss_until_iter is None or output.iter < self.norm_loss_until_iter):

                # Transform the normal map to the local coordinate system
                norm_map = normalize(output.norm_map)
                norm_map = norm_map @ batch.R.mT  # convert to view space
                norm_map = normalize(norm_map)

                # Process the ground truth normal map
                norm = batch.norm * 2. - 1.  # this is generally how normals are stored on disk
                norm = normalize(norm)

                # Compute normal loss
                norm_loss = (norm_map - norm).abs().sum(dim=-1)  # MARK: SYNC
                norm_loss += 1 - F.cosine_similarity(norm_map, norm, dim=-1)  # MARK: SYNC

                # Maybe scale the normal loss with acc_map
                if self.use_acc_scale_norm_loss:
                    scale_acc = output.acc_map[..., 0].detach().clone()
                    norm_loss = norm_loss * scale_acc
                # Maybe scale the normal loss with inverse normalized depth
                if self.use_dpt_scale_norm_loss:
                    if self.max_dpt_scale_percet:
                        # Exclude the points with large depth and zero depth
                        dpt_msk = output.dpt_map[..., 0].detach().clone() > 0
                        dpt_msk = torch.logical_and(dpt_msk, output.dpt_map[..., 0].detach().clone() <= torch.quantile(output.dpt_map[dpt_msk], self.max_dpt_scale_percet))
                        norm_loss[~dpt_msk] = 0
                    else:
                        # Scale by inverse normalized depth
                        scale_dpt = normalize_depth(output.dpt_map[..., 0].detach().clone())
                        norm_loss = norm_loss * scale_dpt

                norm_loss = norm_loss.mean()
                scalar_stats.norm_loss = norm_loss
                loss += self.norm_loss_weight * norm_loss #TBD disable monocular normal loss

        if 'norm_map' in output and 'surf_norm_map' in output and self.gs_norm_loss_weight > 0:
            # Compute the normal consistency loss after a certain iteration
            if output.iter >= self.gs_norm_loss_start_iter and \
                (self.gs_norm_loss_until_iter is None or output.iter < self.gs_norm_loss_until_iter):

                # Compute the normal consistency loss
                gs_norm_loss = 1 - (output.norm_map * output.surf_norm_map).sum(dim=-1)
                # Maybe scale the normal loss with acc_map
                if self.use_acc_scale_gs_norm_loss:
                    scale_acc = output.acc_map[..., 0].detach().clone()
                    gs_norm_loss = gs_norm_loss * scale_acc
                # Maybe scale the normal loss with inverse normalized depth
                if self.use_dpt_scale_gs_norm_loss:
                    if self.max_dpt_scale_percet:
                        # Exclude the points with large depth and zero depth
                        dpt_msk = output.dpt_map[..., 0].detach().clone() > 0
                        dpt_msk = torch.logical_and(dpt_msk, output.dpt_map[..., 0].detach().clone() <= torch.quantile(output.dpt_map[dpt_msk], self.max_dpt_scale_percet))
                        gs_norm_loss[~dpt_msk] = 0
                    else:
                        # Scale by inverse normalized depth
                        scale_dpt = normalize_depth(output.dpt_map[..., 0].detach().clone())
                        gs_norm_loss = gs_norm_loss * scale_dpt

                gs_norm_loss = gs_norm_loss.mean()
                scalar_stats.gs_norm_loss = gs_norm_loss
                loss += self.gs_norm_loss_weight * gs_norm_loss

        if 'acc_map' in output and self.msk_loss_weight > 0:
            # Get the mask
            if output.iter >= self.msk_loss_start_iter and \
              (self.msk_loss_until_iter is None or output.iter < self.msk_loss_until_iter):
                mask = torch.logical_and(batch.msk[..., 0] > 0.5, torch.norm(batch.norm, dim=-1) > 0.25)[..., None]  # (B, P, 1)
                msk_loss = mse(output.acc_map, mask)
                scalar_stats.msk_loss = msk_loss
                loss += self.msk_loss_weight * msk_loss

        if 'dist_map' in output and self.gs_dist_loss_weight > 0:
            # Compute the distance consistency loss after a certain iteration
            if output.iter >= self.gs_dist_loss_start_iter and \
                (self.gs_dist_loss_until_iter is None or output.iter < self.gs_dist_loss_until_iter):
                # Compute the distance consistency loss
                gs_dist_loss = output.dist_map.mean()

                # Log and add the loss
                scalar_stats.gs_dist_loss = gs_dist_loss
                loss += self.gs_dist_loss_weight * gs_dist_loss

        # pseudo_mask Loss - only for full image rendering
        if  'spec_map' in output and 'refraction_map' in output:
            if (self.pseudo_mask_loss_weight > 0 and 
                output.iter >= self.pseudo_mask_loss_start_iter and
                (self.pseudo_mask_loss_until_iter is None or output.iter < self.pseudo_mask_loss_until_iter)):
                
                spec_map_sum = output.spec_map.sum()
                if spec_map_sum > 0:  # 유효한 마스크인 경우에만 loss 계산
                    pseudo_gt = 1.0 - output.spec_map.detach()  # (B, P, 1)
                    prediction = output.refraction_map  # (B, P, 1)

                    # MSE 손실을 계산합니다.
                    spec_refr_loss = mse(prediction, pseudo_gt)
                    
                    # 통계 및 최종 손실에 추가합니다.
                    scalar_stats.spec_refr_loss = spec_refr_loss
                    loss += self.pseudo_mask_loss_weight * spec_refr_loss
                else:
                    # 잘못된 마스크인 경우 loss를 0으로 설정
                    spec_refr_loss = torch.tensor(0.0, device=output.spec_map.device)
                    scalar_stats.spec_refr_loss = spec_refr_loss
                    # loss에는 추가하지 않음 (0이므로 의미 없음)        
                

        return loss
