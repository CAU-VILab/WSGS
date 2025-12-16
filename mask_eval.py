import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from glob import glob
from typing import List, Dict, Any
import re
from PIL import Image

from easyvolcap.utils.console_utils import *
from easyvolcap.utils.base_utils import dotdict
from easyvolcap.utils.json_utils import serialize
from easyvolcap.utils.metric_utils import psnr, ssim, lpips
from easyvolcap.utils.loss_utils import mse as compute_mse
from easyvolcap.utils.data_utils import load_image


def load_image_tensor(image_path: str) -> torch.Tensor:
    """이미지를 torch tensor로 로드 (0-1 범위)"""
    img = load_image(image_path)  # 이미 0-1 범위로 정규화됨
    return torch.from_numpy(img).float()


def load_mask_tensor(mask_path: str) -> torch.Tensor:
    """마스크를 torch tensor로 로드 (0 또는 1)"""
    mask = Image.open(mask_path).convert('L')  # grayscale로 변환
    mask = np.array(mask) / 255.0  # 0-1 범위로 정규화
    mask = (mask > 0.5).astype(np.float32)  # 0.5 threshold로 binary mask
    return torch.from_numpy(mask)


def compute_masked_psnr_accurate(img: torch.Tensor, img_gt: torch.Tensor, mask: torch.Tensor) -> float:
    """마스크된 픽셀에서만 정확한 PSNR 계산 (pseudo_mask_eval.py 방식)"""
    # 유효한 픽셀만 선택
    valid_mask = mask > 0.5
    if valid_mask.sum() == 0:
        return float('inf')
    
    # 각 채널별로 유효한 픽셀만 추출
    img_valid = img[valid_mask.unsqueeze(-1).expand_as(img)].view(-1, img.shape[-1])
    img_gt_valid = img_gt[valid_mask.unsqueeze(-1).expand_as(img_gt)].view(-1, img_gt.shape[-1])
    
    # 코드베이스의 compute_mse 함수 사용
    mse = compute_mse(img_valid, img_gt_valid).mean()
    
    # 코드베이스 스타일의 PSNR 계산
    psnr_val = (1 / mse.clip(1e-10)).log() * 10 / np.log(10)
    return psnr_val.item()


def compute_masked_ssim_accurate(img: torch.Tensor, img_gt: torch.Tensor, mask: torch.Tensor) -> float:
    """마스킹 후 SSIM 계산 (pseudo_blendmask_eval.py 방식)"""
    # 마스크를 3채널로 확장
    if mask.ndim == 2:
        mask_3d = mask.unsqueeze(-1).expand_as(img)
    else:
        mask_3d = mask
    
    # 마스크 적용
    img_masked = img * mask_3d
    img_gt_masked = img_gt * mask_3d
    
    # 표준 SSIM 함수 사용
    return ssim(img_masked, img_gt_masked)


def compute_masked_lpips_accurate(img: torch.Tensor, img_gt: torch.Tensor, mask: torch.Tensor) -> float:
    """마스킹 후 LPIPS 계산 (pseudo_blendmask_eval.py 방식)"""
    # 마스크를 3채널로 확장
    if mask.ndim == 2:
        mask_3d = mask.unsqueeze(-1).expand_as(img)
    else:
        mask_3d = mask
    
    # 마스크 적용
    img_masked = img * mask_3d
    img_gt_masked = img_gt * mask_3d
    
    # 표준 LPIPS 함수 사용
    return lpips(img_masked, img_gt_masked)


def extract_camera_number_from_filename(filename: str) -> int:
    """파일명에서 camera 뒤의 숫자를 추출합니다"""
    # frame0000_camera0000.png 형식에서 camera 뒤의 숫자 추출
    pattern = r'camera(\d+)'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return 0  # camera 번호가 없으면 0 반환


def extract_frame_number_from_filename(filename: str) -> int:
    """파일명에서 frame 뒤의 숫자를 추출합니다"""
    # frame0000_camera0000.png 형식에서 frame 뒤의 숫자 추출
    pattern = r'frame(\d+)'
    match = re.search(pattern, filename)
    if match:
        return int(match.group(1))
    return 0  # frame 번호가 없으면 0 반환


class MaskedVideoEvaluator:
    def __init__(self,
                 data_root: str,
                 render_dir: str = 'RENDER',
                 specular_dir: str = 'SPECULAR',
                 compute_metrics: List[str] = ['PSNR', 'SSIM', 'LPIPS'],
                 odd_camera_only: bool = False,
                 **kwargs
                 ) -> None:
        self.data_root = data_root
        self.render_dir = os.path.join(data_root, render_dir)
        self.specular_dir = os.path.join(data_root, specular_dir)
        self.compute_metrics = compute_metrics
        self.odd_camera_only = odd_camera_only
        
        # 정확한 메트릭 계산 함수 매핑
        self.metric_functions = {
            'PSNR': compute_masked_psnr_accurate,    # pseudo_mask_eval.py에서 가져온 정확한 PSNR
            'SSIM': compute_masked_ssim_accurate,    # pseudo_blendmask_eval.py에서 가져온 정확한 SSIM
            'LPIPS': compute_masked_lpips_accurate   # pseudo_blendmask_eval.py에서 가져온 정확한 LPIPS
        }
        
        log(f"Initialized Combined MaskedVideoEvaluator")
        log(f"Render directory: {self.render_dir}")
        log(f"Specular directory: {self.specular_dir}")
        log(f"Odd camera only: {self.odd_camera_only}")
        log(f"Using accurate implementations: PSNR (mask-only), SSIM & LPIPS (masked)")

    def get_file_list(self) -> List[str]:
        """렌더링 이미지 파일 목록을 가져옵니다"""
        render_files = glob(os.path.join(self.render_dir, "*.png"))
        file_names = []
        
        for render_file in render_files:
            file_name = os.path.basename(render_file)
            # _error.png나 _gt.png가 아닌 파일만 선택
            if not file_name.endswith('_error.png') and not file_name.endswith('_gt.png'):
                base_name = file_name.replace('.png', '')
                
                # 홀수 카메라만 필터링
                if self.odd_camera_only:
                    camera_number = extract_camera_number_from_filename(base_name)
                    if camera_number % 2 == 0:  # 짝수면 스킵
                        continue
                
                file_names.append(base_name)
        
        return sorted(file_names)

    def evaluate_single_image(self, file_name: str) -> Dict[str, float]:
        """단일 이미지에 대한 메트릭 계산"""
        # 파일 경로 설정
        render_path = os.path.join(self.render_dir, f"{file_name}.png")
        gt_path = os.path.join(self.render_dir, f"{file_name}_gt.png")
        mask_path = os.path.join(self.specular_dir, f"{file_name}.png")
        
        # 파일 존재 확인
        if not os.path.exists(render_path):
            log(red(f"Render image not found: {render_path}"))
            return {}
        if not os.path.exists(gt_path):
            log(red(f"Ground truth image not found: {gt_path}"))
            return {}
        if not os.path.exists(mask_path):
            log(red(f"Mask image not found: {mask_path}"))
            return {}
        
        try:
            # 이미지 로드
            img = load_image_tensor(render_path)
            img_gt = load_image_tensor(gt_path)
            mask = load_mask_tensor(mask_path)
            
            # 이미지 크기 확인
            if img.shape[:2] != img_gt.shape[:2] or img.shape[:2] != mask.shape[:2]:
                log(red(f"Image size mismatch for {file_name}"))
                return {}
            
            # RGB 채널만 사용 (alpha 채널 제거)
            img = img[..., :3]
            img_gt = img_gt[..., :3]
            
            # 메트릭 계산 - 각각 정확한 구현 사용
            metrics = {}
            for metric_name in self.compute_metrics:
                if metric_name in self.metric_functions:
                    try:
                        value = self.metric_functions[metric_name](img, img_gt, mask)
                        metrics[metric_name] = value
                    except Exception as e:
                        log(red(f"Error computing {metric_name} for {file_name}: {e}"))
                        metrics[metric_name] = float('nan')
                else:
                    log(yellow(f"Unknown metric: {metric_name}"))
            
            # 마스크 정보 추가
            valid_pixels = (mask > 0.5).sum().item()
            total_pixels = mask.numel()
            metrics['mask_ratio'] = valid_pixels / total_pixels if total_pixels > 0 else 0.0
            
            # 프레임 및 카메라 번호 추가
            frame_number = extract_frame_number_from_filename(file_name)
            camera_number = extract_camera_number_from_filename(file_name)
            metrics['frame_number'] = frame_number
            metrics['camera_number'] = camera_number
            
            return metrics
            
        except Exception as e:
            log(red(f"Error processing {file_name}: {e}"))
            return {}

    def evaluate_all(self) -> dotdict:
        """모든 이미지에 대한 평가 수행"""
        file_names = self.get_file_list()
        if not file_names:
            log(red("No files found for evaluation"))
            return dotdict()
        
        filter_info = " (odd cameras only)" if self.odd_camera_only else ""
        log(f"Found {len(file_names)} files for evaluation{filter_info}")
        
        all_metrics = []
        individual_results = {}
        
        # 각 이미지에 대해 평가 수행
        for file_name in tqdm(file_names, desc="Evaluating images"):
            metrics = self.evaluate_single_image(file_name)
            if metrics:
                all_metrics.append(metrics)
                individual_results[file_name] = metrics
                frame_info = f" (frame {metrics.get('frame_number', 'N/A')}, camera {metrics.get('camera_number', 'N/A')})" if 'frame_number' in metrics else ""
                log(f"{file_name}{frame_info}: {metrics}")
        
        if not all_metrics:
            log(red("No valid metrics computed"))
            return dotdict()
        
        # 통계 계산
        summary = self.compute_summary(all_metrics)
        
        # 결과 반환
        results = dotdict({
            'summary': summary,
            'individual_results': individual_results,
            'total_images': len(file_names),
            'valid_images': len(all_metrics),
            'odd_camera_only': self.odd_camera_only
        })
        
        return results

    def compute_summary(self, all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """메트릭 통계 계산"""
        summary = {}
        
        # 각 메트릭에 대한 평균과 표준편차 계산
        for metric_name in self.compute_metrics + ['mask_ratio']:
            values = []
            for metrics in all_metrics:
                if metric_name in metrics and not np.isnan(metrics[metric_name]) and not np.isinf(metrics[metric_name]):
                    values.append(metrics[metric_name])
            
            if values:
                summary[f'{metric_name}_mean'] = np.mean(values)
                summary[f'{metric_name}_std'] = np.std(values)
                summary[f'{metric_name}_count'] = len(values)
            else:
                summary[f'{metric_name}_mean'] = float('nan')
                summary[f'{metric_name}_std'] = float('nan')
                summary[f'{metric_name}_count'] = 0
        
        # 카메라 개수만 추가
        if all_metrics:
            camera_numbers = [m.get('camera_number', 0) for m in all_metrics if 'camera_number' in m]
            if camera_numbers:
                summary['camera_count'] = len(set(camera_numbers))  # 유니크한 카메라 수
        
        return summary

    def run(self) -> dotdict:
        """평가 실행"""
        filter_info = " (odd cameras only)" if self.odd_camera_only else ""
        log(f"Starting combined masked video evaluation{filter_info}...")
        log(f"Using: PSNR (mask-only accurate), SSIM & LPIPS (masked accurate)")
        results = self.evaluate_all()
        
        if 'summary' in results:
            log("=== Combined Evaluation Summary ===")
            # 출력할 키들만 필터링
            display_keys = ['PSNR_mean', 'SSIM_mean', 'LPIPS_mean', 'camera_count']
            
            for key, value in results.summary.items():
                if key in display_keys:
                    if isinstance(value, float):
                        log(f"{key}: {value:.6f}")
                    else:
                        log(f"{key}: {value}")
        
        return results


# 사용 예시를 위한 스크립트
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Combined Masked Video Evaluation")
    parser.add_argument('--data_root', type=str, required=True, 
                        help='Path to the data directory containing RENDER and SPECULAR folders')
    parser.add_argument('--render_dir', type=str, default='RENDER',
                        help='Name of the render directory')
    parser.add_argument('--specular_dir', type=str, default='SPECULAR',
                        help='Name of the specular (mask) directory')
    parser.add_argument('--metrics', nargs='+', default=['PSNR', 'SSIM', 'LPIPS'],
                        help='Metrics to compute')
    parser.add_argument('--odd_camera_only', action='store_true',
                        help='Evaluate only odd-numbered cameras')
    
    args = parser.parse_args()
    
    evaluator = MaskedVideoEvaluator(
        data_root=args.data_root,
        render_dir=args.render_dir,
        specular_dir=args.specular_dir,
        compute_metrics=args.metrics,
        odd_camera_only=args.odd_camera_only
    )
    
    results = evaluator.run()