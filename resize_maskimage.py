#!/usr/bin/env python3
"""
SPECULAR Image Resize Script
RENDER 폴더의 이미지 크기를 기준으로 SPECULAR 폴더의 이미지들을 리사이즈합니다. (크기가 미세하게 불일치할 경우 사용)
"""

import os
import argparse
from glob import glob
from PIL import Image
from typing import List, Tuple, Optional


def get_image_files(folder_path: str) -> List[str]:
    """폴더에서 이미지 파일 목록 가져오기"""
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in extensions:
        image_files.extend(glob(os.path.join(folder_path, ext)))
        image_files.extend(glob(os.path.join(folder_path, ext.upper())))
    
    return sorted(image_files)


def get_reference_size(render_path: str) -> Tuple[int, int]:
    """RENDER 폴더에서 기준 이미지 크기 가져오기"""
    if not os.path.exists(render_path):
        raise ValueError(f"RENDER folder not found: {render_path}")
    
    image_files = get_image_files(render_path)
    if not image_files:
        raise ValueError(f"No images found in RENDER folder: {render_path}")
    
    # 첫 번째 이미지에서 크기 가져오기
    reference_file = image_files[0]
    try:
        with Image.open(reference_file) as img:
            reference_size = img.size  # (width, height)
        
        print(f"Reference image: {os.path.basename(reference_file)}")
        print(f"Reference size: {reference_size[0]}x{reference_size[1]}")
        
        # 모든 RENDER 이미지가 같은 크기인지 확인
        print(f"Verifying all {len(image_files)} RENDER images have the same size...")
        
        different_sizes = []
        for img_path in image_files:
            try:
                with Image.open(img_path) as img:
                    if img.size != reference_size:
                        different_sizes.append((os.path.basename(img_path), img.size))
            except Exception as e:
                print(f"Warning: Could not read {img_path}: {e}")
        
        if different_sizes:
            print("Warning: Found RENDER images with different sizes:")
            for filename, size in different_sizes:
                print(f"  {filename}: {size[0]}x{size[1]}")
            print("Using the first image size as reference.")
        else:
            print("✓ All RENDER images have the same size")
        
        return reference_size
        
    except Exception as e:
        raise ValueError(f"Could not read reference image {reference_file}: {e}")


def resize_image(img_path: str, target_size: Tuple[int, int], resample_method: int = Image.LANCZOS) -> bool:
    """이미지 리사이즈"""
    try:
        with Image.open(img_path) as img:
            original_size = img.size
            
            # 이미 목표 크기면 스킵
            if original_size == target_size:
                return True
            
            # 리사이즈
            resized_img = img.resize(target_size, resample_method)
            
            # 원본 파일 덮어쓰기
            resized_img.save(img_path)
            
            print(f"    {original_size[0]}x{original_size[1]} → {target_size[0]}x{target_size[1]}")
            return True
            
    except Exception as e:
        print(f"    Error: {e}")
        return False


def resize_specular_folder(specular_path: str, target_size: Tuple[int, int], resample_method: int = Image.LANCZOS):
    """SPECULAR 폴더 내 모든 이미지 리사이즈"""
    if not os.path.exists(specular_path):
        print(f"Warning: SPECULAR folder not found: {specular_path}")
        return
    
    image_files = get_image_files(specular_path)
    if not image_files:
        print(f"No images found in SPECULAR folder: {specular_path}")
        return
    
    print(f"\nProcessing {len(image_files)} images in SPECULAR folder...")
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    
    # 현재 크기 분포 확인
    size_counts = {}
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                size = img.size
                size_counts[size] = size_counts.get(size, 0) + 1
        except Exception:
            continue
    
    print(f"Current size distribution in SPECULAR:")
    for size, count in sorted(size_counts.items()):
        print(f"  {size[0]}x{size[1]}: {count} images")
    
    # 리사이즈 실행
    success_count = 0
    skip_count = 0
    
    for i, img_path in enumerate(image_files, 1):
        filename = os.path.basename(img_path)
        print(f"  [{i}/{len(image_files)}] {filename}", end="")
        
        # 현재 크기 확인
        try:
            with Image.open(img_path) as img:
                if img.size == target_size:
                    print(" (already correct size)")
                    skip_count += 1
                    continue
        except Exception:
            pass
        
        if resize_image(img_path, target_size, resample_method):
            success_count += 1
        
    print(f"\nResults:")
    print(f"  Successfully resized: {success_count}")
    print(f"  Already correct size: {skip_count}")
    print(f"  Failed: {len(image_files) - success_count - skip_count}")


def main():
    parser = argparse.ArgumentParser(description="Resize SPECULAR images to match RENDER image size")
    parser.add_argument('data_root', type=str, 
                        help='Path to the root directory containing RENDER and SPECULAR folders')
    parser.add_argument('--resample', choices=['nearest', 'bilinear', 'bicubic', 'lanczos'], default='lanczos',
                        help='Resampling method (default: lanczos)')
    parser.add_argument('--dry_run', action='store_true', 
                        help='Only analyze sizes without resizing')
    
    args = parser.parse_args()
    
    # 리샘플링 방법 매핑
    resample_map = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS
    }
    resample_method = resample_map[args.resample]
    
    # 경로 설정
    if not os.path.exists(args.data_root):
        print(f"Error: Directory {args.data_root} does not exist")
        return
    
    render_path = os.path.join(args.data_root, 'RENDER')
    specular_path = os.path.join(args.data_root, 'SPECULAR')
    
    print(f"Data root: {args.data_root}")
    print(f"RENDER path: {render_path}")
    print(f"SPECULAR path: {specular_path}")
    
    try:
        # RENDER 폴더에서 기준 크기 가져오기
        reference_size = get_reference_size(render_path)
        
        if args.dry_run:
            print(f"\nDry run mode - would resize SPECULAR images to {reference_size[0]}x{reference_size[1]}")
            
            # SPECULAR 폴더의 현재 상태만 확인
            if os.path.exists(specular_path):
                image_files = get_image_files(specular_path)
                if image_files:
                    print(f"\nSPECULAR folder analysis:")
                    size_counts = {}
                    for img_path in image_files:
                        try:
                            with Image.open(img_path) as img:
                                size = img.size
                                size_counts[size] = size_counts.get(size, 0) + 1
                        except Exception:
                            continue
                    
                    for size, count in sorted(size_counts.items()):
                        status = "✓" if size == reference_size else "✗"
                        print(f"  {status} {size[0]}x{size[1]}: {count} images")
            return
        
        # SPECULAR 폴더 리사이즈 실행
        resize_specular_folder(specular_path, reference_size, resample_method)
        
        print(f"\n✓ Resize operation completed!")
        print(f"All SPECULAR images are now {reference_size[0]}x{reference_size[1]}")
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()