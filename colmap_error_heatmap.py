import os
import numpy as np
from scipy.stats import trim_mean

def load_points3d_txt(path):
    """
    Load COLMAP points3D.txt and return Nx3 array of xyz and N-array of errors.
    """
    verts = []
    errors = []
    
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            x, y, z = map(float, parts[1:4])
            error = float(parts[7])
            verts.append([x, y, z])
            errors.append(error)
    
    return np.array(verts), np.array(errors)

def create_error_heatmap_ply(xyz, errors, output_path='error_heatmap.ply'):
    """
    Map errors to a black->red colormap, 
    override top 5% errors to pure blue, and write PLY file.
    """
    # Normalize errors to [0, 1]
    err_min, err_max = errors.min(), errors.max()
    norm = (errors - err_min) / (err_max - err_min + 1e-12)
    # Base heatmap: black->red
    colors = (np.stack([norm, np.zeros_like(norm), np.zeros_like(norm)], axis=1) * 255).astype(np.uint8)
    
    # Determine top 5% threshold and override those colors to pure blue
    threshold5 = np.percentile(errors, 95)
    mask_top5 = errors >= threshold5
    colors[mask_top5] = np.array([0, 0, 255], dtype=np.uint8)
    
    # Write ASCII PLY
    num_vertices = xyz.shape[0]
    with open(output_path, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {num_vertices}\n')
        f.write('property float x\nproperty float y\nproperty float z\n')
        f.write('property uchar red\nproperty uchar green\nproperty uchar blue\n')
        f.write('end_header\n')
        for (x, y, z), (r, g, b) in zip(xyz, colors):
            f.write(f'{x} {y} {z} {r} {g} {b}\n')

def main():
    # Update this path to point to your points3D.txt file
    path = './data/datasets/original/water_real/fishbawl_v02/colmap/colmap_text/points3D.txt'
    #path = './data/datasets/original/water_synthetic/basin_v04/colmap/colmap_text/points3D.txt'
    #path = './data/datasets/original/water_synthetic/swimming_pool_v8_bg/colmap/colmap_text/points3D.txt'
    if not os.path.isfile(path):
        print(f"File not found: {path}. Please update the path and rerun.")
        return
    
    xyz, errors = load_points3d_txt(path)
    
    # Generate heatmap PLY with top 5% in blue
    create_error_heatmap_ply(xyz, errors, 'error_heatmap_with_top5_blue.ply')
    print("Generated 'error_heatmap_with_top5_blue.ply'. Open it in MeshLab to view the modified heatmap.")
    
    # Compute and print statistics (optional, unchanged)
    for top_percent in [20, 10, 5, 3, 1]:
        threshold = np.percentile(errors, 100 - top_percent)
        group = xyz[errors >= threshold]
        
        mean_coords   = group.mean(axis=0)
        median_coords = np.median(group, axis=0)
        trim10_coords = np.array([trim_mean(group[:, i], proportiontocut=0.10) for i in range(3)])
        trim20_coords = np.array([trim_mean(group[:, i], proportiontocut=0.20) for i in range(3)])
        
        print(f"\n=== Statistics for Top {top_percent}% by Error ===")
        print(f"Count               : {len(group)} (Threshold = {threshold:.3f} px)")
        print(f"Mean coordinates    : {mean_coords}")
        print(f"Median coordinates  : {median_coords}")
        print(f"10% Trimmed mean    : {trim10_coords}")
        print(f"20% Trimmed mean    : {trim20_coords}")

if __name__ == "__main__":
    main()


#path = './data/datasets/original/water_synthetic/swimming_pool_v8_bg/colmap/colmap_text/points3D.txt'