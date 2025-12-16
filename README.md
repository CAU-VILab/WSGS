<div align="center">

# Through the Water: Refractive Gaussian Splatting for Water Surface Scenes

[**Yeonghun Yoon***](https://sites.google.com/vilab.cau.ac.kr/yyhos/) · [**Hojoon Jung***](https://sites.google.com/view/hjjung-homepage/%ED%99%88) · [**Jaeyoon Lee***](https://sites.google.com/vilab.cau.ac.kr/jaeyoonlee) · [**Taegwan Kim***](https://algomalgo.notion.site/CAD-Lab-8c88a7ab65754e018d3e070ee92d79c5) · [**Gyuhyun Kim***](https://algomalgo.notion.site/CAD-Lab-8c88a7ab65754e018d3e070ee92d79c5) · [**Jongwon Choi**](https://scholar.google.co.kr/citations?user=F3u9qHcAAAAJ&hl)<sup>&dagger;</sup> 
<br>

[![paper](https://img.shields.io/badge/Paper-WSGS-b31b1b?logo=adobe&logoColor=fff)](#)
[![page](https://img.shields.io/badge/Project_Page-WSGS-green?logo=html&logoColor=fff)](https://cau-vilab.github.io/WSGS/)
[![Google Drive](https://img.shields.io/badge/Drive-Dataset_Request-4285F4?logo=googledrive&logoColor=fff)](https://forms.gle/3G84ABSGmTHLnbpW9)
</div>


![teaser](assets/imgs/teaser.png)



***News***:

- 25.04.16 : Writing code...
- 25.08.01 : Submitted to AAAI 2026
- 25.11.08 : AAAI 2026 Main Technical Track accepted
- 25.12.13 : Added project page
- 25.12.14 : Writing README.md & refactoring code now...

## Installation
```shell
# Clone the submodules
git submodule update --init --recursive

# Create a new conda environment
conda create -n wsgs "python=3.11" -y
conda activate wsgs

# Install PyTorch (CUDA 11.8)
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118 

# Install pip dependencies
cat requirements-dev.txt | sed -e '/^\s*-.*$/d' -e '/^\s*#.*$/d' -e '/^\s*$/d' | awk '{split($0, a, "#"); if (length(a) > 1) print a[1]; else print $0;}' | awk '{split($0, a, "@"); if (length(a) > 1) print a[2]; else print $0;}' | xargs -n 1 pip install

pip install -e . --no-build-isolation --no-deps

# Install submodules
git submodule update --init --recursive

# Install the 2D Gaussian Tracer & 2D Gaussian rasterizers 
# from EnvGS (https://github.com/zju3dv/EnvGS)
pip install -v submodules/diff-surfel-tracing
pip install submodules/diff-surfel-rasterizations/diff-surfel-rasterization-wet submodules/diff-surfel-rasterizations/diff-surfel-rasterization-wet-ch05 submodules/diff-surfel-rasterizations/diff-surfel-rasterization-wet-ch07

# Install StableNormal(https://github.com/Stable-X/StableNormal) for monocular normal estimation
pip install -r submodules/StableNormal/requirements.txt
```


## Dataset
The Water Real and Water Synthetic datasets can be requested via the Dataset Request link above and are available for academic use only.
The following section describes the data preprocessing procedure.
### Dataset Preparation

To run the code, Structure-from-Motion must first be performed, and the resulting data must then be converted into the easyvolcap format. The final directory structure is shown below:

```shell
data/water_real/fishbowl_v02
│── extri.yml
│── intri.yml
├── images
│   ├── 00
│   │   ├── 000000.jpg
│   │   ├── 000001.jpg
│   │   ...
│   │   ...
│   └── 01
│   ...
└── normal
    ├── 00
    │   ├── 000000.jpg
    │   ├── 000001.jpg
    │   ...
    │   ...
    └── 01
    ...
```


First, extract and save frames from the videos using ffmpeg.
Move to the directory containing the videos, then run:
```shell
mkdir images
ffmpeg -f concat -safe 0 -i list.txt -r 8 -q:v 1 ./images/%06d.jpg -hide_banner -loglevel error
```
The -r option controls the frame rate and determines how many frames are extracted per second. \
Here, -r 8 means that 8 frames per second are extracted.

The list.txt file is used to concatenate and process multiple videos sequentially, and has the following format:
```shell
#list.txt
file 'fishbowl_1.mp4'
file 'fishbowl_2.mp4'
file 'fishbowl_3.mp4'
...
```

Next, run COLMAP to obtain `cameras.bin`, `images.bin`, `points3D.bin`, `project.ini`, and `colmap.db` (in our case, we used COLMAP GUI).
Place these files, along with the `images` directory used for COLMAP, according to the directory structure specified below. (If COLMAP does not work properly, it may be helpful to use COLMAP’s masking feature to prevent feature matching in water regions.)
```shell
data/original/water_real/fishbowl_v02
├── colmap
│   ├── colmap.db
│   └── colmap_sparse
│       └── 0
│           ├── cameras.bin
│           ├── images.bin
│           ├── points3D.bin
│           └── project.ini
└── images
    ├── 000001.jpg
    ├── 000002.jpg
    ├── ...
    └── ...
```

Then, update the values below in project.ini so that they match the actual paths.
```shell
#data/original/water_real/fishbowl_v02/colmap/colmap_sparse/project.ini
database_path=data/datasets/original/water_real/fishbowl_v02/colmap/colmap.db
image_path=data/datasets/original/water_real/fishbowl_v02/images
```


Once Structure-from-Motion using COLMAP has been completed and the results have been placed in the appropriate directory, the data can be converted into the easyvolcap format using `wsgs_scripts/water_dataset_to_easyvolcap.sh`.

```shell
# Ex) Convert Water Real fishbowl data to easyvolcap format

# Define the dataset name and scene name
dataset=water_real
scene=basin
colmap_root=data/datasets/original/$dataset
easyvolcap_root=data/datasets/$dataset

#2. Run COLMAP: once you have images stored in `data/datasets/original/water_synthetic/fishbowl_v02/images/*.jpg`
python scripts/colmap/run_colmap.py --data_root $colmap_root/$scene --images images --use_gpu --colmap_matcher sequential

# # 3. COLMAP to EasyVolcap: convert the colmap format dataset to EasyVolcap format 
# # `--colmap colmap/colmap_sparse/0` is the default COLMAP sparse output directory if you are using the `run_colmap.py` script in the previous step, you can change it to your own COLMAP sparse output directory
python scripts/preprocess/colmap_to_easyvolcap.py --data_root $colmap_root --output $easyvolcap_root --scenes $scene --colmap colmap/colmap_sparse/0

# # 4. Run StableNormal: prepare the monocular normal maps for supervision
python submodules/StableNormal/run.py --data_root $easyvolcap_root --scenes $scene

# 5. Metadata: prepare the scene-specific dataset configs parameters for WSGS
python scripts/preprocess/tools/compute_metadata.py --data_root $easyvolcap_root --scenes $scene --eval
```

### Configurations
For data stored in the easyvolcap format, a configuration file must be created for each dataset. This file specifies the data path, the views used for training and evaluation, as well as the values described below.

```yaml
# Content of configs/datasets/water_real/fishbowl_v02.yaml
configs: configs/datasets/water_real/wsgs.yaml

dataloader_cfg:
    dataset_cfg: &dataset_cfg
        ratio: 0.5
        data_root: data/datasets/water_real/fishbowl_v02
        view_sample: [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 86, 88, 90, 92, 94, 96, 98, 100, 102, 104, 
                    106, 108, 110, 112, 114, 116, 118, 120, 122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 200, 
                    202, 204, 206, 208, 210, 212, 214, 216, 218, 220, 222, 224, 226, 228, 230, 232, 234, 236, 238, 240, 242, 244, 246, 248, 
                    250, 252, 254, 256, 258, 260, 262, 264, 266, 268, 270, 272, 274, 276, 278, 280, 282, 284, 286, 288, 290, 292, 294, 296, 
                    298, 300, 302, 304, 306, 308, 310, 312, 314, 316, 318, 320, 322, 324, 326, 328, 330, 332, 334, 336, 338, 340, 342, 344, 
                    346, 348, 350, 352, 354, 356, 358, 360, 362, 364, 366, 368, 370, 372, 374, 376, 378, 380, 382, 384, 386, 388, 390, 392, 
                    394, 396, 398, 400, 402, 404, 406, 408, 410, 412, 414, 416, 418, 420, 422, 424, 426, 428, 430, 432, 434, 436, 438, 440, 
                    442, 444, 446, 448, 450, 452, 454, 456, 458, 460, 462, 464, 466, 468, 470, 472, 474, 476, 478, 480, 482, 484, 486, 488, 
                    490, 492, 494, 496, 498, 500]
val_dataloader_cfg:
    dataset_cfg:
        <<: *dataset_cfg
        view_sample: [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 85, 87, 89, 91, 93, 95, 97, 99, 101, 103, 105, 107, 109, 111, 
                    113, 115, 117, 119, 121, 123, 125, 127, 129, 131, 133, 135, 137, 139, 141, 201, 203, 205, 207, 209, 211, 213, 215, 217, 219, 221, 223, 
                    225, 227, 229, 231, 233, 235, 237, 239, 241, 243, 245, 247, 249, 251, 253, 255, 257, 259, 261, 263, 265, 267, 269, 271, 273, 275, 277, 279, 
                    281, 283, 285, 287, 289, 291, 293, 295, 297, 299, 301, 303, 305, 307, 309, 311, 313, 315, 317, 319, 321, 323, 325, 327, 329, 331, 333, 335, 
                    337, 339, 341, 343, 345, 347, 349, 351, 353, 355, 357, 359, 361, 363, 365, 367, 369, 371, 373, 375, 377, 379, 381, 383, 385, 387, 389, 391, 
                    393, 395, 397, 399, 401, 403, 405, 407, 409, 411, 413, 415, 417, 419, 421, 423, 425, 427, 429, 431, 433, 435, 437, 439, 441, 443, 445, 447, 
                    449, 451, 453, 455, 457, 459, 461, 463, 465, 467, 469, 471, 473, 475, 477, 479, 481, 483, 485, 487, 489, 491, 493, 495, 497, 499]
model_cfg:
    sampler_cfg:
        preload_gs: data/datasets/water_real/fishbowl_v02/sparse/0/points3D.ply
        spatial_scale: 5.930263676977028
        # Environment Gaussian
        env_preload_gs: data/datasets/water_real/fishbowl_v02/envs/points3D.ply
        env_bounds: [[-7.140890789217361, 1.1771738839981318, -7.674651231798872], [9.847743611337476, 9.918000233277503, 4.248896983072468]]
        refracted_preload_gs: data/datasets/water_real/fishbowl_v02/refracted/points3D.ply
        refracted_bounds: [[-7.140890789217361, 1.1771738839981318, -7.674651231798872], [9.847743611337476, 9.918000233277503, 4.248896983072468]]
        white_bg: False
        initial_plane_offset: -2.15 # used for water height init
        initial_plane_normal: [-0.03054, -0.99845, -0.04661] # used for water plane init
```



#### Required Configurations

There are some specific dataset-related parameters required by *WSGS*, you can get env_bounds parameters after running preprocess script above.

+ `model_cfg.sampler_cfg.env_bounds=[[..., ..., ...], [..., ..., ...]]`: this the calculated 3d bounding box of the COLMAP sparse points., used for the environment Gaussian initialization.
+ `refracted_preload_gs` is specified in a similar manner to the above, and `refracted_bounds` is set to be identical to env_bounds.

+ For `initial_plane_offset`, the value is determined based on the statistics obtained from `colmap_error_heatmap.py`.
+ And for `initial_plane_normal`, an appropriate value is either explored using the COLMAP model orientation aligner or manually determined and specified.



Next, you need to create an experiment file that specifies the model and dataset to be used.

```yaml
# configs/exps/wsgs/water_real/fishbowl_v02.yaml
configs:
    - configs/base.yaml # default arguments for the whole codebase
    - configs/models/wsgs.yaml # model configuration
    - configs/datasets/water_real/fishbowl_v02.yaml # dataset usage configuration

model_cfg:
    supervisor_cfg:
        perc_loss_start_iter: 45000 # less iterations for faster training
    sampler_cfg:
        max_gs: 2000000
runner_cfg:
    epochs: 140
    

# prettier-ignore
exp_name: {{fileBasenameNoExtension}}
```




## Usage
### Training
Once data preprocessing is complete, run the following command to start training:
```shell
# Train on the Water Real dataset
evc-train -c configs/exps/wsgs/water_real/fishbowl_v02.yaml exp_name=wsgs/water_real/bucket # bucket
evc-train -c configs/exps/wsgs/water_real/fishbowl_v02.yaml exp_name=wsgs/water_real/fishbowl # fishbowl
evc-train -c configs/exps/wsgs/water_real/basin.yaml exp_name=wsgs/water_real/basin_real # basin real

# Train on the Water Synthetic dataset
evc-train -c configs/exps/wsgs/water_synthetic/swimming_pool_clear.yaml exp_name=wsgs/water_synthetic/swimming_pool # swimming pool
evc-train -c configs/exps/wsgs/water_synthetic/basin_clear.yaml exp_name=wsgs/water_synthetic/basin # basin
evc-train -c configs/exps/wsgs/water_synthetic/pond_v06.yaml exp_name=wsgs/water_synthetic/pond # pond
evc-train -c configs/exps/wsgs/water_synthetic/kitchen_v06.yaml exp_name=wsgs/water_synthetic/kitchen # kitchen
```

### Evaluation
The rendering results and quantitative evaluation metrics for the test views can be found in the `RENDER/` directory and the `metrics.json` file under the experiment directory specified by `exp_name`.

To perform quantitative evaluation only on *water regions*, the pseudo masks generated during *WSGS* training must be used.
These pseudo mask images are stored in the `SPECULAR/` directory, and water region evaluation can be performed using `mask_eval.py`.
```shell
python mask_eval.py --data_root ./data/result/wsgs/water_real/fishbowl --odd_camera_only 
```
+ `--data_root` : Path to the experiment result directory that contains the RENDER/, SPECULAR/, and metrics.json files.
+ `--odd_camera_only` : If set, evaluation is performed only on test views captured from odd-indexed cameras.

If you want to evaluate only the water regions for results from models other than WSGS (e.g., 2DGS, EnvGS, etc.), you can copy the SPECULAR/ directory from a WSGS result directory and reuse it for evaluation.

### Rendering
Once training is finished, run the following command to perform rendering:
```shell
# Testing with input views and evaluating metrics
evc-test -c configs/exps/wsgs/water_real/fishbowl_v02.yaml # Only rendering some selected testing views

# Rendering a rotating novel view path
evc-test -c configs/exps/wsgs/water_real/fishbowl_v02.yaml,configs/specs/cubic.yaml # Render a cubic novel view path, simple interpolation
evc-test -c configs/exps/wsgs/water_real/fishbowl_v02.yaml,configs/specs/spiral.yaml # Render a spiral novel view path
evc-test -c configs/exps/wsgs/water_real/fishbowl_v02.yaml,configs/specs/orbit.yaml # Render an orbit novel view path

# GUI Rendering
evc-gui -c configs/exps/wsgs/water_real/fishbowl_v02.yaml viewer_cfg.window_size=540,960
```




## Acknowledgments
This work was partly supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) [IITP-2023(2024)-RS-2024-00418847, Graduate School of Metaverse Convergence support program; RS-2021-II211341, Artificial Intelligence Graduate School Program (Chung-Ang University)].


## Related Works
Thank you to the following prior works that provided the basis for conducting this research:
- [EnvGS: Modeling View-Dependent Appearance with Environment Gaussian](https://zju3dv.github.io/envgs/)
- [3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://github.com/graphdeco-inria/gaussian-splatting)
- [2DGS: 2D Gaussian Splatting for Geometrically Accurate Radiance Fields](https://surfsplatting.github.io/)


## Citation

If you find this code useful for your research, please cite us using the following BibTeX entry.

```
@article{yoon2026wsgs,
  title={Through the Water: Refractive Gaussian Splatting for Water Surface Scenes},
  author={Yeonghun Yoon, Hojoon Jung, Jaeyoon Lee, Taegwan Kim, Gyuhyun Kim, Jongwon Choi},
  journal={AAAI},
  year={2026}
}
```
