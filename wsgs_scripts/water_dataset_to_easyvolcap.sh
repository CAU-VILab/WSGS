#!/bin/bash

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