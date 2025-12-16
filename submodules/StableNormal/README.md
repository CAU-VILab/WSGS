# StableNormal

Forked from `https://huggingface.co/spaces/Stable-X/StableNormal`.

This is the modified fork version of the [StableNormal](https://huggingface.co/spaces/Stable-X/StableNormal) gradio space for [EasyVolcap](https://github.com/zju3dv/EasyVolcap) format dataset processing.

## Installation

To install the required dependencies, you can use the following command:

```bash
pip install -r requirements.txt
```

## Usage

Once you have a prepared [EasyVolcap](https://github.com/zju3dv/EasyVolcap?tab=readme-ov-file#dataset-structure) format dataset, and have [EasyVolcap](https://github.com/zju3dv/EasyVolcap) installed, you can use the following command to compute the monocular normal maps for the dataset:

```bash
python run.py \
    --data_root /path/to/dataset/root \
    --resolusion 1024 \
    --align 64 \
    --scenes /list/of/scene/names
```

+ `data_root`: The root directory of the dataset, including a list of scenes inside.
+ `resolusion`: The network processing resolution, default to 1024.
+ `align`: The network alignment resolution, default to 64.
+ `scenes`: This argument is optional, if not provided, all scenes in the dataset will be processed, otherwise only the specified scenes will be processed, split by space.

## Notes

+ We use `Stable-X/yoso-normal-v0-3` and `Stable-X/stable-normal-v0-1` for now.
+ Resolution issue: https://github.com/Stable-X/StableNormal/issues/31
