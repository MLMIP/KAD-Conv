# KAD-CNet

Official code for **Kernel Adaptive Deformable Convolution for Remote Sensing Object Detection** (IEEE TPAMI).

KAD-Conv adapts both where a convolution samples (Deformable Spatial Convolution with a low-cost spatial attention mask) and at what scale it samples (multi-branch downsampling with learnable branch selection). KAD-CNet builds a detector on top of it and adds Multi-task Guided Learning (MGL), two auxiliary training branches for super-resolution and foreground segmentation that are removed at inference.

This folder covers KAD-CNet-S on DIOR: the trained weights, data preparation, inference, evaluation, training, and the optimized inference implementation used for the speed results in the paper. To use KAD-Conv in another network, see [KAD-Conv](../KAD-Conv).

## Results on DIOR test

| Model | MGL | mAP50 | mAP50:95 | Weights |
|---|:---:|:---:|:---:|---|
| KAD-CNet-S | | 80.16 | 58.34 | `kad_cnet_s_dior_base.pt` |
| KAD-CNet-S† | ✓ | 80.59 | 58.44 | `kad_cnet_s_dior_mgl.pt` |

Input size 512, FP32, conf 0.001, NMS IoU 0.4 (see [Evaluation](#evaluation)).

## Installation

Tested with Python 3.10, PyTorch 2.1.2 (CUDA 12.1) and torchvision 0.16.2 on Linux.

```bash
git clone https://github.com/MLMIP/KAD-Conv.git && cd KAD-Conv/KAD-CNet
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The default (`raw`) path needs nothing else. The optimized inference path compiles a small CUDA extension on first use, so it also needs the CUDA 12.1 toolkit (`nvcc`), a C++ compiler and `ninja`:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST=8.0
```

Build files and caches go to `outputs/.runtime` (override with `KADC_RUNTIME_DIR`).

## Data

We use the horizontal-box version of [DIOR](https://gcheng-nwpu.github.io/#Datasets) (not DIOR-R). Download it from the dataset page and keep the original layout:

```
DIOR/
├── JPEGImages-trainval/
├── JPEGImages-test/
├── Annotations/          # or Annotations/Horizontal Bounding Boxes/
└── ImageSets/Main/{train,val,test}.txt
```

Then convert it to YOLO format. The raw folder is not modified; images are linked into the output folder.

```bash
python tools/prepare_dior.py --raw /path/to/DIOR --output /path/to/DIOR_yolo \
    --yaml outputs/data/DIOR.yaml --example-id 12500
```

This gives 5,862 train / 5,863 val / 11,738 test images; train and val are merged for training (11,725). `--example-id` copies one test image to `DIOR_yolo/examples/` for the demo below.

## Weights

Download both checkpoints from the [Releases](https://github.com/MLMIP/KAD-Conv/releases) page into `weights/`, then check them:

```bash
python tools/verify_weight.py
```

KAD-CNet-S† is loaded from the checkpoint's `model` entry and KAD-CNet-S from its `ema` entry. `weights/manifest.json` records this, so the scripts pick the right one automatically. The two models also use slightly different KAD-Conv variants (`configs/dior_s_mgl.yaml` and `configs/dior_s_base.yaml`).

## Inference

```bash
python inference/infer_dior.py --source /path/to/DIOR_yolo/examples/dior_12500.jpg
```

`--source` can also be a folder. Results (annotated images, YOLO-format txt and `detections.json`) are written to `outputs/detect`. Add `--weights weights/kad_cnet_s_dior_base.pt` to run the model without MGL.

`--backend` selects the implementation:

- `raw`: plain PyTorch, runs on any GPU.
- `optimized`: our fused CUDA/Triton kernels for KAD-Conv; tested on A100 only.
- `graph`: `optimized` plus CUDA Graph.

All three give the same detections. `inference/check_equivalence.py` compares their outputs on random inputs.

## Evaluation

```bash
python tools/evaluate.py --data outputs/data/DIOR.yaml
```

This follows the protocol of the paper: input size 512 with rectangular batching (YOLO `rect`, pad 0.5), batch size 16, conf 0.001, NMS IoU 0.4, max 300 detections. Metrics are saved to `outputs/evaluation/metrics.json`.

`--protocol square --batch-size 1` evaluates on fixed 512×512 inputs instead, which is the setting the optimized kernels are built for. Use it with `--backend raw` and `--backend optimized` to compare the two on equal terms. mAP50 is lower under this protocol (78.88 for KAD-CNet-S, identical for both backends) because of the different input padding.

## Speed

```bash
python inference/benchmark_s.py
```

The script times the model forward pass of `raw` and `optimized` at batch size 1, 512×512 and FP32 (TF32 off), both with Conv-BN fusion and CUDA Graph. Image loading, preprocessing and NMS are not included. On one A100-SXM4-80GB, KAD-CNet-S† runs at 57.4 FPS with `raw` and 190.5 FPS with `optimized`. Absolute numbers depend on the GPU and driver.

## Training

```bash
# KAD-CNet-S† (with MGL)
python train.py --cfg configs/dior_s_mgl.yaml --hyp configs/hyp.dior_mgl.yaml \
    --data outputs/data/DIOR.yaml --weights '' --DP --IA \
    --epochs 150 --batch-size 16 --imgsz 512 --close-mosaic 15 --name dior_s_mgl

# KAD-CNet-S (without MGL)
python train.py --cfg configs/dior_s_base.yaml --hyp configs/hyp.dior.yaml \
    --data outputs/data/DIOR.yaml --weights '' \
    --epochs 150 --batch-size 16 --imgsz 512 --close-mosaic 15 --name dior_s_base
```

`--DP` and `--IA` turn on the two MGL branches (detail preservation and instance awareness). Their loss weights are 1.0 and 0.5. The branches are only used during training.

## Code structure

```
models/module/KADC/module.py   KAD-Conv (KADC4, KADC4_m) and the GELAN blocks built on it
models/module/ops.py           DSC (DCN_RCA) and LCSA (RCA)
models/Multi_task_learning/    MGL decoders
utils/loss_tal.py              detection and MGL losses
inference/acceleration/        optimized kernels
tools/                         data preparation, evaluation, weight check
configs/                       model and hyper-parameter files
```

The class names follow the released checkpoints, which store whole module objects.

## Citation, acknowledgements and license

See the [main README](../README.md).
