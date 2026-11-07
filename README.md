# Kernel Adaptive Deformable Convolution for Remote Sensing Object Detection

Official code for our IEEE TPAMI paper.

KAD-Conv adapts both where a convolution samples (Deformable Spatial Convolution, whose modulation mask comes from a low-cost spatial attention) and at what scale it samples (multi-branch downsampling with learnable branch and channel selection). KAD-CNet is a detector built on KAD-Conv and trained with Multi-task Guided Learning (MGL).

| Folder | What it is |
|---|---|
| [KAD-CNet](KAD-CNet) | The detector: DIOR data preparation, released weights, inference, evaluation, training, and the optimized inference used for the speed results. |
| [KAD-Conv](KAD-Conv) | The operator alone, in plain PyTorch, plus a Triton kernel for faster inference. Use this to try KAD-Conv in your own network. |

Weights are on the [Releases](https://github.com/MLMIP/KAD-Conv/releases) page.

| Model | MGL | DIOR mAP50 |
|---|:---:|:---:|
| KAD-CNet-S | | 80.16 |
| KAD-CNet-S† | ✓ | 80.59 |

## Citation

If you find this work useful, please cite our paper. The BibTeX entry will be added once the paper is published.

## Acknowledgements

The detection code is built on [YOLOv9](https://github.com/WongKinYiu/yolov9), and we thank its authors. DIOR was released by Li et al. and is distributed by its authors under CC BY-NC 4.0; please cite the DIOR paper if you use the dataset.

## License

GPL-3.0, see [LICENSE](LICENSE).
