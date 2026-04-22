# CLIPoint3D: Language-Grounded Few-Shot Unsupervised 3D Point Cloud Domain Adaptation (CVPR'26)
> [Mainak Singha](https://mainaksingha01.github.io/), [Sarthak Mehrotra](https://scholar.google.com/citations?user=87yQ-vQAAAAJ&hl=en), [Paolo Casari](https://scholar.google.com/citations?user=CSaXahIAAAAJ&hl=en), [Subhasis Chaudhuri](https://scholar.google.com/citations?user=84VkdegAAAAJ&hl=en), [Elisa Ricci](https://eliricci.eu/), [Biplab Banerjee](https://biplab-banerjee.github.io/)


[![arXiv](https://img.shields.io/badge/arXiv-2602.20409-b31b1b.svg)](https://arxiv.org/abs/2602.20409v2)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://sarthakm320.github.io/CLIPoint3D)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

## Abstract

Recent vision-language models (VLMs) such as CLIP demonstrate impressive cross-modal reasoning, extending beyond images to 3D perception. Yet, these models remain fragile under domain shifts, especially when adapting from synthetic to real-world point clouds. Conventional 3D domain adaptation approaches rely on heavy trainable encoders, yielding strong accuracy but at the cost of efficiency. We introduce CLIPoint3D, the first framework for few-shot unsupervised 3D point cloud domain adaptation built upon CLIP. Our approach projects 3D samples into multiple depth maps and exploits the frozen CLIP backbone, refined through a knowledge-driven prompt tuning scheme that integrates high-level language priors with geometric cues from a lightweight 3D encoder. To adapt task-specific features effectively, we apply parameter-efficient fine-tuning to CLIP's encoders and design an entropy-guided view sampling strategy for selecting confident projections. Furthermore, an optimal transport-based alignment loss and an uncertainty-aware prototype alignment loss collaboratively bridge source-target distribution gaps while maintaining class separability. Extensive experiments on PointDA-10 and GraspNetPC-10 benchmarks show that CLIPoint3D achieves consistent 3-16% accuracy gains over both CLIP-based and conventional encoder-based baselines.

## Setup

### 1. Create the conda environment

```bash
conda env create -f environment.yml
conda activate clipoint3d
```

The environment supports Python 3.9.20 and all dependencies including PyTorch 2.5.1 with CUDA 12.

### 3. Install Dassl

Follow the installation instructions in [Dassl.pytorch/README.md](Dassl.pytorch/README.md). The relevant steps from their guide are:

```bash
cd Dassl.pytorch/

# Install dependencies
pip install -r requirements.txt

# Install this library (no need to re-build if the source code is modified)
python setup.py develop

cd ..
```

### 4. CLIP model weights

CLIP weights are downloaded automatically on first use via the `clip` library. Ensure you have internet access on the first run, or pre-download the `ViT-B/16` weights.

## Dataset Setup

### PointDA

Download the [PointDA-10 dataset](https://drive.google.com/file/d/1-LfJWL5geF9h0Z2QpdTL0n4lShy8wy2J/view) and place it under `PointDA_data/`:

```
PointDA_data/
├── shapenet/
├── modelnet/
└── scannet/
```

### GraspNet

Download the [GraspNet point cloud data](https://graspnet.net/) and place it under `GraspNetPointClouds/`:

```
GraspNetPointClouds/
├── synthetic/
├── kinect/
└── realsense/
```

## Training

### Single experiment

```bash
python train.py \
  --config-file configs/trainers/trainer_200.yaml \
  --dataset-config-file configs/datasets/pointda_shapenet_modelnet.yaml \
  --output-dir experiments/run1 \
  --seed 42 \
  --use_sinkhorn_loss \
  --use_entropy_loss \
  --use_confidence_sampling
```

### Key arguments

| Argument | Default | Description |
|---|---|---|
| `--config-file` | `configs/trainers/trainer.yaml` | Trainer configuration |
| `--dataset-config-file` | `configs/datasets/pointda_shapenet_modelnet.yaml` | Dataset configuration |
| `--output-dir` | `test_runs_with_sinkhorn` | Output directory for checkpoints and logs |
| `--root` | `PointDA_data` | Path to dataset root |
| `--seed` | `42` | Random seed (positive = fixed) |
| `--source-domains` | — | Override source domains |
| `--target-domains` | — | Override target domains |
| `--use_sinkhorn_loss` | off | Optimal transport loss between source/target |
| `--use_entropy_loss` | off | Entropy minimization on target predictions |
| `--use_align_loss` | off | Direct feature alignment loss |
| `--use_prototype_loss` | off | Prototype-based domain alignment |
| `--use_kl_loss` | off | KL divergence loss |
| `--use_w1_loss` | off | Wasserstein-1 distance loss |
| `--use_confidence_sampling` | off | Sample target points by prediction confidence |

Output is saved to `<output-dir>/<model>/<source>/<target>/`.

## Configuration

Configs use [YACS](https://github.com/rbgirshick/yacs) and are split into two files:

- **Trainer config** (`configs/trainers/`): Model architecture, optimizer, batch size, learning rate, number of context tokens. The recommended config is `trainer_200.yaml`.
- **Dataset config** (`configs/datasets/`): Dataset name, source and target domain names. Named as `pointda_<source>_<target>.yaml` or `graspnet_<source>_<target>.yaml`.

You can also override any config value directly from the command line using YACS syntax at the end of the command:

```bash
python train.py ... OPTIM.LR 0.001 DATALOADER.TRAIN_X.BATCH_SIZE 32
```

Key trainer config options:

```yaml
MODEL:
  NAME: CLIPoint3D
  BACKBONE:
    NAME: "ViT-B/16"   # CLIP backbone

OPTIM:
  NAME: "sgd"
  LR: 0.002
  MAX_EPOCH: 200
  LR_SCHEDULER: "cosine"

TRAINER:
  MODEL:
    N_CTX: 4       # Number of learnable context tokens in prompts
    PREC: "fp32"   # Precision: fp32, fp16, or amp
```

## Project Structure

```
clipoint3d/
├── train.py                  # Entry point
├── trainer.py                # Trainer class with loss implementations
├── environment.yml           # Conda environment spec
├── train_single.sh           # Run all PointDA domain pairs
├── train_graspnet.sh         # Run all GraspNet domain pairs
├── ablations.sh              # Ablation study runs
├── models/
│   ├── model.py              # Main model (PointNet + CLIP + cross-attention)
│   ├── pointnet.py           # PointNet 3D encoder
│   ├── prompt_learner.py     # Learnable text prompt module
│   ├── text_encoder.py       # CLIP text encoder wrapper
│   ├── image_encoder.py      # CLIP image encoder wrapper
│   ├── cross_attention.py    # Cross-modal attention module
│   └── lora.py               # LoRA parameter-efficient fine-tuning
├── clip/                     # CLIP model integration
├── utils/
│   ├── config_defaults.py    # YACS config defaults
│   ├── dataloader.py         # Data loading utilities
│   ├── loss.py               # Domain adaptation loss functions
│   ├── render.py             # Point cloud -> multi-view image renderer
│   └── peft_utils.py         # Parameter-efficient fine-tuning helpers
├── configs/
│   ├── datasets/             # Dataset YAML configs
│   └── trainers/             # Trainer YAML configs
├── Dassl.pytorch/            # Domain adaptation framework
├── PointDA_data/             # PointDA dataset (ShapeNet/ModelNet/ScanNet)
└── GraspNetPointClouds/      # GraspNet dataset
```

## Citation

```bibtex
@article{singha2026clipoint3d,
  title={CLIPoint3D: Language-Grounded Few-Shot Unsupervised 3D Point Cloud Domain Adaptation},
  author={Singha, Mainak and Mehrotra, Sarthak and Casari, Paolo and Chaudhuri, Subhasis and Ricci, Elisa and Banerjee, Biplab},
  journal={arXiv preprint arXiv:2602.20409},
  year={2026}
}
```

