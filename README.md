# NGD: Neural Gradient Based Deformation for Monocular Garment Reconstruction

[Project Page](https://astonishingwolf.github.io/pages/NGD/)
[arxiv](https://arxiv.org/pdf/2508.17712)


## Installation

This installation guide assumes you have Python 3.9+ and recommend using Anaconda to manage the environment.

### Prerequisites

- Python 3.9+
- Anaconda (recommended)
- NVIDIA GPU with CUDA support (CUDA 11.8+ recommended)
- SMPL model files (download from [SMPL website](https://smpl.is.tue.mpg.de/))

### Step 1: Clone the Repository

Clone the repository with submodules:

```bash
git clone --recurse-submodules https://github.com/your-username/MonoClothRecon.git
cd MonoClothRecon
```

If you've already cloned the repository without submodules, initialize them:

```bash
git submodule update --init --recursive
```

### Step 2: Create Conda Environment

```bash
conda create -n monoclothrecon python=3.9
conda activate monoclothrecon
```

### Step 3: Install PyTorch

Install PyTorch with CUDA support. For CUDA 11.8:

```bash
pip install torch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 --index-url https://download.pytorch.org/whl/cu118
```

Or using conda:

```bash
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Step 4: Install pytorch3d

First, install the required dependencies:

```bash
conda install -c fvcore -c iopath -c conda-forge fvcore iopath
```

Then install pytorch3d:

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### Step 5: Install nvdiffrast

Build and install nvdiffrast from the submodule:

```bash
cd packages/nvdiffrast
pip install .
cd ../..
```

### Step 6: Install Additional Dependencies

Install the remaining required Python packages:

```bash
pip install trimesh pyyaml easydict tensorboard torchvision numpy scipy
```

### Step 7: Install Local Packages

Install the local nvdiffmodeling package:

```bash
pip install -e packages/nvdiffmodeling
```

## Usage

### Configuration

1. Copy the configuration template:

```bash
cp scripts/configs/config_template.yml my_config.yml
```

2. Edit `my_config.yml` and update the following paths:
   - `smpl_path`: Path to SMPL_NEUTRAL.pkl
   - `smpl_pkl`: Path to SMPL fitting results
   - `mesh`: Path to template mesh (.obj file)
   - `target_images`: Path to input images
   - `target_diffuse_maps`: Path to diffuse maps
   - `target_shil_maps`: Path to silhouette maps
   - `output_path`: Path for output directory
   - Other dataset-specific paths

### Running Training

Run the full pipeline (geometry + texture reconstruction):

```bash
python main.py --config my_config.yml
```

Or run specific stages:

```python
# Geometry training only
from scripts.geometry import geometry_training_loop
geometry_training_loop(cfg, device)

# Texture/appearance training
from scripts.appearance import appearance_training_loop
appearance_training_loop(cfg, device)

# Inference
from scripts.inference import Inference
Inference(cfg, texture=True, device=device)
```

### Configuration Options

Key configuration parameters in `config.yml`:

- **Model Configuration**:
  - `model_type`: Dataset type (`Dress4D` or `PeopleSnapshot`)
  - `model`: Model architecture (`hashgrid`, `hashgrid_vd`, `general`, `diffusion`)
  
- **Training Configuration**:
  - `num_epochs`: Number of training epochs
  - `warm_ups`: Warm-up epochs before deformation
  - `remeshing`: Enable/disable remeshing
  
- **Texture Configuration**:
  - `texture_recon`: Enable texture reconstruction
  - `use_dynamic_texture`: Use dynamic texture model
  - `texture_map`: Texture map resolution [width, height]

See `scripts/configs/config_template.yml` for all available options.

## Dataset Preparation

### Dress4D Dataset

1. Organize your data in the following structure:
```
data/
  ├── images/
  ├── masks/
  ├── smpl_fittings.pkl
  └── template_mesh.obj
```

2. Update the configuration file with your data paths.

### PeopleSnapshot Dataset

Follow similar structure as Dress4D. Ensure SMPL fittings are available for each frame.

## Project Structure

```
MonoClothRecon/
├── main.py                 # Main entry point
├── scripts/
│   ├── geometry.py         # Geometry training loop
│   ├── appearance.py       # Texture/appearance training
│   ├── inference.py        # Inference and evaluation
│   ├── models/            # Model definitions
│   ├── dataloader/        # Data loading utilities
│   ├── losses/            # Loss functions
│   ├── renderer/          # Rendering utilities
│   └── utils/             # Utility functions
├── packages/              # External dependencies (submodules)
│   ├── nvdiffrast/       # NVIDIA differentiable rasterizer
│   ├── nvdiffrec/        # NVIDIA differentiable reconstruction
│   ├── fashion-clip/      # Fashion-CLIP model
│   ├── nvdiffmodeling/   # Local modeling utilities
│   └── fashion_clip/     # Local fashion-clip copy
└── preprocessing/         # Data preprocessing scripts
```

## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size` in config
- Reduce `image_size`
- Enable gradient checkpointing if available

### Submodule Issues

If submodules are not initialized:

```bash
git submodule update --init --recursive
```

### nvdiffrast Build Errors

Ensure you have:
- CUDA toolkit installed
- Proper CUDA version matching PyTorch
- GCC compiler with C++17 support

## Citation

If you use this code, please cite:

```bibtex
@article{dasgupta2025ngd,
  title={NGD: Neural Gradient Based Deformation for Monocular Garment Reconstruction},
  author={Dasgupta, Soham and Naik, Shanthika and Savalia, Preet and Ingle, Sujay Kumar and Sharma, Avinash},
  journal={arXiv preprint arXiv:2508.17712},
  year={2025}
}
```

## License

Please check individual package licenses:
- nvdiffrast: NVIDIA Source Code License
- nvdiffrec: NVIDIA Source Code License
- fashion-clip: Check original repository license
- This codebase: [Your License]

## Acknowledgments

- NVIDIA nvdiffrast and nvdiffrec for differentiable rendering
- SMPL model from Max Planck Institute
- Fashion-CLIP for fashion understanding

