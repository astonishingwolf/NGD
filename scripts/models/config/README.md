# Configuration Files

This directory contains configuration files for the MonoClothRecon project.

## Directory Structure

```
config/
├── README.md                    # This file
├── config_hash.json            # Hash grid encoding configuration
├── config_tex.json             # Texture model configuration
└── dress4D/                    # Dataset-specific configurations
    └── Demo/
        ├── upper_people.yml    # Upper garment configuration
        ├── lower_people.yml    # Lower garment configuration
        ├── monocular_synthetic_top.yaml
        └── monocular_synthetic_botton.yaml
```

## Configuration Files

### JSON Configuration Files

#### `config_hash.json`
Configuration for hash grid encoding used in the neural network models. Contains:
- Loss function settings
- Optimizer parameters
- Hash grid encoding parameters
- Network architecture settings

#### `config_tex.json`
Configuration for texture model. Similar structure to `config_hash.json` but includes additional `tex_network` settings for texture generation.

### YAML Configuration Files

All YAML config files follow a standardized structure with the following sections:

1. **Global Settings**: GPU index, random seed
2. **Model Configuration**: Model type, architecture settings
3. **Dataset Configuration**: Frame ranges, image size
4. **Input Paths**: Paths to SMPL models, meshes, and target images
5. **Training Configuration**: Epochs, learning rates, augmentation
6. **Remeshing Configuration**: Remeshing parameters and thresholds
7. **Loss Configuration**: Loss function weights
8. **Logging Configuration**: TensorBoard logging settings
9. **Output Configuration**: Output directories and save settings
10. **Texture Configuration**: Texture model and training parameters

## Usage

### Loading Configurations

Use the config utility functions to load configurations:

```python
from scripts.utils.config_utils import load_hash_config, load_texture_config

# Load hash grid config
hash_cfg = load_hash_config()

# Load texture config
tex_cfg = load_texture_config()
```

### Running with a Config File

```bash
python main.py --config scripts/models/config/dress4D/Demo/upper_people.yml
```

## Configuration Parameters

### Model Types
- `hashgrid`: Standard hash grid model
- `hashgrid_vd`: Hash grid with pose-dependent variation
- `general`: General SIREN model
- `diffusion`: DiffusionNet-based model

### Skinning Functions
- `rbf`: Radial basis function skinning
- `k-near`: K-nearest neighbors skinning

### Texture Models
- `hashgrid_base`: Base hash grid texture model
- `hashgrid_reg`: Hash grid with regularization

## Notes

- All paths in config files should use absolute paths
- Boolean values should use `true`/`false` (lowercase) in YAML files
- Comments in YAML files start with `#`
- Numeric values can use underscores as separators (e.g., `100_000`)


