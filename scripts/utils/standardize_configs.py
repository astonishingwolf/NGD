"""
Script to standardize all YAML config files to match the template format.
"""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional


def get_dataset_info_from_path(config_path: Path) -> tuple:
    """Extract dataset type and garment type from file path."""
    parts = config_path.parts
    dataset_type = "Unknown"
    garment_type = "Unknown"
    
    if "dress4D" in parts:
        dataset_type = "Dress4D"
    elif "people_snapshot" in parts:
        dataset_type = "PeopleSnapshot"
    
    if "upper" in config_path.stem.lower() or "top" in config_path.stem.lower() or "tshirt" in config_path.stem.lower() or "shirt" in config_path.stem.lower():
        garment_type = "Upper"
    elif "lower" in config_path.stem.lower() or "botton" in config_path.stem.lower() or "pants" in config_path.stem.lower() or "skirt" in config_path.stem.lower() or "shorts" in config_path.stem.lower():
        garment_type = "Lower"
    
    return dataset_type, garment_type


def normalize_value(value: Any) -> Any:
    """Normalize boolean values and numeric formatting."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        # Convert string booleans
        if value.lower() in ('true', 'yes', '1'):
            return True
        elif value.lower() in ('false', 'no', '0'):
            return False
        # Convert numeric strings with underscores
        if '_' in value and value.replace('_', '').isdigit():
            return int(value.replace('_', ''))
    return value


def extract_config_values(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract and normalize all config values from the loaded YAML."""
    if 'timeloop' not in cfg:
        return {}
    
    t = cfg['timeloop']
    
    # Define default values based on template
    values = {
        # Model Configuration
        'model_type': t.get('model_type', 'Dress4D'),
        'model': t.get('model', 'hashgrid_vd'),
        'hidden_layers': t.get('hidden_layers', 3),
        'batch_size': t.get('batch_size', 1),
        'device': t.get('device', 'cuda:0'),
        
        # Dataset Configuration
        'start_frame': t.get('start_frame', 0),
        'end_frame': t.get('end_frame', 100),
        'skip_frames': t.get('skip_frames', 1),
        'image_size': t.get('image_size', 1080),
        
        # Input Paths
        'smpl_path': t.get('smpl_path', '/path/to/SMPL_NEUTRAL.pkl'),
        'smpl_pkl': t.get('smpl_pkl', '/path/to/smpl_fitting.pkl'),
        'template_smpl_pkl': t.get('template_smpl_pkl'),
        'custom_template': normalize_value(t.get('custom_template', False)),
        'mesh': t.get('mesh', '/path/to/template_mesh.obj'),
        'target_images': t.get('target_images', '/path/to/target_images'),
        'target_diffuse_maps': t.get('target_diffuse_maps', '/path/to/diffuse_maps'),
        'target_shil_maps': t.get('target_shil_maps', '/path/to/silhouette_maps'),
        'target_shil_seg_maps': t.get('target_shil_seg_maps', '/path/to/segmented_silhouette_maps'),
        'target_complete_shil_maps': t.get('target_complete_shil_maps', '/path/to/complete_silhouette_maps'),
        'target_hand_mask': t.get('target_hand_mask', '/path/to/hand_masks'),
        'target_depth': t.get('target_depth'),
        'target_normal': t.get('target_normal'),
        
        # Training Configuration
        'num_epochs': t.get('num_epochs', 1200),
        'warm_ups': t.get('warm_ups', 100),
        'warm_ups_remesh': t.get('warm_ups_remesh', 400),
        'remesh_freq': t.get('remesh_freq', 200),
        'position_lr_init': t.get('position_lr_init', 0.0003),
        'position_lr_final': t.get('position_lr_final', 0.00000016),
        'position_lr_delay_mult': t.get('position_lr_delay_mult', 0.01),
        'position_lr_max_steps': t.get('position_lr_max_steps', 100000),
        'deform_lr_max_steps': t.get('deform_lr_max_steps', 140000),
        'custom_training_cannonical': normalize_value(t.get('custom_training_cannonical', True)),
        'pose_noise': normalize_value(t.get('pose_noise', False)),
        'noise_level': t.get('noise_level', 0.002),
        'vertex_noise': normalize_value(t.get('vertex_noise', False)),
        'skinning_func': t.get('skinning_func', 'rbf'),
        'gradient_clipping': normalize_value(t.get('gradient_clipping', True)),
        'stop_train_after_warmup': normalize_value(t.get('stop_train_after_warmup', True)),
        
        # Remeshing Configuration
        'remeshing': normalize_value(t.get('remeshing', True)),
        'remesh_stop': t.get('remesh_stop', 750),
        'remesh_percentile': t.get('remesh_percentile', 0.8),
        'max_verts': t.get('max_verts', 15000),
        'start_edge_len': t.get('start_edge_len', 0.01),
        'end_edge_len': t.get('end_edge_len', 0.007),
        'start_percentile': t.get('start_percentile', 0.85),
        'end_percentile': t.get('end_percentile', 0.95),
        'retriangulate': t.get('retriangulate', 0),
        'create_new_optimizer': normalize_value(t.get('create_new_optimizer', False)),
        'interpolate_new_optimizer': normalize_value(t.get('interpolate_new_optimizer', True)),
        
        # Loss Configuration
        'losses': t.get('losses', {
            'RenderingLoss': 5.0,
            'Regularization': 0.0,
            'StVKLoss': 1.0,
            'BendingLoss': 1.0,
            'DepthDistillation': 1.0
        }),
        
        # Logging Configuration
        'logging': normalize_value(t.get('logging', False)),
        'log_interval': t.get('log_interval', 5),
        'log_interval_im': t.get('log_interval_im', 150),
        'log_elev': t.get('log_elev', 0),
        'log_fov': t.get('log_fov', 60.0),
        'log_dist': t.get('log_dist', 3.0),
        'log_res': t.get('log_res', 512),
        'log_light_power': t.get('log_light_power', 3.0),
        
        # Output Configuration
        'output_path': t.get('output_path', '/path/to/output'),
        'Exp_name': t.get('Exp_name', 'experiment_name'),
        'save_instance': normalize_value(t.get('save_instance', True)),
        'save_index': t.get('save_index', 10),
        'save_freq': t.get('save_freq', 100),
        
        # Texture Configuration
        'texture_recon': normalize_value(t.get('texture_recon', True)),
        'texture_model': t.get('texture_model', 'hashgrid_base'),
        'texture_map': t.get('texture_map', [1024, 1024]),
        'max_mip_level': t.get('max_mip_level', 3),
        'tex_epochs': t.get('tex_epochs', 1100),
        'texture_warm_ups': t.get('texture_warm_ups', 100),
        'use_dynamic_texture': normalize_value(t.get('use_dynamic_texture', True)),
        'texture_lr_init': t.get('texture_lr_init', 0.002),
        'texture_lr_final': t.get('texture_lr_final', 0.000016),
        'texture_lr_delay_mult': t.get('texture_lr_delay_mult', 0.01),
        'texture_lr_max_steps': t.get('texture_lr_max_steps', 100000),
        'spatial_lr_scale': t.get('spatial_lr_scale', 5),
        'ssim_coeff': t.get('ssim_coeff', 0.001),
        'texture_lr_base': t.get('texture_lr_base', 1e-4),
        'texture_lr_base_dyn': t.get('texture_lr_base_dyn', 1e-4),
    }
    
    # Handle remesh_percentile if it appears twice (take the last one)
    if isinstance(t.get('remesh_percentile'), list):
        values['remesh_percentile'] = t['remesh_percentile'][-1]
    elif 'remesh_percentile' in t:
        values['remesh_percentile'] = t['remesh_percentile']
    
    # Set DepthDistillation to 0 if target_depth is null
    if values['target_depth'] is None or str(values['target_depth']).lower() == 'null':
        if 'DepthDistillation' in values['losses']:
            values['losses']['DepthDistillation'] = 0.0
    
    return values


def format_config_file(config_path: Path, values: Dict[str, Any], dataset_type: str, garment_type: str) -> str:
    """Format config values into the standardized template format."""
    
    # Create header
    if dataset_type != "Unknown":
        header = f"# ============================================================================\n"
        header += f"# MonoClothRecon Configuration File\n"
        header += f"# Dataset: {dataset_type}"
        if garment_type != "Unknown":
            header += f" - {garment_type} Garment"
        header += f"\n# ============================================================================\n"
    else:
        header = f"# ============================================================================\n"
        header += f"# MonoClothRecon Configuration File\n"
        header += f"# ============================================================================\n"
    
    content = header + "\n"
    content += "# Global Settings\n"
    content += f"gpu: {values.get('gpu', 0)}\n"
    content += f"seed: {values.get('seed', 99)}\n\n"
    
    content += "timeloop:\n"
    
    # Model Configuration
    content += "  # ========================================================================\n"
    content += "  # Model Configuration\n"
    content += "  # ========================================================================\n"
    content += f"  model_type: {values['model_type']}  # Options: Dress4D, PeopleSnapshot\n"
    content += f"  model: {values['model']}  # Options: hashgrid, hashgrid_vd, general, diffusion\n"
    content += f"  hidden_layers: {values['hidden_layers']}\n"
    content += f"  batch_size: {values['batch_size']}\n"
    content += f"  device: {values['device']}\n"
    content += "  \n"
    
    # Dataset Configuration
    content += "  # ========================================================================\n"
    content += "  # Dataset Configuration\n"
    content += "  # ========================================================================\n"
    content += f"  start_frame: {values['start_frame']}\n"
    content += f"  end_frame: {values['end_frame']}\n"
    content += f"  skip_frames: {values['skip_frames']}\n"
    content += f"  image_size: {values['image_size']}\n"
    content += "  \n"
    
    # Input Paths
    content += "  # ========================================================================\n"
    content += "  # Input Paths\n"
    content += "  # ========================================================================\n"
    content += "  # SMPL Model and Fittings\n"
    content += f"  smpl_path: {values['smpl_path']}\n"
    content += f"  smpl_pkl: {values['smpl_pkl']}\n"
    template_smpl = values.get('template_smpl_pkl')
    if template_smpl:
        content += f"  template_smpl_pkl: {template_smpl}  # Set to null if not using custom template\n"
    else:
        content += f"  template_smpl_pkl: null  # Set to null if not using custom template\n"
    content += f"  custom_template: {str(values['custom_template']).lower()}\n"
    content += "  \n"
    content += "  # Mesh Template\n"
    content += f"  mesh: {values['mesh']}\n"
    content += "  \n"
    content += "  # Target Images and Masks\n"
    content += f"  target_images: {values['target_images']}\n"
    content += f"  target_diffuse_maps: {values['target_diffuse_maps']}\n"
    content += f"  target_shil_maps: {values['target_shil_maps']}\n"
    content += f"  target_shil_seg_maps: {values['target_shil_seg_maps']}\n"
    content += f"  target_complete_shil_maps: {values['target_complete_shil_maps']}\n"
    content += f"  target_hand_mask: {values['target_hand_mask']}\n"
    depth = values.get('target_depth')
    if depth:
        content += f"  target_depth: {depth}\n"
    else:
        content += f"  target_depth: null  # Set to null if not available\n"
    normal = values.get('target_normal')
    if normal:
        content += f"  target_normal: {normal}\n"
    else:
        content += f"  target_normal: null  # Set to null if not available\n"
    content += "  \n"
    
    # Training Configuration
    content += "  # ========================================================================\n"
    content += "  # Training Configuration\n"
    content += "  # ========================================================================\n"
    content += f"  num_epochs: {values['num_epochs']}\n"
    content += f"  warm_ups: {values['warm_ups']}\n"
    content += f"  warm_ups_remesh: {values['warm_ups_remesh']}\n"
    content += f"  remesh_freq: {values['remesh_freq']}\n"
    content += "  \n"
    content += "  # Learning Rate Schedules\n"
    content += f"  position_lr_init: {values['position_lr_init']}\n"
    content += f"  position_lr_final: {values['position_lr_final']}\n"
    content += f"  position_lr_delay_mult: {values['position_lr_delay_mult']}\n"
    content += f"  position_lr_max_steps: {values['position_lr_max_steps']}\n"
    content += f"  deform_lr_max_steps: {values['deform_lr_max_steps']}\n"
    content += f"  custom_training_cannonical: {str(values['custom_training_cannonical']).lower()}\n"
    content += "  \n"
    content += "  # Data Augmentation\n"
    content += f"  pose_noise: {str(values['pose_noise']).lower()}\n"
    content += f"  noise_level: {values['noise_level']}\n"
    content += f"  vertex_noise: {str(values['vertex_noise']).lower()}\n"
    content += "  \n"
    content += "  # Optimization\n"
    content += f"  skinning_func: {values['skinning_func']}  # Options: rbf, k-near\n"
    content += f"  gradient_clipping: {str(values['gradient_clipping']).lower()}\n"
    content += f"  stop_train_after_warmup: {str(values['stop_train_after_warmup']).lower()}\n"
    content += "  \n"
    
    # Remeshing Configuration
    content += "  # ========================================================================\n"
    content += "  # Remeshing Configuration\n"
    content += "  # ========================================================================\n"
    content += f"  remeshing: {str(values['remeshing']).lower()}\n"
    content += f"  remesh_stop: {values['remesh_stop']}\n"
    content += f"  remesh_percentile: {values['remesh_percentile']}\n"
    content += f"  max_verts: {values['max_verts']}\n"
    content += f"  start_edge_len: {values['start_edge_len']}\n"
    content += f"  end_edge_len: {values['end_edge_len']}\n"
    content += f"  start_percentile: {values['start_percentile']}\n"
    content += f"  end_percentile: {values['end_percentile']}\n"
    content += f"  retriangulate: {values['retriangulate']}\n"
    content += f"  create_new_optimizer: {str(values['create_new_optimizer']).lower()}\n"
    content += f"  interpolate_new_optimizer: {str(values['interpolate_new_optimizer']).lower()}\n"
    content += "  \n"
    
    # Loss Configuration
    content += "  # ========================================================================\n"
    content += "  # Loss Configuration\n"
    content += "  # ========================================================================\n"
    content += "  losses:\n"
    losses = values['losses']
    content += f"    RenderingLoss: {losses.get('RenderingLoss', 5.0)}\n"
    content += f"    Regularization: {losses.get('Regularization', 0.0)}\n"
    content += f"    StVKLoss: {losses.get('StVKLoss', 1.0)}\n"
    content += f"    BendingLoss: {losses.get('BendingLoss', 1.0)}\n"
    depth_loss = losses.get('DepthDistillation', 1.0)
    if depth is None or str(depth).lower() == 'null':
        content += f"    DepthDistillation: 0.0  # Disabled when target_depth is null\n"
    else:
        content += f"    DepthDistillation: {depth_loss}\n"
    content += "  \n"
    
    # Logging Configuration
    content += "  # ========================================================================\n"
    content += "  # Logging Configuration\n"
    content += "  # ========================================================================\n"
    content += f"  logging: {str(values['logging']).lower()}\n"
    content += f"  log_interval: {values['log_interval']}\n"
    content += f"  log_interval_im: {values['log_interval_im']}\n"
    content += f"  log_elev: {values['log_elev']}\n"
    content += f"  log_fov: {values['log_fov']}\n"
    content += f"  log_dist: {values['log_dist']}\n"
    content += f"  log_res: {values['log_res']}\n"
    content += f"  log_light_power: {values['log_light_power']}\n"
    content += "  \n"
    
    # Output Configuration
    content += "  # ========================================================================\n"
    content += "  # Output Configuration\n"
    content += "  # ========================================================================\n"
    content += f"  output_path: {values['output_path']}\n"
    content += f"  Exp_name: {values['Exp_name']}\n"
    content += f"  save_instance: {str(values['save_instance']).lower()}\n"
    content += f"  save_index: {values['save_index']}\n"
    content += f"  save_freq: {values['save_freq']}\n"
    content += "  \n"
    
    # Texture Configuration
    content += "  # ========================================================================\n"
    content += "  # Texture Configuration\n"
    content += "  # ========================================================================\n"
    content += f"  texture_recon: {str(values['texture_recon']).lower()}\n"
    content += f"  texture_model: {values['texture_model']}  # Options: hashgrid_base, hashgrid_reg\n"
    texture_map = values['texture_map']
    if isinstance(texture_map, list):
        content += f"  texture_map: [{texture_map[0]}, {texture_map[1]}]\n"
    else:
        content += f"  texture_map: {texture_map}\n"
    content += f"  max_mip_level: {values['max_mip_level']}\n"
    content += f"  tex_epochs: {values['tex_epochs']}\n"
    content += f"  texture_warm_ups: {values['texture_warm_ups']}\n"
    content += f"  use_dynamic_texture: {str(values['use_dynamic_texture']).lower()}\n"
    content += f"  texture_lr_init: {values['texture_lr_init']}\n"
    content += f"  texture_lr_final: {values['texture_lr_final']}\n"
    content += f"  texture_lr_delay_mult: {values['texture_lr_delay_mult']}\n"
    content += f"  texture_lr_max_steps: {values['texture_lr_max_steps']}\n"
    content += f"  spatial_lr_scale: {values['spatial_lr_scale']}\n"
    content += f"  ssim_coeff: {values['ssim_coeff']}\n"
    content += f"  texture_lr_base: {values['texture_lr_base']}\n"
    content += f"  texture_lr_base_dyn: {values['texture_lr_base_dyn']}\n"
    
    return content


def standardize_config_file(config_path: Path) -> bool:
    """Standardize a single config file."""
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        if not cfg:
            print(f"Warning: Empty config file: {config_path}")
            return False
        
        # Get dataset info from path
        dataset_type, garment_type = get_dataset_info_from_path(config_path)
        
        # Extract and normalize values
        values = extract_config_values(cfg)
        
        # Add global settings
        values['gpu'] = cfg.get('gpu', 0)
        values['seed'] = cfg.get('seed', 99)
        
        # Format the config
        formatted_content = format_config_file(config_path, values, dataset_type, garment_type)
        
        # Write back
        with open(config_path, 'w') as f:
            f.write(formatted_content)
        
        print(f"✓ Standardized: {config_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error processing {config_path}: {e}")
        return False


if __name__ == '__main__':
    import sys
    
    config_dir = Path(__file__).parent.parent / "configs"
    
    if len(sys.argv) > 1:
        # Process specific file
        config_path = Path(sys.argv[1])
        standardize_config_file(config_path)
    else:
        # Process all config files
        config_files = list(config_dir.rglob("*.yml")) + list(config_dir.rglob("*.yaml"))
        config_files = [f for f in config_files if f.name != "config_template.yml"]
        
        print(f"Found {len(config_files)} config files to standardize...\n")
        
        success_count = 0
        for config_file in sorted(config_files):
            if standardize_config_file(config_file):
                success_count += 1
        
        print(f"\n✓ Standardized {success_count}/{len(config_files)} config files")

