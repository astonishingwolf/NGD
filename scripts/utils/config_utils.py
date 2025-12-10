"""
Configuration utility functions for loading and managing config files.
"""
import json
from pathlib import Path
from typing import Dict, Any

# Base paths for config files
_CONFIG_BASE_DIR = Path(__file__).parent.parent.parent
_CONFIG_DIR = _CONFIG_BASE_DIR / "scripts" / "configs"


def load_json_config(config_name: str) -> Dict[str, Any]:
    """
    Load a JSON configuration file from the configs directory.
    
    Args:
        config_name: Name of the config file (e.g., "config_hash.json")
        
    Returns:
        Dictionary containing the configuration
    """
    config_path = _CONFIG_DIR / config_name
    if not config_path.exists():
        # Fallback paths for backward compatibility
        fallback_paths = [
            _CONFIG_BASE_DIR / "scripts" / config_name,
            _CONFIG_BASE_DIR / "scripts" / "models" / "config" / config_name
        ]
        for fallback_path in fallback_paths:
            if fallback_path.exists():
                config_path = fallback_path
                break
        else:
            raise FileNotFoundError(f"Config file not found: {config_name}")
    
    with open(config_path, 'r') as f:
        return json.load(f)


def load_hash_config() -> Dict[str, Any]:
    """
    Load the hash grid configuration.
    
    Returns:
        Dictionary containing hash grid configuration
    """
    return load_json_config("config_hash.json")


def load_texture_config() -> Dict[str, Any]:
    """
    Load the texture model configuration.
    
    Returns:
        Dictionary containing texture model configuration
    """
    return load_json_config("config_tex.json")


def get_config_path(config_name: str) -> Path:
    """
    Get the full path to a configuration file.
    
    Args:
        config_name: Name of the config file
        
    Returns:
        Path object to the config file
    """
    config_path = _CONFIG_DIR / config_name
    if not config_path.exists():
        # Fallback paths for backward compatibility
        fallback_paths = [
            _CONFIG_BASE_DIR / "scripts" / config_name,
            _CONFIG_BASE_DIR / "scripts" / "models" / "config" / config_name
        ]
        for fallback_path in fallback_paths:
            if fallback_path.exists():
                return fallback_path
        raise FileNotFoundError(f"Config file not found: {config_name}")
    return config_path

