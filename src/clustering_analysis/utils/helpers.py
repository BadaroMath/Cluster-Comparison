"""Utility helper functions."""

import json
import yaml
from pathlib import Path
from typing import Any, Dict
from loguru import logger
import sys


def setup_logging(level: str = "INFO", log_file: str = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level
        log_file: Optional log file path
    """
    logger.remove()  # Remove default handler
    
    # Console handler
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True
    )
    
    # File handler if specified
    if log_file:
        logger.add(
            log_file,
            level=level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB"
        )


def ensure_dir(path: Path) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
        
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_yaml(data: Dict[str, Any], filepath: str) -> None:
    """
    Save data to YAML file.
    
    Args:
        data: Data to save
        filepath: Output file path
    """
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def load_yaml(filepath: str) -> Dict[str, Any]:
    """
    Load data from YAML file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r') as f:
        return yaml.safe_load(f)