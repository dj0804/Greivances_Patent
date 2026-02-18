"""
Helper utility functions.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict


def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Load JSON file.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Parsed JSON data
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: Dict[str, Any], filepath: Path, indent: int = 2):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=indent)


def ensure_dir(directory: Path) -> Path:
    """
    Ensure directory exists.
    
    Args:
        directory: Directory path
        
    Returns:
        Directory path
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def get_timestamp(format: str = "%Y%m%d_%H%M%S") -> str:
    """
    Get current timestamp as string.
    
    Args:
        format: Datetime format string
        
    Returns:
        Formatted timestamp
    """
    return datetime.now().strftime(format)


def chunks(lst: list, n: int):
    """
    Yield successive n-sized chunks from list.
    
    Args:
        lst: Input list
        n: Chunk size
        
    Yields:
        List chunks
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
