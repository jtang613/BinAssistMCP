"""
Utility functions for BinAssist-MCP

This module provides common utility functions used across the project.
"""

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import binaryninja as bn
    BINJA_AVAILABLE = True
except ImportError:
    BINJA_AVAILABLE = False


def extract_binary_name(binary_view) -> str:
    """Extract a clean name from a Binary Ninja BinaryView
    
    Args:
        binary_view: Binary Ninja BinaryView object
        
    Returns:
        Clean name string suitable for use as an identifier
    """
    if not BINJA_AVAILABLE or not binary_view:
        return "unknown"
        
    try:
        # Try to get filename from the file object
        if hasattr(binary_view, 'file') and hasattr(binary_view.file, 'filename'):
            filename = binary_view.file.filename
            if filename:
                return Path(filename).name
                
        # Try to get name directly from binary view
        if hasattr(binary_view, 'name') and binary_view.name:
            return binary_view.name
            
        # Try to get name from file object
        if hasattr(binary_view, 'file') and hasattr(binary_view.file, 'original_filename'):
            original_filename = binary_view.file.original_filename
            if original_filename:
                return Path(original_filename).name
                
    except Exception as e:
        logger.debug(f"Failed to extract name from binary view: {e}")
        
    return "unknown"


def sanitize_identifier(name: str) -> str:
    """Sanitize a name to be safe for use as an identifier
    
    Args:
        name: Input name string
        
    Returns:
        Sanitized name safe for use in URLs, filenames, etc.
    """
    if not name or not isinstance(name, str):
        return "unnamed"
        
    # Replace problematic characters
    invalid_chars = '/\\:*?"<>| \t\n\r'
    for char in invalid_chars:
        name = name.replace(char, '_')
        
    # Remove leading/trailing dots and underscores
    name = name.strip('_.')
    
    # Ensure non-empty result
    if not name:
        name = "unnamed"
        
    # Ensure it starts with a letter or underscore (identifier rules)
    if name and not (name[0].isalpha() or name[0] == '_'):
        name = f"bin_{name}"
        
    return name


def format_address(address: int, width: Optional[int] = None) -> str:
    """Format an address as a hex string
    
    Args:
        address: Integer address
        width: Optional width for zero-padding
        
    Returns:
        Formatted hex address string
    """
    if width:
        return f"0x{address:0{width}x}"
    else:
        return f"0x{address:x}"


def parse_address(address_str: str) -> Optional[int]:
    """Parse an address string to integer
    
    Args:
        address_str: Address string (hex or decimal)
        
    Returns:
        Integer address or None if parsing fails
    """
    if not address_str:
        return None
        
    try:
        # Try hex format first
        if address_str.startswith('0x') or address_str.startswith('0X'):
            return int(address_str, 16)
            
        # Try decimal format
        if address_str.isdigit():
            return int(address_str)
            
        # Try pure hex without prefix
        return int(address_str, 16)
        
    except ValueError:
        logger.debug(f"Failed to parse address: {address_str}")
        return None


def format_size(size_bytes: int) -> str:
    """Format a size in bytes to human-readable format
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Human-readable size string
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"


def truncate_string(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate a string to a maximum length
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add when truncating
        
    Returns:
        Truncated string
    """
    if not text or len(text) <= max_length:
        return text
        
    return text[:max_length - len(suffix)] + suffix


def safe_get_attribute(obj, attr_path: str, default=None):
    """Safely get a nested attribute from an object
    
    Args:
        obj: Object to get attribute from
        attr_path: Dot-separated attribute path (e.g., "file.filename")
        default: Default value if attribute not found
        
    Returns:
        Attribute value or default
    """
    try:
        attrs = attr_path.split('.')
        result = obj
        
        for attr in attrs:
            if hasattr(result, attr):
                result = getattr(result, attr)
            else:
                return default
                
        return result
        
    except Exception:
        return default


def validate_binary_view(binary_view) -> bool:
    """Validate that a binary view is usable
    
    Args:
        binary_view: Binary Ninja BinaryView object
        
    Returns:
        True if valid, False otherwise
    """
    if not BINJA_AVAILABLE or not binary_view:
        return False
        
    try:
        # Try to access basic properties
        _ = binary_view.file
        _ = binary_view.start
        _ = binary_view.end
        return True
        
    except Exception as e:
        logger.debug(f"Binary view validation failed: {e}")
        return False


def ensure_absolute_path(path: str) -> Path:
    """Ensure a path is absolute
    
    Args:
        path: Input path string
        
    Returns:
        Absolute Path object
    """
    path_obj = Path(path)
    if path_obj.is_absolute():
        return path_obj
    else:
        return Path.cwd() / path_obj