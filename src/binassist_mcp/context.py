"""
Binary context management for BinAssist-MCP

This module provides context management for multiple Binary Ninja BinaryViews
with automatic name deduplication and lifecycle management.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import binaryninja as bn
    BINJA_AVAILABLE = True
except ImportError:
    BINJA_AVAILABLE = False
    logger.warning("Binary Ninja not available")


@dataclass
class BinaryInfo:
    """Information about a loaded binary"""
    name: str
    view: Optional[object]  # bn.BinaryView when available
    file_path: Optional[Path] = None
    load_time: Optional[float] = None
    analysis_complete: bool = False
    
    def __post_init__(self):
        if self.file_path and isinstance(self.file_path, str):
            self.file_path = Path(self.file_path)


class BinAssistMCPBinaryContextManager:
    """Context manager for multiple Binary Ninja BinaryViews"""
    
    def __init__(self, max_binaries: int = 10):
        """Initialize the context manager
        
        Args:
            max_binaries: Maximum number of binaries to keep loaded
        """
        self.max_binaries = max_binaries
        self._binaries: Dict[str, BinaryInfo] = {}
        self._name_counter: Dict[str, int] = {}
        
    def add_binary(self, binary_view: object, name: Optional[str] = None) -> str:
        """Add a BinaryView to the context with automatic name deduplication
        
        Args:
            binary_view: The BinaryView to add
            name: Optional name to use (defaults to filename)
            
        Returns:
            The name used for the BinaryView
        """
        if not BINJA_AVAILABLE:
            raise RuntimeError("Binary Ninja not available")
            
        if name is None:
            name = self._extract_name(binary_view)
            
        # Sanitize name for URL usage
        sanitized_name = self._sanitize_name(name)
        
        # Deduplicate name if needed
        unique_name = self._get_unique_name(sanitized_name)
        
        # Check if we need to evict old binaries
        if len(self._binaries) >= self.max_binaries:
            self._evict_oldest_binary()
            
        # Add binary info
        import time
        binary_info = BinaryInfo(
            name=unique_name,
            view=binary_view,
            file_path=self._get_file_path(binary_view),
            load_time=time.time(),
            analysis_complete=self._is_analysis_complete(binary_view)
        )
        
        self._binaries[unique_name] = binary_info
        logger.info(f"Added binary '{unique_name}' to context (total: {len(self._binaries)})")
        
        return unique_name
        
    def get_binary(self, name: str) -> object:
        """Get a BinaryView by name
        
        Args:
            name: The name of the BinaryView
            
        Returns:
            The BinaryView if found
            
        Raises:
            KeyError: If the binary is not found
        """
        if name not in self._binaries:
            available = ", ".join(self._binaries.keys()) if self._binaries else "none"
            raise KeyError(f"Binary '{name}' not found. Available: {available}")
            
        binary_info = self._binaries[name]
        
        # Verify the binary view is still valid
        if not self._is_binary_valid(binary_info.view):
            logger.warning(f"Binary '{name}' is no longer valid, removing from context")
            del self._binaries[name]
            raise KeyError(f"Binary '{name}' is no longer valid")
            
        return binary_info.view
        
    def get_binary_info(self, name: str) -> BinaryInfo:
        """Get binary information by name
        
        Args:
            name: The name of the binary
            
        Returns:
            BinaryInfo object
            
        Raises:
            KeyError: If the binary is not found
        """
        if name not in self._binaries:
            available = ", ".join(self._binaries.keys()) if self._binaries else "none"
            raise KeyError(f"Binary '{name}' not found. Available: {available}")
            
        return self._binaries[name]
        
    def list_binaries(self) -> List[str]:
        """List all loaded binary names
        
        Returns:
            List of binary names
        """
        return list(self._binaries.keys())
        
    def list_binary_info(self) -> Dict[str, BinaryInfo]:
        """Get information about all loaded binaries
        
        Returns:
            Dictionary mapping names to BinaryInfo objects
        """
        return self._binaries.copy()
        
    def remove_binary(self, name: str) -> bool:
        """Remove a binary from the context
        
        Args:
            name: Name of the binary to remove
            
        Returns:
            True if removed, False if not found
        """
        if name in self._binaries:
            del self._binaries[name]
            logger.info(f"Removed binary '{name}' from context")
            return True
        return False
        
    def clear(self):
        """Clear all binaries from the context"""
        count = len(self._binaries)
        self._binaries.clear()
        self._name_counter.clear()
        logger.info(f"Cleared {count} binaries from context")
        
    def update_analysis_status(self, name: str):
        """Update the analysis status for a binary
        
        Args:
            name: Name of the binary to update
        """
        if name in self._binaries:
            binary_info = self._binaries[name]
            if binary_info.view:
                binary_info.analysis_complete = self._is_analysis_complete(binary_info.view)
                logger.debug(f"Updated analysis status for '{name}': {binary_info.analysis_complete}")
                
    def _extract_name(self, binary_view: object) -> str:
        """Extract name from a BinaryView"""
        if not BINJA_AVAILABLE or not binary_view:
            return "unknown"
            
        try:
            if hasattr(binary_view, 'file') and hasattr(binary_view.file, 'filename'):
                filename = binary_view.file.filename
                if filename:
                    return Path(filename).name
                    
            if hasattr(binary_view, 'name'):
                return binary_view.name
                
        except Exception as e:
            logger.warning(f"Failed to extract name from binary view: {e}")
            
        return "unknown"
        
    def _sanitize_name(self, name: str) -> str:
        """Sanitize name for URL usage"""
        if not name:
            return "unnamed"
            
        # Replace invalid characters
        invalid_chars = '/\\:*?"<>| '
        for char in invalid_chars:
            name = name.replace(char, '_')
            
        # Remove leading/trailing dots and underscores
        name = name.strip('_.')
        
        # Ensure non-empty name
        if not name:
            name = "unnamed"
            
        return name
        
    def _get_unique_name(self, base_name: str) -> str:
        """Get a unique name by adding a counter if needed"""
        if base_name not in self._binaries:
            return base_name
            
        # Find the next available counter value
        counter = self._name_counter.get(base_name, 1)
        while True:
            unique_name = f"{base_name}_{counter}"
            if unique_name not in self._binaries:
                self._name_counter[base_name] = counter + 1
                return unique_name
            counter += 1
            
    def _get_file_path(self, binary_view: object) -> Optional[Path]:
        """Get file path from a BinaryView"""
        if not BINJA_AVAILABLE or not binary_view:
            return None
            
        try:
            if hasattr(binary_view, 'file') and hasattr(binary_view.file, 'filename'):
                filename = binary_view.file.filename
                if filename:
                    return Path(filename)
        except Exception as e:
            logger.debug(f"Failed to get file path: {e}")
            
        return None
        
    def _is_analysis_complete(self, binary_view: object) -> bool:
        """Check if analysis is complete for a BinaryView"""
        if not BINJA_AVAILABLE or not binary_view:
            return False
            
        try:
            if hasattr(binary_view, 'analysis_progress'):
                progress = binary_view.analysis_progress
                return progress.state == progress.state.AnalysisStateInactive
                
            # Fallback: check if we have functions
            if hasattr(binary_view, 'functions'):
                return len(list(binary_view.functions)) > 0
                
        except Exception as e:
            logger.debug(f"Failed to check analysis status: {e}")
            
        return False
        
    def _is_binary_valid(self, binary_view: object) -> bool:
        """Check if a BinaryView is still valid"""
        if not BINJA_AVAILABLE or not binary_view:
            return False
            
        try:
            # Try to access a basic property
            if hasattr(binary_view, 'file'):
                _ = binary_view.file
                return True
        except Exception as e:
            logger.debug(f"Binary view validation failed: {e}")
            
        return False
        
    def _evict_oldest_binary(self):
        """Evict the oldest binary to make room for a new one"""
        if not self._binaries:
            return
            
        # Find the binary with the oldest load time
        oldest_name = None
        oldest_time = float('inf')
        
        for name, binary_info in self._binaries.items():
            if binary_info.load_time and binary_info.load_time < oldest_time:
                oldest_time = binary_info.load_time
                oldest_name = name
                
        if oldest_name:
            logger.info(f"Evicting oldest binary '{oldest_name}' to make room")
            del self._binaries[oldest_name]
            
    def __len__(self) -> int:
        """Return the number of loaded binaries"""
        return len(self._binaries)
        
    def __contains__(self, name: str) -> bool:
        """Check if a binary name is in the context"""
        return name in self._binaries
        
    def __repr__(self) -> str:
        """String representation of the context manager"""
        return f"BinaryContextManager(binaries={len(self._binaries)}, max={self.max_binaries})"