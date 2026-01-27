"""
Binary context management for BinAssistMCP

This module provides context management for multiple Binary Ninja BinaryViews
with automatic name deduplication and lifecycle management.
"""

import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path

from .logging import log

try:
    import binaryninja as bn
    from binaryninja import AnalysisState
    BINJA_AVAILABLE = True
except ImportError:
    BINJA_AVAILABLE = False
    AnalysisState = None
    log.log_warn("Binary Ninja not available")


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
        self._lock = threading.RLock()  # Thread safety for binary operations
        
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

        with self._lock:
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
            log.log_info(f"Added binary '{unique_name}' to context (total: {len(self._binaries)})")

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
        with self._lock:
            if name not in self._binaries:
                available = ", ".join(self._binaries.keys()) if self._binaries else "none"
                raise KeyError(f"Binary '{name}' not found. Available: {available}")

            binary_info = self._binaries[name]

            # Verify the binary view is still valid
            if not self._is_binary_valid(binary_info.view):
                log.log_warn(f"Binary '{name}' is no longer valid, removing from context")
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
        with self._lock:
            if name not in self._binaries:
                available = ", ".join(self._binaries.keys()) if self._binaries else "none"
                raise KeyError(f"Binary '{name}' not found. Available: {available}")

            return self._binaries[name]

    def list_binaries(self) -> List[str]:
        """List all loaded binary names

        Returns:
            List of binary names
        """
        with self._lock:
            return list(self._binaries.keys())

    def list_binary_info(self) -> Dict[str, BinaryInfo]:
        """Get information about all loaded binaries

        Returns:
            Dictionary mapping names to BinaryInfo objects
        """
        with self._lock:
            return self._binaries.copy()

    def remove_binary(self, name: str) -> bool:
        """Remove a binary from the context

        Args:
            name: Name of the binary to remove

        Returns:
            True if removed, False if not found
        """
        with self._lock:
            if name in self._binaries:
                del self._binaries[name]
                log.log_info(f"Removed binary '{name}' from context")
                return True
            return False

    def clear(self):
        """Clear all binaries from the context"""
        with self._lock:
            count = len(self._binaries)
            self._binaries.clear()
            self._name_counter.clear()
            log.log_info(f"Cleared {count} binaries from context")

    def update_analysis_status(self, name: str):
        """Update the analysis status for a binary

        Args:
            name: Name of the binary to update
        """
        with self._lock:
            if name in self._binaries:
                binary_info = self._binaries[name]
                if binary_info.view:
                    binary_info.analysis_complete = self._is_analysis_complete(binary_info.view)
                    log.log_debug(f"Updated analysis status for '{name}': {binary_info.analysis_complete}")
                
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
            log.log_warn(f"Failed to extract name from binary view: {e}")
            
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
            log.log_debug(f"Failed to get file path: {e}")
            
        return None
        
    def _is_analysis_complete(self, binary_view: object) -> bool:
        """Check if analysis is complete for a BinaryView"""
        if not BINJA_AVAILABLE or not binary_view or not AnalysisState:
            return False
            
        try:
            # Method 1: Check analysis_progress state
            if hasattr(binary_view, 'analysis_progress'):
                progress = binary_view.analysis_progress
                current_state = progress.state
                log.log_debug(f"Analysis progress state: {current_state} (IdleState={AnalysisState.IdleState})")
                # Correct API: compare state directly to AnalysisState.IdleState
                return current_state == AnalysisState.IdleState
                
            # Method 2: Check analysis_info state (alternative)
            if hasattr(binary_view, 'analysis_info'):
                info_state = binary_view.analysis_info.state
                log.log_debug(f"Analysis info state: {info_state} (IdleState={AnalysisState.IdleState})")
                return info_state == AnalysisState.IdleState
                
            # Fallback: check if we have functions
            if hasattr(binary_view, 'functions'):
                func_count = len(list(binary_view.functions))
                log.log_debug(f"Analysis status fallback: {func_count} functions found")
                return func_count > 0
                
        except Exception as e:
            log.log_debug(f"Failed to check analysis status: {e}")
            # Additional debug info
            try:
                if hasattr(binary_view, 'analysis_progress'):
                    progress = binary_view.analysis_progress
                    log.log_debug(f"Progress object type: {type(progress)}")
                    log.log_debug(f"Progress state type: {type(progress.state)}")
                    log.log_debug(f"Available AnalysisState values: {[attr for attr in dir(AnalysisState) if not attr.startswith('_')]}")
            except Exception as debug_error:
                log.log_debug(f"Failed to get debug info: {debug_error}")
            
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
            log.log_debug(f"Binary view validation failed: {e}")
            
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
            log.log_info(f"Evicting oldest binary '{oldest_name}' to make room")
            del self._binaries[oldest_name]
            
    def __len__(self) -> int:
        """Return the number of loaded binaries"""
        with self._lock:
            return len(self._binaries)

    def __contains__(self, name: str) -> bool:
        """Check if a binary name is in the context"""
        with self._lock:
            return name in self._binaries

    def __repr__(self) -> str:
        """String representation of the context manager"""
        with self._lock:
            return f"BinaryContextManager(binaries={len(self._binaries)}, max={self.max_binaries})"

    def sync_with_binja(self) -> dict:
        """Synchronize context with Binary Ninja's currently open views.

        Enumerates all open BinaryViews via Binary Ninja UI context,
        adds newly opened binaries to context, and removes closed/invalid
        binaries from context.

        Returns:
            Dictionary with sync status report:
            - added: list of newly added binary names
            - removed: list of removed binary names
            - unchanged: list of binaries that remained
            - synced: bool indicating if sync was performed
            - error: optional error message if sync failed
        """
        result = {
            "added": [],
            "removed": [],
            "unchanged": [],
            "synced": False,
            "error": None
        }

        if not BINJA_AVAILABLE:
            result["error"] = "Binary Ninja not available"
            return result

        # Try to access UI context for open views
        try:
            from binaryninjaui import UIContext
            ui_available = True
        except ImportError:
            ui_available = False
            log.log_debug("binaryninjaui not available, running in headless mode")

        with self._lock:
            # First, remove invalid/closed binaries from context
            names_to_remove = []
            for name, binary_info in self._binaries.items():
                if not self._is_binary_valid(binary_info.view):
                    names_to_remove.append(name)

            for name in names_to_remove:
                del self._binaries[name]
                result["removed"].append(name)
                log.log_info(f"Removed invalid/closed binary '{name}' from context")

            # If UI is available, enumerate open views and add new ones
            if ui_available:
                try:
                    ctx = UIContext.activeContext()
                    if ctx is not None:
                        # Get all available binary views from the UI context
                        # Try different methods to get open views
                        open_views = []

                        # Method 1: getAvailableBinaryViews (preferred)
                        if hasattr(ctx, 'getAvailableBinaryViews'):
                            open_views = ctx.getAvailableBinaryViews()
                        # Method 2: getAllOpenBinaryViews
                        elif hasattr(ctx, 'getAllOpenBinaryViews'):
                            open_views = ctx.getAllOpenBinaryViews()
                        # Method 3: Iterate through view frames
                        elif hasattr(ctx, 'viewFrames'):
                            for frame in ctx.viewFrames():
                                if hasattr(frame, 'getCurrentBinaryView'):
                                    bv = frame.getCurrentBinaryView()
                                    if bv is not None:
                                        open_views.append(bv)

                        # Build a set of file paths currently in context for comparison
                        context_paths = set()
                        for binary_info in self._binaries.values():
                            if binary_info.file_path:
                                context_paths.add(str(binary_info.file_path))

                        # Add any open views not already in context
                        for bv in open_views:
                            if bv is None:
                                continue

                            file_path = self._get_file_path(bv)
                            file_path_str = str(file_path) if file_path else None

                            # Check if this view is already in context (by path)
                            if file_path_str and file_path_str in context_paths:
                                continue

                            # Check if view object is already tracked
                            already_tracked = False
                            for binary_info in self._binaries.values():
                                if binary_info.view is bv:
                                    already_tracked = True
                                    break

                            if already_tracked:
                                continue

                            # Add this new binary
                            try:
                                name = self._extract_name(bv)
                                sanitized_name = self._sanitize_name(name)
                                unique_name = self._get_unique_name(sanitized_name)

                                # Check if we need to evict old binaries
                                if len(self._binaries) >= self.max_binaries:
                                    self._evict_oldest_binary()

                                import time
                                binary_info = BinaryInfo(
                                    name=unique_name,
                                    view=bv,
                                    file_path=file_path,
                                    load_time=time.time(),
                                    analysis_complete=self._is_analysis_complete(bv)
                                )

                                self._binaries[unique_name] = binary_info
                                result["added"].append(unique_name)
                                log.log_info(f"Added newly opened binary '{unique_name}' to context")
                            except Exception as add_error:
                                log.log_warn(f"Failed to add binary view to context: {add_error}")
                    else:
                        log.log_debug("No active UI context available")
                except Exception as ui_error:
                    log.log_debug(f"Error accessing UI context: {ui_error}")
                    result["error"] = f"UI context error: {ui_error}"

            # Record unchanged binaries
            for name in self._binaries.keys():
                if name not in result["added"]:
                    result["unchanged"].append(name)

            result["synced"] = True
            log.log_debug(f"Sync complete: added={len(result['added'])}, removed={len(result['removed'])}, unchanged={len(result['unchanged'])}")

        return result