"""
MCP Resources for BinAssistMCP

This module provides browsable, cacheable data resources for MCP clients.
Resources are accessed via URI patterns like binja://binary/{name}/info
"""

import json
from typing import Any, Dict, List, Optional

from .logging import log

try:
    import binaryninja as bn
    BINJA_AVAILABLE = True
except ImportError:
    BINJA_AVAILABLE = False


def get_binary_info_resource(binary_view) -> Dict[str, Any]:
    """Get comprehensive binary metadata.

    Args:
        binary_view: Binary Ninja BinaryView

    Returns:
        Dictionary with binary metadata
    """
    if not BINJA_AVAILABLE or not binary_view:
        return {"error": "Binary view not available"}

    try:
        info = {
            "filename": binary_view.file.filename if binary_view.file else None,
            "architecture": str(binary_view.arch) if binary_view.arch else None,
            "platform": str(binary_view.platform) if binary_view.platform else None,
            "entry_point": hex(binary_view.entry_point) if binary_view.entry_point else None,
            "start_address": hex(binary_view.start),
            "end_address": hex(binary_view.end),
            "length": binary_view.length,
            "address_size": binary_view.address_size,
            "executable": binary_view.executable,
            "relocatable": binary_view.relocatable,
            "view_type": binary_view.view_type,
            "modified": binary_view.modified,
            "analysis_progress": {
                "state": str(binary_view.analysis_progress.state) if hasattr(binary_view, 'analysis_progress') else "unknown"
            },
            "statistics": {
                "function_count": len(list(binary_view.functions)),
                "string_count": len(list(binary_view.strings)),
                "data_var_count": len(binary_view.data_vars),
                "segment_count": len(list(binary_view.segments)),
                "section_count": len(list(binary_view.sections))
            }
        }

        # Add symbol counts
        try:
            symbols = list(binary_view.symbols.values())
            info["statistics"]["symbol_count"] = len(symbols)
        except Exception:
            info["statistics"]["symbol_count"] = 0

        return info

    except Exception as e:
        log.log_error(f"Error getting binary info resource: {e}")
        return {"error": str(e)}


def get_functions_resource(binary_view, page: int = 1, page_size: int = 100) -> Dict[str, Any]:
    """Get paginated function list.

    Args:
        binary_view: Binary Ninja BinaryView
        page: Page number (1-indexed)
        page_size: Functions per page

    Returns:
        Dictionary with functions and pagination info
    """
    if not BINJA_AVAILABLE or not binary_view:
        return {"error": "Binary view not available"}

    try:
        all_functions = list(binary_view.functions)
        total_count = len(all_functions)
        total_pages = (total_count + page_size - 1) // page_size

        # Get page slice
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_count)
        page_functions = all_functions[start_idx:end_idx]

        functions = []
        for func in page_functions:
            functions.append({
                "name": func.name,
                "address": hex(func.start),
                "size": func.total_bytes if hasattr(func, 'total_bytes') else None,
                "parameter_count": len(func.parameter_vars) if func.parameter_vars else 0,
                "return_type": str(func.return_type) if func.return_type else None
            })

        return {
            "functions": functions,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        }

    except Exception as e:
        log.log_error(f"Error getting functions resource: {e}")
        return {"error": str(e)}


def get_strings_resource(binary_view, page: int = 1, page_size: int = 100) -> Dict[str, Any]:
    """Get paginated string table.

    Args:
        binary_view: Binary Ninja BinaryView
        page: Page number (1-indexed)
        page_size: Strings per page

    Returns:
        Dictionary with strings and pagination info
    """
    if not BINJA_AVAILABLE or not binary_view:
        return {"error": "Binary view not available"}

    try:
        all_strings = list(binary_view.strings)
        total_count = len(all_strings)
        total_pages = (total_count + page_size - 1) // page_size

        # Get page slice
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_count)
        page_strings = all_strings[start_idx:end_idx]

        strings = []
        for string in page_strings:
            strings.append({
                "address": hex(string.start),
                "value": string.value,
                "length": string.length,
                "type": str(string.type)
            })

        return {
            "strings": strings,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        }

    except Exception as e:
        log.log_error(f"Error getting strings resource: {e}")
        return {"error": str(e)}


def get_imports_resource(binary_view) -> Dict[str, Any]:
    """Get imported symbols grouped by module.

    Args:
        binary_view: Binary Ninja BinaryView

    Returns:
        Dictionary with imports grouped by module
    """
    if not BINJA_AVAILABLE or not binary_view:
        return {"error": "Binary view not available"}

    try:
        imports_by_module: Dict[str, List[Dict]] = {}

        for addr, sym_list in binary_view.symbols.items():
            for sym in sym_list if isinstance(sym_list, list) else [sym_list]:
                if sym.type in (bn.SymbolType.ImportedFunctionSymbol,
                               bn.SymbolType.ImportedDataSymbol,
                               bn.SymbolType.ImportAddressSymbol):
                    # Try to get module name from namespace
                    module = sym.namespace.name if sym.namespace else "unknown"
                    if module not in imports_by_module:
                        imports_by_module[module] = []

                    imports_by_module[module].append({
                        "name": sym.name,
                        "address": hex(sym.address),
                        "type": str(sym.type)
                    })

        return {
            "imports": imports_by_module,
            "module_count": len(imports_by_module),
            "total_imports": sum(len(v) for v in imports_by_module.values())
        }

    except Exception as e:
        log.log_error(f"Error getting imports resource: {e}")
        return {"error": str(e)}


def get_exports_resource(binary_view) -> Dict[str, Any]:
    """Get exported symbols.

    Args:
        binary_view: Binary Ninja BinaryView

    Returns:
        Dictionary with exported symbols
    """
    if not BINJA_AVAILABLE or not binary_view:
        return {"error": "Binary view not available"}

    try:
        exports = []

        for addr, sym_list in binary_view.symbols.items():
            for sym in sym_list if isinstance(sym_list, list) else [sym_list]:
                if sym.type in (bn.SymbolType.FunctionSymbol,
                               bn.SymbolType.DataSymbol,
                               bn.SymbolType.ExternalSymbol):
                    # Check if it's exported
                    if sym.binding == bn.SymbolBinding.GlobalBinding:
                        exports.append({
                            "name": sym.name,
                            "address": hex(sym.address),
                            "type": str(sym.type)
                        })

        return {
            "exports": exports,
            "count": len(exports)
        }

    except Exception as e:
        log.log_error(f"Error getting exports resource: {e}")
        return {"error": str(e)}


def get_segments_resource(binary_view) -> Dict[str, Any]:
    """Get memory segments.

    Args:
        binary_view: Binary Ninja BinaryView

    Returns:
        Dictionary with segment information
    """
    if not BINJA_AVAILABLE or not binary_view:
        return {"error": "Binary view not available"}

    try:
        segments = []

        for segment in binary_view.segments:
            segments.append({
                "start": hex(segment.start),
                "end": hex(segment.end),
                "length": segment.length,
                "data_length": segment.data_length,
                "readable": segment.readable,
                "writable": segment.writable,
                "executable": segment.executable
            })

        return {
            "segments": segments,
            "count": len(segments)
        }

    except Exception as e:
        log.log_error(f"Error getting segments resource: {e}")
        return {"error": str(e)}


def get_sections_resource(binary_view) -> Dict[str, Any]:
    """Get binary sections.

    Args:
        binary_view: Binary Ninja BinaryView

    Returns:
        Dictionary with section information
    """
    if not BINJA_AVAILABLE or not binary_view:
        return {"error": "Binary view not available"}

    try:
        sections = []

        for section in binary_view.sections.values():
            sections.append({
                "name": section.name,
                "start": hex(section.start),
                "end": hex(section.end),
                "length": section.length,
                "type": str(section.type) if hasattr(section, 'type') else None,
                "semantics": str(section.semantics) if hasattr(section, 'semantics') else None
            })

        return {
            "sections": sections,
            "count": len(sections)
        }

    except Exception as e:
        log.log_error(f"Error getting sections resource: {e}")
        return {"error": str(e)}


def get_data_vars_resource(binary_view, page: int = 1, page_size: int = 100) -> Dict[str, Any]:
    """Get paginated data variables.

    Args:
        binary_view: Binary Ninja BinaryView
        page: Page number (1-indexed)
        page_size: Variables per page

    Returns:
        Dictionary with data variables and pagination info
    """
    if not BINJA_AVAILABLE or not binary_view:
        return {"error": "Binary view not available"}

    try:
        all_vars = list(binary_view.data_vars.items())
        total_count = len(all_vars)
        total_pages = (total_count + page_size - 1) // page_size

        # Get page slice
        start_idx = (page - 1) * page_size
        end_idx = min(start_idx + page_size, total_count)
        page_vars = all_vars[start_idx:end_idx]

        data_vars = []
        for addr, var in page_vars:
            var_info = {
                "address": hex(addr),
                "type": str(var.type) if var.type else None
            }

            # Try to get symbol name
            if hasattr(var, 'symbol') and var.symbol:
                var_info["name"] = var.symbol.name

            data_vars.append(var_info)

        return {
            "data_vars": data_vars,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_prev": page > 1
            }
        }

    except Exception as e:
        log.log_error(f"Error getting data vars resource: {e}")
        return {"error": str(e)}


def register_resources(mcp, context_manager):
    """Register all MCP resources.

    Args:
        mcp: FastMCP instance
        context_manager: BinAssistMCPBinaryContextManager instance
    """
    # Note: Resources are registered in server.py using @mcp.resource decorators
    # This function provides the resource data extraction logic
    pass
