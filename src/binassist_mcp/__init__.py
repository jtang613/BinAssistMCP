"""
BinAssist-MCP: A comprehensive MCP server for Binary Ninja

This package provides a FastMCP-based server that exposes Binary Ninja's 
reverse engineering capabilities through the Model Context Protocol (MCP).

Features:
- Dual transport support (SSE and STDIO)
- Comprehensive tool set for binary analysis
- Multi-binary session management
- Configurable server settings
- Binary Ninja plugin integration
"""

__version__ = "1.0.0"
__author__ = "BinAssist Team"
__description__ = "Comprehensive MCP server for Binary Ninja reverse engineering"

from .server import BinAssistMCPServer
from .config import BinAssistMCPConfig
from .tools import BinAssistMCPTools

__all__ = ["BinAssistMCPServer", "BinAssistMCPConfig", "BinAssistMCPTools"]