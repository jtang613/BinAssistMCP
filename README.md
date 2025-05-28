# BinAssist-MCP

<div align="center">

**Comprehensive MCP Server for Binary Ninja Reverse Engineering**

[![Binary Ninja](https://img.shields.io/badge/Binary%20Ninja-4000%2B-orange.svg)](https://binary.ninja)

*Seamless integration between Binary Ninja and LLM clients through the Model Context Protocol*

</div>

## Overview

BinAssist-MCP is a comprehensive Model Context Protocol (MCP) server that exposes Binary Ninja's powerful reverse engineering capabilities to LLM clients like Claude Desktop. It provides dual transport support (SSE and STDIO), extensive binary analysis tools, and seamless multi-binary session management.

### Key Features

- **Dual Transport Support**: Both SSE (Server-Sent Events) and STDIO transports
- **Comprehensive Tool Set**: 35+ reverse engineering tools combining the best from existing implementations
- **Advanced Architecture**: FastMCP-based server with intelligent context management
- **Configurable Settings**: Flexible configuration with Binary Ninja integration
- **Multi-Binary Support**: Handle multiple binaries simultaneously with automatic lifecycle management
- **Plugin Integration**: Native Binary Ninja plugin with menu integration and auto-startup
- **CLI Interface**: Full command-line interface for standalone operation

## Installation

### Via Binary Ninja Plugin Manager

1. Open Binary Ninja
2. Go to `Plugins > Manage Plugins`
3. Search for "BinAssist-MCP"
4. Click Install

### Manual Installation

1. Clone the repository:
```bash
git clone https://github.com/binassist/binassist-mcp.git
cd BinAssist-MCP
```

2. Install the package:
```bash
pip install -r requirements.txt
```

3. For Binary Ninja plugin installation, copy to your plugins directory:
```bash
# Linux/macOS
cp -r src/binassist_mcp ~/.binaryninja/plugins/

# Windows
copy src\binassist_mcp "%APPDATA%\Binary Ninja\plugins\"
```

## Quick Start

### Binary Ninja Plugin

1. Open Binary Ninja and load a binary file
2. Go to `BinAssist-MCP > Start Server` (or enable auto-startup in settings)
3. The server will start automatically and expose your binary for analysis

### Standalone Server

```bash
# Start server with a binary file
binassist-mcp serve /path/to/binary.exe

# Start server with multiple binaries
binassist-mcp serve binary1.exe binary2.so binary3.dll

# Custom host/port
binassist-mcp serve --host 0.0.0.0 --port 9000 binary.exe
```

### STDIO Transport for MCP Clients

```bash
# For Claude Desktop or other STDIO MCP clients
binassist-mcp stdio
```

## Configuration

### Binary Ninja Settings

BinAssist-MCP integrates with Binary Ninja's settings system. Configure through:
- `Settings > Preferences > binassist.*`

Key settings:
- `binassist.server.host`: Server host address (default: localhost)
- `binassist.server.port`: Server port (default: 9090)
- `binassist.server.transport`: Transport type (sse/stdio/both)
- `binassist.plugin.auto_startup`: Auto-start server on file open
- `binassist.binary.max_binaries`: Maximum concurrent binaries

### Environment Variables

```bash
export BINASSIST_SERVER__HOST=localhost
export BINASSIST_SERVER__PORT=9090
export BINASSIST_SERVER__TRANSPORT=both
export BINASSIST_PLUGIN__AUTO_STARTUP=true
```

### Configuration File

Create a JSON configuration file:
```json
{
  "server": {
    "host": "localhost",
    "port": 9090,
    "transport": "both"
  },
  "plugin": {
    "auto_startup": true,
    "show_notifications": true
  },
  "binary": {
    "max_binaries": 10,
    "auto_analysis": true
  }
}
```

## Available Tools

BinAssist-MCP provides comprehensive reverse engineering tools across multiple categories:

### Core Analysis Tools
- `rename_symbol`: Rename functions and data variables
- `decompile_function`: High-level decompilation 
- `get_function_pseudo_c`: Pseudo C code generation
- `get_function_high_level_il` / `get_function_medium_level_il`: IL representations
- `get_disassembly`: Function and range disassembly
- `get_assembly_function`: Annotated assembly with comments

### Class & Type Management
- `get_classes`: List all classes/structs in the binary
- `create_class`: Create new class/struct types
- `add_class_member`: Add members to existing classes
- `get_namespaces`: List all namespaces and their symbols
- `create_type`: Create custom type definitions
- `get_types`: List all user-defined types
- `create_enum`: Create enumeration types
- `create_typedef`: Create type aliases
- `get_type_info`: Get detailed type information

### Data Management
- `create_data_var`: Create data variables at addresses
- `get_data_vars`: List all data variables
- `get_data_at_address`: Read and interpret data at addresses

### Variable Management
- `create_variable`: Create local variables in functions
- `get_variables`: List function parameters and local variables
- `rename_variable`: Rename variables in functions
- `set_variable_type`: Set/change variable types

### Comment Management
- `set_comment`: Add comments at addresses
- `get_comment`: Get comment at specific address
- `get_all_comments`: List all comments in the binary
- `remove_comment`: Remove comments
- `set_function_comment`: Set comments for entire functions

### Function Analysis
- `get_call_graph`: Generate call graphs (single function or global)
- `analyze_function`: Comprehensive function analysis with metrics
- `get_cross_references`: Find code/data cross-references
- `get_functions_advanced`: Advanced function listing with filtering
- `search_functions_advanced`: Multi-target function search
- `get_function_statistics`: Binary-wide function statistics

### Information Retrieval
- `get_functions`: List all functions with metadata
- `search_functions_by_name`: Search functions by substring
- `get_imports` / `get_exports`: Symbol import/export analysis
- `get_strings`: String analysis with metadata
- `get_segments` / `get_sections`: Memory layout information
- `get_triage_summary`: Comprehensive binary overview

### Session Management
- `list_binaries`: List loaded binaries
- `get_binary_status`: Binary status and metadata
- `update_analysis_and_wait`: Force analysis update

## MCP Resources

All tools are also available as URI-accessible resources:

```
binassist://{filename}/triage_summary
binassist://{filename}/functions
binassist://{filename}/imports
binassist://{filename}/exports
binassist://{filename}/strings
binassist://{filename}/segments
binassist://{filename}/sections
```

## Integration Examples

### Claude Desktop

Add to your Claude Desktop configuration (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "binassist-mcp": {
      "command": "binassist-mcp",
      "args": ["stdio"]
    }
  }
}
```

### SSE Integration

For web-based MCP clients:
```
SSE Endpoint: http://localhost:9090/sse
```

## Development

### Setup Development Environment

```bash
# Clone repository
git clone https://github.com/binassist/binassist-mcp.git
cd BinAssist-MCP

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=binassist_mcp

# Run specific test categories
pytest -m unit
pytest -m integration
```

### Code Quality

```bash
# Format code
ruff format

# Lint code
ruff check

# Type checking
mypy src/binassist_mcp
```

## CLI Commands

### Server Commands

```bash
# Start server with binaries
binassist-mcp serve binary1.exe binary2.so

# STDIO transport only
binassist-mcp stdio

# Show configuration
binassist-mcp config

# System check
binassist-mcp check
```

### Analysis Commands

```bash
# Quick binary analysis
binassist-mcp analyze /path/to/binary.exe

# Show version
binassist-mcp version
```

## Architecture

BinAssist-MCP uses a modular architecture:

- **Server Layer**: FastMCP-based server with dual transport support
- **Context Management**: Multi-binary session handling with lifecycle management
- **Tools Layer**: Comprehensive Binary Ninja API integration
- **Configuration**: Pydantic-based settings with Binary Ninja integration
- **Plugin Integration**: Native Binary Ninja plugin with UI integration

## Performance

- **Concurrent Binaries**: Supports up to 50 concurrent binaries (configurable)
- **Memory Management**: Automatic cleanup and eviction policies
- **Analysis Caching**: Optional result caching for improved performance
- **Transport Efficiency**: Optimized SSE and STDIO transport implementations

## Contributing

Contributions welcome!

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Run the test suite and linting
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Binary Ninja**: For providing the comprehensive reverse engineering platform
- **Anthropic**: For the Model Context Protocol specification
- **FastMCP**: For the excellent MCP server framework
