# BinAssistMCP

> Comprehensive Model Context Protocol (MCP) server for Binary Ninja with AI-powered reverse engineering capabilities

## Summary

BinAssistMCP is a powerful bridge between Binary Ninja and Large Language Models (LLMs) like Claude, providing comprehensive reverse engineering tools through the Model Context Protocol (MCP). It enables AI-assisted binary analysis by exposing Binary Ninja's advanced capabilities through both Server-Sent Events (SSE) and STDIO transports.

### Key Features

- **Dual Transport Support**: Both SSE (web-based) and STDIO (command-line) transports
- **40+ Analysis Tools**: Complete Binary Ninja API wrapper with advanced functionality
- **Multi-Binary Sessions**: Concurrent analysis of multiple binaries with intelligent context management
- **Smart Symbol Management**: Advanced function searching, renaming, and type management
- **Auto-Integration**: Seamless Binary Ninja plugin with automatic startup capabilities
- **Flexible Configuration**: Comprehensive settings management through Binary Ninja's interface
- **AI-Ready**: Optimized for LLM integration with structured tool responses

### Use Cases

- **AI-Assisted Reverse Engineering**: Leverage LLMs for intelligent code analysis and documentation
- **Automated Binary Analysis**: Script complex analysis workflows with natural language
- **Research and Education**: Teach reverse engineering concepts with AI guidance
- **Security Analysis**: Accelerate vulnerability research and malware analysis
- **Code Understanding**: Generate comprehensive documentation and explanations

## Tool Details

BinAssistMCP provides over 40 specialized tools organized into functional categories:

### Binary Management
- **`list_binaries`** - List all loaded binary files
- **`get_binary_status`** - Check analysis status and metadata
- **`update_analysis_and_wait`** - Force analysis update and wait for completion

### Code Analysis & Decompilation
- **`decompile_function`** - Generate high-level decompiled code
- **`get_function_pseudo_c`** - Extract pseudo-C representation
- **`get_function_high_level_il`** - Access High-Level Intermediate Language
- **`get_function_medium_level_il`** - Access Medium-Level Intermediate Language
- **`get_disassembly`** - Retrieve assembly code with annotations

### Information Retrieval
- **`get_functions`** - List all functions with metadata
- **`search_functions_by_name`** - Find functions by name patterns
- **`get_functions_advanced`** - Advanced filtering (size, complexity, parameters)
- **`search_functions_advanced`** - Multi-target searching (name, comments, calls, variables)
- **`get_function_statistics`** - Comprehensive binary statistics
- **`get_imports`** - Import table analysis grouped by module
- **`get_exports`** - Export table with symbol information
- **`get_strings`** - String extraction with context
- **`get_segments`** - Memory layout analysis
- **`get_sections`** - Binary section information

### Symbol & Naming Management
- **`rename_symbol`** - Rename functions and data variables
- **`get_cross_references`** - Find all references to/from symbols
- **`analyze_function`** - Comprehensive function analysis
- **`get_call_graph`** - Call relationship mapping

### Documentation & Comments
- **`set_comment`** - Add comments to specific addresses
- **`get_comment`** - Retrieve comments at addresses
- **`get_all_comments`** - Export all comments with context
- **`remove_comment`** - Delete existing comments
- **`set_function_comment`** - Add function-level documentation

### Variable Management
- **`create_variable`** - Define local variables in functions
- **`get_variables`** - List function parameters and locals
- **`rename_variable`** - Rename variables for clarity
- **`set_variable_type`** - Update variable type information

### Type System Management
- **`create_type`** - Define custom types and structures
- **`get_types`** - List all user-defined types
- **`create_enum`** - Create enumeration types
- **`create_typedef`** - Create type aliases
- **`get_type_info`** - Detailed type information
- **`get_classes`** - List classes and structures
- **`create_class`** - Define new classes/structures
- **`add_class_member`** - Add members to existing types

### Data Analysis
- **`create_data_var`** - Define data variables at addresses
- **`get_data_vars`** - List all defined data variables
- **`get_data_at_address`** - Analyze raw data with type inference

### Navigation & Context
- **`get_current_address`** - Get current cursor position
- **`get_current_function`** - Identify function at current address
- **`get_namespaces`** - Namespace and symbol organization

### Advanced Analysis
- **`get_triage_summary`** - Complete binary overview
- **`get_function_statistics`** - Statistical analysis of all functions

Each tool is designed for seamless integration with AI workflows, providing structured responses that LLMs can easily interpret and act upon.

## Installation

### Prerequisites

- **Binary Ninja**: Version 4000 or higher
- **Python**: 3.8+ (typically bundled with Binary Ninja)
- **Platform**: Windows, macOS, or Linux

### Option 1: Binary Ninja Plugin Manager (Recommended)

1. Open Binary Ninja
2. Navigate to **Tools** → **Manage Plugins**
3. Search for "BinAssistMCP"
4. Click **Install**
5. Restart Binary Ninja

### Option 2: Manual Installation

#### Step 1: Download and Extract
```bash
git clone https://github.com/jtang613/BinAssistMCP.git
cd BinAssistMCP
```

#### Step 2: Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Or install individually:
pip install anyio>=4.0.0 hypercorn>=0.16.0 mcp>=1.0.0 trio>=0.27.0 pydantic>=2.0.0 pydantic-settings>=2.0.0 click>=8.0.0
```

#### Step 3: Copy to Plugin Directory

**Windows:**
```cmd
copy BinAssistMCP "%APPDATA%\Binary Ninja\plugins\"
```

**macOS:**
```bash
cp -r BinAssistMCP ~/Library/Application\ Support/Binary\ Ninja/plugins/
```

**Linux:**
```bash
cp -r BinAssistMCP ~/.binaryninja/plugins/
```

#### Step 4: Verify Installation

1. Restart Binary Ninja
2. Open any binary file
3. Check **Tools** menu for "BinAssistMCP" submenu
4. Look for startup messages in the log panel

### Configuration

#### Basic Setup

1. Open Binary Ninja Settings (**Edit** → **Preferences**)
2. Navigate to the **binassistmcp** section
3. Configure server settings:
   - **Host**: `localhost` (default)
   - **Port**: `9090` (default)
   - **Transport**: `both` (SSE + STDIO)

#### Advanced Configuration

```python
# Environment variables (optional)
export BINASSISTMCP_SERVER__HOST=localhost
export BINASSISTMCP_SERVER__PORT=9090
export BINASSISTMCP_SERVER__TRANSPORT=both
export BINASSISTMCP_BINARY__MAX_BINARIES=10
```

### Usage

#### Starting the Server

**Via Binary Ninja Menu:**
1. **Tools** → **BinAssistMCP** → **Start Server**
2. Check log panel for startup confirmation
3. Note the server URL (e.g., `http://localhost:9090`)

**Auto-Startup (Default):**
- Server starts automatically when Binary Ninja loads a file
- Configurable via settings: `binassistmcp.plugin.auto_startup`

#### Connecting with Claude Desktop

Add to your Claude Desktop MCP configuration:

```json
{
  "mcpServers": {
    "binassist": {
      "command": "python",
      "args": ["/path/to/BinAssistMCP"],
      "env": {
        "BINASSISTMCP_SERVER__TRANSPORT": "stdio"
      }
    }
  }
}
```

#### Using with SSE Transport

Connect web-based MCP clients to:
```
http://localhost:9090/sse
```

### Integration Examples

#### Basic Function Analysis
```
Ask Claude: "Analyze the main function in the loaded binary and explain what it does"

Claude will use tools like:
- get_functions() to find main
- decompile_function() to get readable code
- get_function_pseudo_c() for C representation
- analyze_function() for comprehensive analysis
```

#### Vulnerability Research
```
Ask Claude: "Find all functions that handle user input and check for buffer overflows"

Claude will use:
- search_functions_advanced() to find input handlers
- get_cross_references() to trace data flow
- get_variables() to analyze buffer usage
- set_comment() to document findings
```

### Troubleshooting

#### Common Issues

**Server won't start:**
- Check Binary Ninja log panel for error messages
- Verify all dependencies are installed
- Ensure port 9090 is not in use

**Binary Ninja crashes:**
- Check Python environment compatibility
- Try reducing `max_binaries` setting
- Restart with a single binary file

**Tools return errors:**
- Ensure binary analysis is complete
- Check if Binary Ninja file is still open
- Verify function/address exists

#### Support

- **Issues**: Report bugs on GitHub Issues
- **Binary Ninja**: Check official Binary Ninja documentation

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes following the existing code style
4. Test with multiple binary types
5. Submit a pull request

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
