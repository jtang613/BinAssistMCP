"""
FastMCP server implementation for BinAssist-MCP

This module provides the main MCP server with dual transport support (SSE and STDIO)
and comprehensive Binary Ninja integration.
"""

import logging
from contextlib import asynccontextmanager
from threading import Event, Thread
from typing import AsyncIterator, List, Optional

import anyio
from anyio import to_thread
from hypercorn.config import Config as HypercornConfig
from hypercorn.trio import serve
from mcp.server.fastmcp import Context, FastMCP

from .config import BinAssistMCPConfig, TransportType
from .context import BinAssistMCPBinaryContextManager
from .tools import BinAssistMCPTools

logger = logging.getLogger(__name__)

try:
    import binaryninja as bn
    BINJA_AVAILABLE = True
except ImportError:
    BINJA_AVAILABLE = False
    logger.warning("Binary Ninja not available")


@asynccontextmanager
async def server_lifespan(server: FastMCP) -> AsyncIterator[BinAssistMCPBinaryContextManager]:
    """Application lifecycle manager for the MCP server"""
    context_manager = BinAssistMCPBinaryContextManager(
        max_binaries=getattr(server, '_config', BinAssistMCPConfig()).binary.max_binaries
    )
    
    # Add initial binaries if provided
    initial_binaries = getattr(server, '_initial_binaries', [])
    for binary_view in initial_binaries:
        try:
            context_manager.add_binary(binary_view)
        except Exception as e:
            logger.error(f"Failed to add initial binary: {e}")
    
    logger.info(f"Server started with {len(context_manager)} initial binaries")
    
    try:
        yield context_manager
    finally:
        logger.info("Shutting down server, clearing binary context")
        context_manager.clear()


class SSEServerThread(Thread):
    """Thread for running the SSE server"""
    
    def __init__(self, asgi_app, config: BinAssistMCPConfig):
        super().__init__(name="BinAssist-SSE-Server", daemon=True)
        self.asgi_app = asgi_app
        self.config = config
        self.shutdown_signal = Event()
        self.hypercorn_config = HypercornConfig()
        self.hypercorn_config.bind = [f"{config.server.host}:{config.server.port}"]
        
    def run(self):
        """Run the SSE server"""
        try:
            logger.info(f"Starting SSE server on {self.config.get_sse_url()}")
            logger.info(f"Hypercorn config: {self.hypercorn_config.bind}")
            anyio.run(self._run_server, backend='trio')
        except Exception as e:
            logger.error(f"SSE server error: {e}")
            import traceback
            logger.error(f"SSE server traceback: {traceback.format_exc()}")
            
    async def _run_server(self):
        """Async server runner"""
        try:
            await serve(
                self.asgi_app, 
                self.hypercorn_config, 
                shutdown_trigger=self._shutdown_trigger
            )
        except Exception as e:
            logger.error(f"Server serve error: {e}")
            
    async def _shutdown_trigger(self):
        """Wait for shutdown signal"""
        logger.debug("Waiting for shutdown signal")
        await to_thread.run_sync(self.shutdown_signal.wait)
        logger.info("Shutdown signal received")
        
    def stop(self):
        """Stop the server"""
        logger.info("Stopping SSE server")
        self.shutdown_signal.set()


class BinAssistMCPServer:
    """Main BinAssist-MCP server class"""
    
    def __init__(self, config: Optional[BinAssistMCPConfig] = None):
        """Initialize the MCP server
        
        Args:
            config: Configuration object, creates default if None
        """
        self.config = config or BinAssistMCPConfig()
        self.mcp_server: Optional[FastMCP] = None
        self.sse_thread: Optional[SSEServerThread] = None
        self._initial_binaries: List = []
        self._running = False
        
        logger.info(f"Initialized BinAssist-MCP server with config: {self.config}")
        
    def add_initial_binary(self, binary_view):
        """Add a binary view to be loaded on server start
        
        Args:
            binary_view: Binary Ninja BinaryView object
        """
        if not BINJA_AVAILABLE:
            logger.warning("Binary Ninja not available, cannot add binary")
            return
            
        self._initial_binaries.append(binary_view)
        logger.info(f"Added initial binary (total: {len(self._initial_binaries)})")
        
    def create_mcp_server(self) -> FastMCP:
        """Create and configure the FastMCP server instance"""
        try:
            logger.info("Creating FastMCP instance...")
            mcp = FastMCP(
                name="BinAssist-MCP",
                version="1.0.0",
                description="Comprehensive MCP server for Binary Ninja reverse engineering",
                lifespan=server_lifespan
            )
            logger.info("FastMCP instance created")
            
            # Store configuration and initial binaries for lifespan access
            logger.info("Storing configuration and initial binaries...")
            mcp._config = self.config
            mcp._initial_binaries = self._initial_binaries
            
            logger.info("Registering tools...")
            self._register_tools(mcp)
            logger.info("Tools registered successfully")
            
            logger.info("Registering resources...")
            self._register_resources(mcp)
            logger.info("Resources registered successfully")
            
            return mcp
            
        except Exception as e:
            logger.error(f"Failed to create MCP server: {e}")
            import traceback
            logger.error(f"MCP server creation traceback: {traceback.format_exc()}")
            raise
        
    def _register_tools(self, mcp: FastMCP):
        """Register all MCP tools"""
        
        @mcp.tool()
        def list_binaries(ctx: Context) -> List[str]:
            """List all currently loaded binary names"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            return context_manager.list_binaries()
            
        @mcp.tool()
        def get_binary_status(filename: str, ctx: Context) -> dict:
            """Get status information for a specific binary"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            try:
                binary_info = context_manager.get_binary_info(filename)
                return {
                    "name": binary_info.name,
                    "loaded": True,
                    "file_path": str(binary_info.file_path) if binary_info.file_path else None,
                    "analysis_complete": binary_info.analysis_complete,
                    "load_time": binary_info.load_time
                }
            except KeyError as e:
                return {
                    "name": filename,
                    "loaded": False,
                    "error": str(e)
                }
                
        # Analysis tools
        @mcp.tool()
        def rename_symbol(filename: str, address_or_name: str, new_name: str, ctx: Context) -> str:
            """Rename a function or data variable"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.rename_symbol(address_or_name, new_name)
            
        @mcp.tool()
        def decompile_function(filename: str, address_or_name: str, ctx: Context) -> str:
            """Decompile a function to high-level representation"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.decompile_function(address_or_name)
            
        @mcp.tool()
        def get_function_pseudo_c(filename: str, address_or_name: str, ctx: Context) -> str:
            """Get pseudo C code for a function"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_function_pseudo_c(address_or_name)
            
        @mcp.tool()
        def get_function_high_level_il(filename: str, address_or_name: str, ctx: Context) -> str:
            """Get High Level IL for a function"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_function_high_level_il(address_or_name)
            
        @mcp.tool()
        def get_function_medium_level_il(filename: str, address_or_name: str, ctx: Context) -> str:
            """Get Medium Level IL for a function"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_function_medium_level_il(address_or_name)
            
        @mcp.tool()
        def get_disassembly(filename: str, address_or_name: str, ctx: Context, length: Optional[int] = None) -> str:
            """Get disassembly for a function or address range"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_disassembly(address_or_name, length)
            
        # Information retrieval tools
        @mcp.tool()
        def get_functions(filename: str, ctx: Context) -> list:
            """Get list of all functions in the binary"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_functions()
            
        @mcp.tool()
        def search_functions_by_name(filename: str, search_term: str, ctx: Context) -> list:
            """Search functions by name substring"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.search_functions_by_name(search_term)
            
        @mcp.tool()
        def get_imports(filename: str, ctx: Context) -> dict:
            """Get imported symbols"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_imports()
            
        @mcp.tool()
        def get_exports(filename: str, ctx: Context) -> list:
            """Get exported symbols"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_exports()
            
        @mcp.tool()
        def get_strings(filename: str, ctx: Context) -> list:
            """Get strings found in the binary"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_strings()
            
        @mcp.tool()
        def get_segments(filename: str, ctx: Context) -> list:
            """Get memory segments"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_segments()
            
        @mcp.tool()
        def get_sections(filename: str, ctx: Context) -> list:
            """Get binary sections"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_sections()
            
        @mcp.tool()
        def update_analysis_and_wait(filename: str, ctx: Context) -> str:
            """Update binary analysis and wait for completion"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            result = tools.update_analysis_and_wait()
            # Update context manager status
            context_manager.update_analysis_status(filename)
            return result
            
        # Class and namespace management tools
        @mcp.tool()
        def get_classes(filename: str, ctx: Context) -> list:
            """Get all classes/structs/types in the binary"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_classes()
            
        @mcp.tool()
        def create_class(filename: str, name: str, size: int, ctx: Context) -> str:
            """Create a new class/struct type"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.create_class(name, size)
            
        @mcp.tool()
        def add_class_member(filename: str, class_name: str, member_name: str, member_type: str, offset: int, ctx: Context) -> str:
            """Add a member to an existing class/struct"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.add_class_member(class_name, member_name, member_type, offset)
            
        @mcp.tool()
        def get_namespaces(filename: str, ctx: Context) -> list:
            """Get all namespaces in the binary"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_namespaces()
            
        # Advanced data management tools
        @mcp.tool()
        def create_data_var(filename: str, address: str, var_type: str, ctx: Context, name: Optional[str] = None) -> str:
            """Create a data variable at the specified address"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.create_data_var(address, var_type, name)
            
        @mcp.tool()
        def get_data_vars(filename: str, ctx: Context) -> list:
            """Get all data variables in the binary"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_data_vars()
            
        @mcp.tool()
        def get_data_at_address(filename: str, address: str, ctx: Context, size: Optional[int] = None) -> dict:
            """Get data at a specific address"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_data_at_address(address, size)
            
        # Comment management tools
        @mcp.tool()
        def set_comment(filename: str, address: str, comment: str, ctx: Context) -> str:
            """Set a comment at the specified address"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.set_comment(address, comment)
            
        @mcp.tool()
        def get_comment(filename: str, address: str, ctx: Context) -> Optional[str]:
            """Get comment at the specified address"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_comment(address)
            
        @mcp.tool()
        def get_all_comments(filename: str, ctx: Context) -> list:
            """Get all comments in the binary"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_all_comments()
            
        @mcp.tool()
        def remove_comment(filename: str, address: str, ctx: Context) -> str:
            """Remove comment at the specified address"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.remove_comment(address)
            
        @mcp.tool()
        def set_function_comment(filename: str, function_name_or_address: str, comment: str, ctx: Context) -> str:
            """Set a comment for an entire function"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.set_function_comment(function_name_or_address, comment)
            
        # Variable management tools
        @mcp.tool()
        def create_variable(filename: str, function_name_or_address: str, var_name: str, var_type: str, ctx: Context, storage: str = "auto") -> str:
            """Create a local variable in a function"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.create_variable(function_name_or_address, var_name, var_type, storage)
            
        @mcp.tool()
        def get_variables(filename: str, function_name_or_address: str, ctx: Context) -> list:
            """Get all variables in a function"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_variables(function_name_or_address)
            
        @mcp.tool()
        def rename_variable(filename: str, function_name_or_address: str, old_name: str, new_name: str, ctx: Context) -> str:
            """Rename a variable in a function"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.rename_variable(function_name_or_address, old_name, new_name)
            
        @mcp.tool()
        def set_variable_type(filename: str, function_name_or_address: str, var_name: str, var_type: str, ctx: Context) -> str:
            """Set the type of a variable in a function"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.set_variable_type(function_name_or_address, var_name, var_type)
            
        # Type system tools
        @mcp.tool()
        def create_type(filename: str, name: str, definition: str, ctx: Context) -> str:
            """Create a new type definition"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.create_type(name, definition)
            
        @mcp.tool()
        def get_types(filename: str, ctx: Context) -> list:
            """Get all user-defined types"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_types()
            
        @mcp.tool()
        def create_enum(filename: str, name: str, members: dict, ctx: Context) -> str:
            """Create an enumeration type"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.create_enum(name, members)
            
        @mcp.tool()
        def create_typedef(filename: str, name: str, base_type: str, ctx: Context) -> str:
            """Create a type alias (typedef)"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.create_typedef(name, base_type)
            
        @mcp.tool()
        def get_type_info(filename: str, type_name: str, ctx: Context) -> dict:
            """Get detailed information about a specific type"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_type_info(type_name)
            
        # Function analysis tools
        @mcp.tool()
        def get_call_graph(filename: str, ctx: Context, function_name_or_address: Optional[str] = None) -> dict:
            """Get call graph information for a function or entire binary"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_call_graph(function_name_or_address)
            
        @mcp.tool()
        def analyze_function(filename: str, function_name_or_address: str, ctx: Context) -> dict:
            """Perform comprehensive analysis of a function"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.analyze_function(function_name_or_address)
            
        @mcp.tool()
        def get_cross_references(filename: str, address_or_name: str, ctx: Context) -> dict:
            """Get cross-references for a function or address"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_cross_references(address_or_name)
            
        # Enhanced function listing tools
        @mcp.tool()
        def get_functions_advanced(filename: str, ctx: Context,
                                   name_filter: Optional[str] = None,
                                   min_size: Optional[int] = None,
                                   max_size: Optional[int] = None,
                                   has_parameters: Optional[bool] = None,
                                   sort_by: str = "address",
                                   limit: Optional[int] = None) -> list:
            """Get functions with advanced filtering and search capabilities"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_functions_advanced(name_filter, min_size, max_size, has_parameters, sort_by, limit)
            
        @mcp.tool()
        def search_functions_advanced(filename: str, search_term: str, ctx: Context,
                                      search_in: str = "name",
                                      case_sensitive: bool = False) -> list:
            """Advanced function search with multiple search targets"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.search_functions_advanced(search_term, search_in, case_sensitive)
            
        @mcp.tool()
        def get_function_statistics(filename: str, ctx: Context) -> dict:
            """Get comprehensive statistics about all functions in the binary"""
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_function_statistics()
            
        logger.info("Registered MCP tools")
        
    def _register_resources(self, mcp: FastMCP):
        """Register MCP resources"""
        
        @mcp.resource("binassist://{filename}/triage_summary")
        def get_triage_summary_resource(filename: str) -> dict:
            """Get binary triage summary"""
            context_manager: BinAssistMCPBinaryContextManager = mcp.get_context().request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_triage_summary()
            
        @mcp.resource("binassist://{filename}/functions")
        def get_functions_resource(filename: str) -> list:
            """Get functions as a resource"""
            context_manager: BinAssistMCPBinaryContextManager = mcp.get_context().request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_functions()
            
        @mcp.resource("binassist://{filename}/imports")
        def get_imports_resource(filename: str) -> dict:
            """Get imports as a resource"""
            context_manager: BinAssistMCPBinaryContextManager = mcp.get_context().request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_imports()
            
        @mcp.resource("binassist://{filename}/exports")
        def get_exports_resource(filename: str) -> list:
            """Get exports as a resource"""
            context_manager: BinAssistMCPBinaryContextManager = mcp.get_context().request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_exports()
            
        @mcp.resource("binassist://{filename}/strings")
        def get_strings_resource(filename: str) -> list:
            """Get strings as a resource"""
            context_manager: BinAssistMCPBinaryContextManager = mcp.get_context().request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_strings()
            
        logger.info("Registered MCP resources")
        
    def start(self) -> bool:
        """Start the MCP server with configured transports
        
        Returns:
            True if started successfully, False otherwise
        """
        if self._running:
            logger.warning("Server is already running")
            return True
            
        try:
            logger.info("Starting BinAssist-MCP server...")
            
            # Also log to Binary Ninja
            try:
                import binaryninja as bn
                bn.log_info("BinAssist-MCP: Server.start() method called")
            except Exception as bn_log_error:
                logger.error(f"Failed to log to Binary Ninja: {bn_log_error}")
                import traceback
                logger.error(f"BN log traceback: {traceback.format_exc()}")
            
            # Validate configuration
            logger.info("Validating configuration...")
            errors = self.config.validate()
            if errors:
                logger.error(f"Configuration errors: {errors}")
                try:
                    import binaryninja as bn
                    bn.log_error(f"BinAssist-MCP configuration errors: {errors}")
                except Exception as bn_log_error:
                    logger.error(f"Failed to log config errors to Binary Ninja: {bn_log_error}")
                    import traceback
                    logger.error(f"BN log traceback: {traceback.format_exc()}")
                return False
            logger.info("Configuration validation passed")
            
            try:
                import binaryninja as bn
                bn.log_info("BinAssist-MCP: Configuration validation passed")
            except Exception as bn_log_error:
                logger.error(f"Failed to log validation success to Binary Ninja: {bn_log_error}")
                import traceback
                logger.error(f"BN log traceback: {traceback.format_exc()}")
                
            # Create MCP server
            logger.info("Creating MCP server instance...")
            self.mcp_server = self.create_mcp_server()
            logger.info("MCP server instance created successfully")
            
            # Start SSE transport if enabled
            if self.config.is_transport_enabled(TransportType.SSE):
                logger.info("SSE transport is enabled, starting SSE server...")
                self._start_sse_server()
            else:
                logger.info("SSE transport is disabled")
                
            self._running = True
            logger.info(f"BinAssist-MCP server started successfully")
            logger.info(f"Available transports: {self.config.server.transport.value}")
            
            if self.config.is_transport_enabled(TransportType.SSE):
                logger.info(f"SSE endpoint: {self.config.get_sse_url()}")
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            # Also log to Binary Ninja if available
            try:
                import binaryninja as bn
                bn.log_error(f"BinAssist-MCP server startup failed: {e}")
                import traceback
                traceback_msg = traceback.format_exc()
                bn.log_error(f"Server startup traceback: {traceback_msg}")
            except Exception as bn_log_error:
                logger.error(f"Failed to log startup error to Binary Ninja: {bn_log_error}")
                import traceback
                logger.error(f"BN log error traceback: {traceback.format_exc()}")
            self.stop()
            return False
            
    def _start_sse_server(self):
        """Start the SSE server thread"""
        if not self.mcp_server:
            raise RuntimeError("MCP server not created")
            
        try:
            # Create ASGI app for SSE transport
            logger.info(f"Available MCP server methods: {[m for m in dir(self.mcp_server) if not m.startswith('_')]}")
            
            if hasattr(self.mcp_server, 'sse_app'):
                logger.info("Using sse_app method")
                asgi_app = self.mcp_server.sse_app()
            elif hasattr(self.mcp_server, 'create_asgi_app'):
                logger.info("Using create_asgi_app method")
                asgi_app = self.mcp_server.create_asgi_app()
            elif hasattr(self.mcp_server, 'asgi'):
                logger.info("Using asgi property")
                asgi_app = self.mcp_server.asgi
            elif hasattr(self.mcp_server, '_asgi_app'):
                logger.info("Using _asgi_app property")
                asgi_app = self.mcp_server._asgi_app
            elif hasattr(self.mcp_server, 'app'):
                logger.info("Using app property")
                asgi_app = self.mcp_server.app
            elif callable(self.mcp_server):
                logger.info("MCP server is callable, using it directly as ASGI app")
                asgi_app = self.mcp_server
            else:
                # Let's see what attributes it actually has
                all_attrs = [attr for attr in dir(self.mcp_server) if not attr.startswith('__')]
                logger.error(f"MCP server attributes: {all_attrs}")
                
                # Try to find any ASGI-like method
                asgi_methods = [attr for attr in all_attrs if 'asgi' in attr.lower() or 'app' in attr.lower()]
                logger.error(f"Potential ASGI methods: {asgi_methods}")
                
                raise RuntimeError("Cannot create ASGI app for SSE transport")
            logger.info(f"Created ASGI app: {asgi_app}")
            
            self.sse_thread = SSEServerThread(asgi_app, self.config)
            logger.info(f"Created SSE server thread for {self.config.server.host}:{self.config.server.port}")
            
            self.sse_thread.start()
            logger.info("SSE server thread started")
            
            # Give the thread a moment to start
            import time
            time.sleep(0.1)
            
            if self.sse_thread.is_alive():
                logger.info("SSE server thread is running")
            else:
                logger.error("SSE server thread failed to start")
                
        except Exception as e:
            logger.error(f"Failed to start SSE server: {e}")
            raise
        
    def stop(self):
        """Stop the MCP server"""
        if not self._running:
            logger.warning("Server is not running")
            return
            
        logger.info("Stopping BinAssist-MCP server")
        
        # Stop SSE server
        if self.sse_thread:
            self.sse_thread.stop()
            self.sse_thread.join(timeout=5.0)
            self.sse_thread = None
            
        self.mcp_server = None
        self._running = False
        
        logger.info("BinAssist-MCP server stopped")
        
    def is_running(self) -> bool:
        """Check if the server is running"""
        return self._running
        
    def get_stdio_server(self):
        """Get the MCP server for STDIO transport
        
        Returns:
            FastMCP server instance for STDIO transport
        """
        if not self.config.is_transport_enabled(TransportType.STDIO):
            raise RuntimeError("STDIO transport not enabled")
            
        if not self.mcp_server:
            self.mcp_server = self.create_mcp_server()
            
        return self.mcp_server
        
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()