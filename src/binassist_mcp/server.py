"""
FastMCP server implementation for BinAssistMCP

This module provides the main MCP server with SSE transport
and comprehensive Binary Ninja integration.
"""

import warnings
from contextlib import asynccontextmanager
from threading import Event, Thread
from typing import AsyncIterator, List, Optional

import asyncio
from hypercorn.config import Config as HypercornConfig
from hypercorn.asyncio import serve
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.transport_security import TransportSecuritySettings

# Suppress ResourceWarnings for memory streams to reduce noise in logs
warnings.filterwarnings("ignore", category=ResourceWarning)


class ResourceManagedASGIApp:
    """ASGI app wrapper that ensures proper resource cleanup"""

    def __init__(self, app):
        self.app = app
        self._response_started = {}

    async def __call__(self, scope, receive, send):
        """ASGI callable with resource management"""
        # Track if response has started for this connection
        scope_id = id(scope)
        self._response_started[scope_id] = False

        async def wrapped_send(message):
            """Wrap send to track response state and prevent ASGI violations"""
            if message["type"] == "http.response.start":
                self._response_started[scope_id] = True
            elif message["type"] == "http.response.body":
                # Only send if response hasn't already completed
                if not self._response_started.get(scope_id):
                    log.log_debug("Attempted to send response body before response start")
                    return
            try:
                await send(message)
            except Exception as e:
                # Handle send errors gracefully (client disconnections, etc.)
                error_msg = str(e)
                # Check for expected ASGI state errors and connection issues
                if ("connection" in error_msg.lower() or
                    "closed" in error_msg.lower() or
                    "ASGIHTTPState" in error_msg or
                    "response already" in error_msg.lower() or
                    "Unexpected message type" in error_msg):
                    log.log_debug(f"Client disconnected or ASGI state error (expected): {e}")
                else:
                    log.log_warn(f"Error sending ASGI message: {e}")

        try:
            await self.app(scope, receive, wrapped_send)
        except BaseException as e:
            # Handle both exception groups and regular exceptions
            import sys
            if sys.version_info >= (3, 11):
                # Check if this is an ExceptionGroup
                if isinstance(e, BaseExceptionGroup):
                    log.log_error(f"ASGI exception group: {e}")
                    all_connection_errors = True
                    for exc in e.exceptions:
                        error_msg = str(exc)
                        # Check for all types of expected ASGI/connection errors
                        if ("ASGIHTTPState" in error_msg or
                            "connection" in error_msg.lower() or
                            "closed" in error_msg.lower() or
                            "response already" in error_msg.lower() or
                            "Unexpected message type" in error_msg):
                            log.log_debug(f"Client disconnect or ASGI state error (expected): {exc}")
                        else:
                            log.log_error(f"Unexpected exception in group: {exc}")
                            all_connection_errors = False
                    # Don't re-raise if all errors are connection-related
                    if all_connection_errors:
                        log.log_debug("All exceptions are connection-related, suppressing")
                        return
                    else:
                        raise

            # Handle single exceptions
            error_msg = str(e)
            # Check for all types of expected ASGI/connection errors
            if ("ASGIHTTPState" in error_msg or
                "connection" in error_msg.lower() or
                "closed" in error_msg.lower() or
                "response already" in error_msg.lower() or
                "Unexpected message type" in error_msg):
                log.log_debug(f"Client disconnect or ASGI state error (expected): {e}")
            else:
                # Log unexpected errors with full details
                log.log_error(f"Unexpected ASGI exception: {e}")
                import traceback
                log.log_error(f"Traceback: {traceback.format_exc()}")
            # Don't re-raise connection errors as they're expected during shutdown
        finally:
            # Clean up response tracking
            self._response_started.pop(scope_id, None)
            # Force cleanup of any lingering resources
            try:
                import gc
                gc.collect()
            except Exception:
                pass

from .config import BinAssistMCPConfig, TransportType
from .context import BinAssistMCPBinaryContextManager
from .logging import log
from .tools import BinAssistMCPTools

try:
    import binaryninja as bn
    BINJA_AVAILABLE = True
except ImportError:
    BINJA_AVAILABLE = False
    log.log_warn("Binary Ninja not available")


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
            log.log_error(f"Failed to add initial binary: {e}")

    log.log_info(f"Server started with {len(context_manager)} initial binaries")

    try:
        yield context_manager
    except BaseException as e:
        # Handle both exception groups and regular exceptions
        import sys
        if sys.version_info >= (3, 11):
            # Check if this is an ExceptionGroup
            if isinstance(e, BaseExceptionGroup):
                log.log_error(f"Server lifespan TaskGroup error: {e}")
                all_connection_errors = True
                for exc in e.exceptions:
                    log.log_error(f"Lifespan sub-exception: {exc}")
                    if not ("connection" in str(exc).lower() or "closed" in str(exc).lower()):
                        all_connection_errors = False
                # Don't re-raise if these are just connection errors
                if all_connection_errors:
                    log.log_debug("All lifespan exceptions are connection-related, suppressing")
                    return
                else:
                    raise

        # Handle regular exceptions
        log.log_error(f"Server lifespan error: {e}")
        import traceback
        log.log_error(f"Lifespan traceback: {traceback.format_exc()}")
        raise
    finally:
        try:
            log.log_info("Shutting down server, clearing binary context")
            context_manager.clear()

            # Force garbage collection to help clean up any lingering references
            import gc
            gc.collect()

            # Give more time for async cleanup and stream finalization
            await asyncio.sleep(0.5)

            log.log_info("Server lifespan cleanup completed")
        except Exception as e:
            log.log_error(f"Error during server shutdown: {e}")


class SSEServerThread(Thread):
    """Thread for running the SSE server with improved resource management"""
    
    def __init__(self, asgi_app, config: BinAssistMCPConfig):
        super().__init__(name="BinAssist-SSE-Server", daemon=True)
        self.asgi_app = asgi_app
        self.config = config
        self.shutdown_signal = Event()
        self.hypercorn_config = HypercornConfig()
        self.hypercorn_config.bind = [f"{config.server.host}:{config.server.port}"]

        # Configure better connection handling for resource cleanup
        self.hypercorn_config.keep_alive_timeout = 5
        self.hypercorn_config.graceful_timeout = 10
        
        # Disable hypercorn's logging to avoid ScriptingProvider messages
        self.hypercorn_config.access_log_format = ""
        self.hypercorn_config.error_logger = None
        self.hypercorn_config.access_logger = None
        
        # Completely disable hypercorn logging
        import logging
        logging.getLogger('hypercorn').disabled = True
        logging.getLogger('hypercorn.error').disabled = True
        logging.getLogger('hypercorn.access').disabled = True
        
        # Suppress resource warnings specifically for this thread
        warnings.filterwarnings("ignore", category=ResourceWarning)
        
    def run(self):
        """Run the SSE server"""
        try:
            log.log_info(f"Starting SSE server on {self.config.get_sse_url()}")
            log.log_info(f"Hypercorn config: {self.hypercorn_config.bind}")
            
            # Create a new event loop for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                loop.run_until_complete(self._run_server())
            finally:
                loop.close()
                
        except Exception as e:
            log.log_error(f"SSE server error: {e}")
            import traceback
            log.log_error(f"SSE server traceback: {traceback.format_exc()}")
            
    async def _run_server(self):
        """Async server runner with improved resource cleanup"""
        try:
            await serve(
                self.asgi_app,
                self.hypercorn_config,
                shutdown_trigger=self._shutdown_trigger
            )
        except BaseException as e:
            # Handle both exception groups and regular exceptions
            import sys
            import traceback
            if sys.version_info >= (3, 11):
                # Check if this is an ExceptionGroup
                if isinstance(e, BaseExceptionGroup):
                    log.log_error(f"Server TaskGroup error: {e}")
                    for exc in e.exceptions:
                        log.log_error(f"Sub-exception: {exc}")
                        log.log_error(f"Traceback: {''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))}")
                    return

            # Handle regular exceptions
            log.log_error(f"Server serve error: {e}")
            log.log_error(f"Traceback: {traceback.format_exc()}")
        finally:
            # Ensure proper cleanup with sufficient time for stream finalization
            try:
                log.log_debug("Starting SSE server cleanup")

                # Allow time for all pending connections and streams to close
                await asyncio.sleep(1.0)

                # Force garbage collection to clean up any orphaned streams
                import gc
                gc.collect()

                log.log_debug("SSE server cleanup completed")
            except Exception as cleanup_error:
                log.log_error(f"Error during SSE server cleanup: {cleanup_error}")
            
    async def _shutdown_trigger(self):
        """Wait for shutdown signal"""
        log.log_debug("Waiting for shutdown signal")
        # Use asyncio to run the blocking wait in a thread
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.shutdown_signal.wait)
        log.log_info("Shutdown signal received")
        
        # Allow time for existing connections to close gracefully
        await asyncio.sleep(0.5)
        
    def stop(self):
        """Stop the server with improved cleanup"""
        log.log_info("Stopping SSE server")
        self.shutdown_signal.set()
        
        # Wait for thread to finish with longer timeout for proper cleanup
        if self.is_alive():
            self.join(timeout=5.0)
            if self.is_alive():
                log.log_warn("SSE server thread did not shut down cleanly within 5 seconds")
            else:
                log.log_info("SSE server thread shutdown completed")


class BinAssistMCPServer:
    """Main BinAssistMCP server class"""
    
    def __init__(self, config: Optional[BinAssistMCPConfig] = None):
        """Initialize the MCP server
        
        Args:
            config: Configuration object, creates default if None
        """
        self.config = config or BinAssistMCPConfig()
        self.mcp_server: Optional[FastMCP] = None
        self.sse_thread: Optional[SSEServerThread] = None
        self.streamablehttp_thread: Optional[SSEServerThread] = None  # Reuse SSEServerThread for streamablehttp
        self._initial_binaries: List = []
        self._running = False
        
        log.log_info(f"Initialized BinAssistMCP server with config: {self.config}")
        
    def add_initial_binary(self, binary_view):
        """Add a binary view to be loaded on server start
        
        Args:
            binary_view: Binary Ninja BinaryView object
        """
        if not BINJA_AVAILABLE:
            log.log_warn("Binary Ninja not available, cannot add binary")
            return
            
        self._initial_binaries.append(binary_view)
        log.log_info(f"Added initial binary (total: {len(self._initial_binaries)})")
        
    def create_mcp_server(self) -> FastMCP:
        """Create and configure the FastMCP server instance"""
        try:
            log.log_info("Creating FastMCP instance...")
            mcp = FastMCP(
                name="BinAssistMCP",
#                version="1.0.0",
#                description="Comprehensive MCP server for Binary Ninja reverse engineering",
                lifespan=server_lifespan,
                # Disable DNS rebinding protection to allow binding to any IP address
                transport_security=TransportSecuritySettings(
                    enable_dns_rebinding_protection=False
                )
            )
            log.log_info("FastMCP instance created")
            
            # Store configuration and initial binaries for lifespan access
            log.log_info("Storing configuration and initial binaries...")
            mcp._config = self.config
            mcp._initial_binaries = self._initial_binaries
            
            log.log_info("Registering tools...")
            self._register_tools(mcp)
            log.log_info("Tools registered successfully")
            
            log.log_info("Registering resources...")
            self._register_resources(mcp)
            log.log_info("Resources registered successfully")
            
            return mcp
            
        except Exception as e:
            log.log_error(f"Failed to create MCP server: {e}")
            import traceback
            log.log_error(f"MCP server creation traceback: {traceback.format_exc()}")
            raise
        
    def _register_tools(self, mcp: FastMCP):
        """Register all MCP tools"""
        
        @mcp.tool()
        def list_binaries(ctx: Context) -> List[str]:
            """List all currently loaded binary names

            Returns:
                List of binary filenames currently loaded in the MCP server
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            return context_manager.list_binaries()
            
        @mcp.tool()
        def get_binary_status(filename: str, ctx: Context) -> dict:
            """Get status information for a specific binary

            Args:
                filename: Name of the binary file

            Returns:
                Dictionary with binary name, load status, file path, and analysis status
            """
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
            """Rename a function or data variable

            Args:
                filename: Name of the binary file
                address_or_name: Address (hex string) or name of the symbol
                new_name: New name for the symbol

            Returns:
                Success message string
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.rename_symbol(address_or_name, new_name)
            
        @mcp.tool()
        def decompile_function(filename: str, address_or_name: str, ctx: Context) -> str:
            """Decompile a function to high-level representation

            Args:
                filename: Name of the binary file
                address_or_name: Function name or address

            Returns:
                Decompiled function code
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.decompile_function(address_or_name)
            
        @mcp.tool()
        def get_function_pseudo_c(filename: str, address_or_name: str, ctx: Context) -> str:
            """Get pseudo C code for a function

            Args:
                filename: Name of the binary file
                address_or_name: Function name or address

            Returns:
                Pseudo C code as string
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_function_pseudo_c(address_or_name)
            
        @mcp.tool()
        def get_function_high_level_il(filename: str, address_or_name: str, ctx: Context) -> str:
            """Get High Level IL for a function

            Args:
                filename: Name of the binary file
                address_or_name: Function name or address

            Returns:
                HLIL as string
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_function_high_level_il(address_or_name)
            
        @mcp.tool()
        def get_function_medium_level_il(filename: str, address_or_name: str, ctx: Context) -> str:
            """Get Medium Level IL for a function

            Args:
                filename: Name of the binary file
                address_or_name: Function name or address

            Returns:
                MLIL as string
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_function_medium_level_il(address_or_name)
            
        @mcp.tool()
        def get_disassembly(filename: str, address_or_name: str, ctx: Context, length: Optional[int] = None) -> str:
            """Get disassembly for a function or address range

            Args:
                filename: Name of the binary file
                address_or_name: Function name or start address
                length: Optional length in bytes for range disassembly

            Returns:
                Disassembly as string
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_disassembly(address_or_name, length)
            
        # Information retrieval tools
        @mcp.tool()
        def get_functions(filename: str, ctx: Context) -> list:
            """Get list of all functions in the binary

            Args:
                filename: Name of the binary file

            Returns:
                List of function dictionaries with name, address, size, and metadata
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_functions()
            
        @mcp.tool()
        def search_functions_by_name(filename: str, search_term: str, ctx: Context) -> list:
            """Search functions by name substring

            Args:
                filename: Name of the binary file
                search_term: Substring to search for

            Returns:
                List of matching functions
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.search_functions_by_name(search_term)
            
        @mcp.tool()
        def get_imports(filename: str, ctx: Context) -> dict:
            """Get imported symbols grouped by module

            Args:
                filename: Name of the binary file

            Returns:
                Dictionary mapping module names to lists of imported symbols
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_imports()
            
        @mcp.tool()
        def get_exports(filename: str, ctx: Context) -> dict:
            """Get exported symbols

            Args:
                filename: Name of the binary file

            Returns:
                List of exported symbols with names, addresses, and types
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_exports()
            
        @mcp.tool()
        def get_strings(filename: str, ctx: Context) -> list:
            """Get strings found in the binary

            Args:
                filename: Name of the binary file

            Returns:
                List of strings with value, address, length, and type
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_strings()
            
        @mcp.tool()
        def get_segments(filename: str, ctx: Context) -> list:
            """Get memory segments

            Args:
                filename: Name of the binary file

            Returns:
                List of segments with start, end, length, and permissions
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_segments()
            
        @mcp.tool()
        def get_sections(filename: str, ctx: Context) -> list:
            """Get binary sections

            Args:
                filename: Name of the binary file

            Returns:
                List of sections with name, start, end, length, and metadata
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_sections()
            
        @mcp.tool()
        def update_analysis_and_wait(filename: str, ctx: Context) -> bool:
            """Update binary analysis and wait for completion

            Args:
                filename: Name of the binary file

            Returns:
                Success message string
            """
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
            """Get all classes/structs/types in the binary

            Args:
                filename: Name of the binary file

            Returns:
                List of class/struct definitions with members
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_classes()
            
        @mcp.tool()
        def create_class(filename: str, name: str, size: int, ctx: Context) -> str:
            """Create a new class/struct type

            Args:
                filename: Name of the binary file
                name: Name of the class/struct
                size: Size in bytes

            Returns:
                Success message
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.create_class(name, size)
            
        @mcp.tool()
        def add_class_member(filename: str, class_name: str, member_name: str, member_type: str, offset: int, ctx: Context) -> str:
            """Add a member to an existing class/struct

            Args:
                filename: Name of the binary file
                class_name: Name of the class/struct
                member_name: Name of the member
                member_type: Type of the member (e.g., 'int32_t', 'char*')
                offset: Offset within the struct

            Returns:
                Success message
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.add_class_member(class_name, member_name, member_type, offset)
            
        @mcp.tool()
        def get_namespaces(filename: str, ctx: Context) -> list:
            """Get all namespaces in the binary

            Args:
                filename: Name of the binary file

            Returns:
                List of namespaces with their symbols
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_namespaces()
            
        # Advanced data management tools
        @mcp.tool()
        def create_data_var(filename: str, address: str, var_type: str, ctx: Context, name: Optional[str] = None) -> str:
            """Create a data variable at the specified address

            Args:
                filename: Name of the binary file
                address: Address in hex format (e.g., '0x401000')
                var_type: Type of the variable (e.g., 'int32_t', 'char*')
                name: Optional name for the variable

            Returns:
                Success message
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.create_data_var(address, var_type, name)
            
        @mcp.tool()
        def get_data_vars(filename: str, ctx: Context) -> list:
            """Get all data variables in the binary

            Args:
                filename: Name of the binary file

            Returns:
                List of data variables with address, type, size, and name
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_data_vars()
            
        @mcp.tool()
        def get_data_at_address(filename: str, address: str, ctx: Context, size: Optional[int] = None) -> dict:
            """Get data at a specific address

            Args:
                filename: Name of the binary file
                address: Address in hex format
                size: Optional size to read (if not specified, uses data var size or default 16)

            Returns:
                Dictionary with data information including hex, raw bytes, and interpreted values
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_data_at_address(address, size)
            
        # Comment management tools
        @mcp.tool()
        def set_comment(filename: str, address: str, comment: str, ctx: Context) -> str:
            """Set a comment at the specified address

            Args:
                filename: Name of the binary file
                address: Address in hex format
                comment: Comment text

            Returns:
                Success message
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.set_comment(address, comment)
            
        @mcp.tool()
        def get_comment(filename: str, address: str, ctx: Context) -> Optional[str]:
            """Get comment at the specified address

            Args:
                filename: Name of the binary file
                address: Address in hex format

            Returns:
                Comment text or None if no comment exists
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_comment(address)
            
        @mcp.tool()
        def get_all_comments(filename: str, ctx: Context) -> dict:
            """Get all comments in the binary

            Args:
                filename: Name of the binary file

            Returns:
                List of all comments with addresses and types
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_all_comments()
            
        @mcp.tool()
        def remove_comment(filename: str, address: str, ctx: Context) -> str:
            """Remove comment at the specified address

            Args:
                filename: Name of the binary file
                address: Address in hex format

            Returns:
                Success message
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.remove_comment(address)
            
        @mcp.tool()
        def set_function_comment(filename: str, function_name_or_address: str, comment: str, ctx: Context) -> str:
            """Set a comment for an entire function

            Args:
                filename: Name of the binary file
                function_name_or_address: Function name or address
                comment: Comment text

            Returns:
                Success message
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.set_function_comment(function_name_or_address, comment)
            
        # Variable management tools
        @mcp.tool()
        def create_variable(filename: str, function_name_or_address: str, var_name: str, var_type: str, ctx: Context, storage: str = "auto"):
            """Create a local variable in a function

            Args:
                filename: Name of the binary file
                function_name_or_address: Function name or address
                var_name: Variable name
                var_type: Variable type (e.g., 'int32_t', 'char*')
                storage: Storage type ('auto', 'register', etc.)

            Returns:
                Success message
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.create_variable(function_name_or_address, var_name, var_type, storage)
            
        @mcp.tool()
        def get_variables(filename: str, function_name_or_address: str, ctx: Context) -> list:
            """Get all variables in a function

            Args:
                filename: Name of the binary file
                function_name_or_address: Function name or address

            Returns:
                List of variables with their information
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_variables(function_name_or_address)
            
        @mcp.tool()
        def rename_variable(filename: str, function_name_or_address: str, old_name: str, new_name: str, ctx: Context) -> str:
            """Rename a variable in a function

            Args:
                filename: Name of the binary file
                function_name_or_address: Function name or address
                old_name: Current variable name
                new_name: New variable name

            Returns:
                Success message
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.rename_variable(function_name_or_address, old_name, new_name)
            
        @mcp.tool()
        def set_variable_type(filename: str, function_name_or_address: str, var_name: str, var_type: str, ctx: Context) -> str:
            """Set the type of a variable in a function

            Args:
                filename: Name of the binary file
                function_name_or_address: Function name or address
                var_name: Variable name
                var_type: New variable type (e.g., 'int32_t', 'char*')

            Returns:
                Success message
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.set_variable_type(function_name_or_address, var_name, var_type)
            
        # Type system tools
        @mcp.tool()
        def create_type(filename: str, name: str, definition: str, ctx: Context) -> str:
            """Create a new data type from a C-like definition

            Args:
                filename: Name of the binary file
                name: Name of the type
                definition: Type definition (e.g., 'struct { int x; int y; }', 'int*')

            Returns:
                Success message
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.create_type(name, definition)
            
        @mcp.tool()
        def get_types(filename: str, page_size: int = 100, page_number: int = 1, ctx: Context = None) -> dict:
            """Get all user-defined types with pagination

            Args:
                filename: Name of the binary file
                page_size: Number of types per page (default: 100)
                page_number: Page number starting from 1 (default: 1)

            Returns:
                Dictionary with types, page_size, page_number, total_count, and total_pages
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_types(page_size=page_size, page_number=page_number)
            
        @mcp.tool()
        def create_enum(filename: str, name: str, members: dict, ctx: Context) -> str:
            """Create an enumeration type

            Args:
                filename: Name of the binary file
                name: Name of the enum
                members: Dictionary of member names to values

            Returns:
                Success message
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.create_enum(name, members)
            
        @mcp.tool()
        def create_typedef(filename: str, name: str, base_type: str, ctx: Context):
            """Create a type alias (typedef)

            Args:
                filename: Name of the binary file
                name: Name of the typedef
                base_type: Base type to alias

            Returns:
                Success message
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.create_typedef(name, base_type)
            
        @mcp.tool()
        def get_type_info(filename: str, type_name: str, ctx: Context):
            """Get detailed information about a specific type

            Args:
                filename: Name of the binary file
                type_name: Name of the type

            Returns:
                Dictionary with type information including members, size, and category
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_type_info(type_name)
            
        # Function analysis tools
        @mcp.tool()
        def get_call_graph(filename: str, ctx: Context, function_name_or_address: str = ""):
            """Get call graph information for a function or entire binary

            Args:
                filename: Name of the binary file
                function_name_or_address: Optional function name or address (if empty, returns global call graph)

            Returns:
                Call graph information with caller/callee relationships
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            # Convert empty string back to None for the underlying function
            func_param = function_name_or_address if function_name_or_address else None
            return tools.get_call_graph(func_param)
            
        @mcp.tool()
        def analyze_function(filename: str, function_name_or_address: str, ctx: Context):
            """Perform comprehensive analysis of a function

            Args:
                filename: Name of the binary file
                function_name_or_address: Function name or address

            Returns:
                Comprehensive function analysis including control flow, complexity, and call information
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.analyze_function(function_name_or_address)
            
        @mcp.tool()
        def get_cross_references(filename: str, address_or_name: str, ctx: Context):
            """Get cross-references for a function or address

            Args:
                filename: Name of the binary file
                address_or_name: Function name or address

            Returns:
                Cross-reference information showing where the address is referenced
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_cross_references(address_or_name)
            
        # Enhanced function listing tools
        @mcp.tool()
        def get_functions_advanced(filename: str, ctx: Context,
                                   name_filter: str = "",
                                   min_size: int = 0,
                                   max_size: int = 0,
                                   has_parameters: bool = False,
                                   sort_by: str = "address",
                                   limit: int = 0):
            """Get functions with advanced filtering and search capabilities

            Args:
                filename: Name of the binary file
                name_filter: Filter by function name (substring match)
                min_size: Minimum function size in bytes
                max_size: Maximum function size in bytes
                has_parameters: Filter by whether function has parameters
                sort_by: Sort by 'address', 'name', 'size', or 'complexity'
                limit: Maximum number of results

            Returns:
                Filtered and sorted list of functions
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            # Convert empty/zero values back to None for the underlying function
            name_filter_val = name_filter if name_filter else None
            min_size_val = min_size if min_size > 0 else None
            max_size_val = max_size if max_size > 0 else None
            has_parameters_val = has_parameters if has_parameters else None
            limit_val = limit if limit > 0 else None
            return tools.get_functions_advanced(name_filter_val, min_size_val, max_size_val, has_parameters_val, sort_by, limit_val)
            
        @mcp.tool()
        def search_functions_advanced(filename: str, search_term: str, ctx: Context,
                                      search_in: str = "name",
                                      case_sensitive: bool = False):
            """Advanced function search with multiple search targets

            Args:
                filename: Name of the binary file
                search_term: Term to search for
                search_in: Where to search ('name', 'comment', 'calls', 'variables')
                case_sensitive: Whether search should be case sensitive

            Returns:
                List of matching functions
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.search_functions_advanced(search_term, search_in, case_sensitive)
            
        @mcp.tool()
        def get_function_statistics(filename: str, ctx: Context):
            """Get comprehensive statistics about all functions in the binary

            Args:
                filename: Name of the binary file

            Returns:
                Statistics including size, complexity, parameters, and top functions
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_function_statistics()
            
        # Current context tools
        @mcp.tool()
        def get_current_address(filename: str, ctx: Context):
            """Get the current address/offset in the binary view

            Args:
                filename: Name of the binary file

            Returns:
                Dictionary containing current address information with context
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_current_address()
            
        @mcp.tool()
        def get_current_function(filename: str, ctx: Context):
            """Get the current function (function containing the current address)

            Args:
                filename: Name of the binary file

            Returns:
                Dictionary containing current function name and address
            """
            context_manager: BinAssistMCPBinaryContextManager = ctx.request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_current_function()
            
        log.log_info("Registered MCP tools")
        
    def _register_resources(self, mcp: FastMCP):
        """Register MCP resources"""
        
        @mcp.resource("binassist://{filename}/triage_summary")
        def get_triage_summary_resource(filename: str):
            """Get binary triage summary"""
            context_manager: BinAssistMCPBinaryContextManager = mcp.get_context().request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_triage_summary()
            
        @mcp.resource("binassist://{filename}/functions")
        def get_functions_resource(filename: str):
            """Get functions as a resource"""
            context_manager: BinAssistMCPBinaryContextManager = mcp.get_context().request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_functions()
            
        @mcp.resource("binassist://{filename}/imports")
        def get_imports_resource(filename: str):
            """Get imports as a resource"""
            context_manager: BinAssistMCPBinaryContextManager = mcp.get_context().request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_imports()
            
        @mcp.resource("binassist://{filename}/exports")
        def get_exports_resource(filename: str):
            """Get exports as a resource"""
            context_manager: BinAssistMCPBinaryContextManager = mcp.get_context().request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_exports()
            
        @mcp.resource("binassist://{filename}/strings")
        def get_strings_resource(filename: str):
            """Get strings as a resource"""
            context_manager: BinAssistMCPBinaryContextManager = mcp.get_context().request_context.lifespan_context
            binary_view = context_manager.get_binary(filename)
            tools = BinAssistMCPTools(binary_view)
            return tools.get_strings()
            
        log.log_info("Registered MCP resources")
        
    def start(self):
        """Start the MCP server with configured transports
        
        Returns:
            True if started successfully, False otherwise
        """
        if self._running:
            log.log_warn("Server is already running")
            return True
            
        try:
            log.log_info("Starting BinAssistMCP server...")
            
            # Also log to Binary Ninja
            try:
                import binaryninja as bn
                log.log_info("BinAssistMCP: Server.start() method called")
            except Exception as bn_log_error:
                log.log_error(f"Failed to log to Binary Ninja: {bn_log_error}")
                import traceback
                log.log_error(f"BN log traceback: {traceback.format_exc()}")
            
            # Validate configuration
            log.log_info("Validating configuration...")
            errors = self.config.validate()
            if errors:
                log.log_error(f"Configuration errors: {errors}")
                try:
                    import binaryninja as bn
                    log.log_error(f"BinAssistMCP configuration errors: {errors}")
                except Exception as bn_log_error:
                    log.log_error(f"Failed to log config errors to Binary Ninja: {bn_log_error}")
                    import traceback
                    log.log_error(f"BN log traceback: {traceback.format_exc()}")
                return False
            log.log_info("Configuration validation passed")
            
            try:
                import binaryninja as bn
                log.log_info("BinAssistMCP: Configuration validation passed")
            except Exception as bn_log_error:
                log.log_error(f"Failed to log validation success to Binary Ninja: {bn_log_error}")
                import traceback
                log.log_error(f"BN log traceback: {traceback.format_exc()}")
                
            # Create MCP server
            log.log_info("Creating MCP server instance...")
            self.mcp_server = self.create_mcp_server()
            log.log_info("MCP server instance created successfully")
            
            # Start SSE transport if enabled
            if self.config.is_transport_enabled(TransportType.SSE):
                log.log_info("SSE transport is enabled, starting SSE server...")
                self._start_sse_server()
            # Start Streamable HTTP transport if enabled
            elif self.config.is_transport_enabled(TransportType.STREAMABLEHTTP):
                log.log_info("Streamable HTTP transport is enabled, starting Streamable HTTP server...")
                self._start_streamablehttp_server()
            else:
                log.log_warn(f"Unknown transport type: {self.config.server.transport}")

            self._running = True
            log.log_info(f"BinAssistMCP server started successfully")
            log.log_info(f"Available transports: {self.config.server.transport.value}")

            if self.config.is_transport_enabled(TransportType.SSE):
                log.log_info(f"SSE endpoint: {self.config.get_sse_url()}")
            elif self.config.is_transport_enabled(TransportType.STREAMABLEHTTP):
                log.log_info(f"Streamable HTTP endpoint: {self.config.get_streamablehttp_url()}")
                
            return True
            
        except Exception as e:
            log.log_error(f"Failed to start server: {e}")
            # Also log to Binary Ninja if available
            try:
                import binaryninja as bn
                log.log_error(f"BinAssistMCP server startup failed: {e}")
                import traceback
                traceback_msg = traceback.format_exc()
                log.log_error(f"Server startup traceback: {traceback_msg}")
            except Exception as bn_log_error:
                log.log_error(f"Failed to log startup error to Binary Ninja: {bn_log_error}")
                import traceback
                log.log_error(f"BN log error traceback: {traceback.format_exc()}")
            self.stop()
            return False
            
    def _start_sse_server(self):
        """Start the SSE server thread with improved error handling"""
        if not self.mcp_server:
            raise RuntimeError("MCP server not created")

        try:
            # Create ASGI app for SSE transport
            log.log_info("Creating SSE ASGI app...")
            log.log_info(f"MCP server type: {type(self.mcp_server)}")

            # FastMCP 2.4.0+ uses sse_app() method
            if hasattr(self.mcp_server, 'sse_app'):
                log.log_info("Using FastMCP sse_app() method")
                asgi_app = self.mcp_server.sse_app()
            elif hasattr(self.mcp_server, 'create_asgi_app'):
                log.log_info("Using create_asgi_app method")
                asgi_app = self.mcp_server.create_asgi_app()
            elif hasattr(self.mcp_server, 'asgi'):
                log.log_info("Using asgi property")
                asgi_app = self.mcp_server.asgi
            elif hasattr(self.mcp_server, '_asgi_app'):
                log.log_info("Using _asgi_app property")
                asgi_app = self.mcp_server._asgi_app
            elif hasattr(self.mcp_server, 'app'):
                log.log_info("Using app property")
                asgi_app = self.mcp_server.app
            elif callable(self.mcp_server):
                log.log_info("MCP server is callable, using it directly as ASGI app")
                asgi_app = self.mcp_server
            else:
                # Let's see what attributes it actually has
                all_attrs = [attr for attr in dir(self.mcp_server) if not attr.startswith('__')]
                log.log_error(f"MCP server attributes: {all_attrs}")

                # Try to find any ASGI-like method
                asgi_methods = [attr for attr in all_attrs if 'asgi' in attr.lower() or 'app' in attr.lower()]
                log.log_error(f"Potential ASGI methods: {asgi_methods}")

                raise RuntimeError("Cannot create ASGI app for SSE transport")

            log.log_info(f"Created SSE ASGI app: {type(asgi_app)}")

            # Wrap the ASGI app with resource management
            wrapped_asgi_app = ResourceManagedASGIApp(asgi_app)
            log.log_info("Wrapped SSE ASGI app with error handling and resource management")

            self.sse_thread = SSEServerThread(wrapped_asgi_app, self.config)
            log.log_info(f"Created SSE server thread for {self.config.server.host}:{self.config.server.port}")
            log.log_info(f"SSE endpoint will be available at: {self.config.get_sse_url()}")

            self.sse_thread.start()
            log.log_info("SSE server thread started")

            # Give the thread a moment to start with better timing
            import time
            time.sleep(0.2)

            if self.sse_thread.is_alive():
                log.log_info("SSE server thread is running and ready for connections")
            else:
                log.log_error("SSE server thread failed to start")
                # Clean up the failed thread reference
                self.sse_thread = None
                raise RuntimeError("SSE server thread failed to start")

        except Exception as e:
            log.log_error(f"Failed to start SSE server: {e}")
            import traceback
            log.log_error(f"SSE startup traceback: {traceback.format_exc()}")
            # Clean up on failure
            if hasattr(self, 'sse_thread') and self.sse_thread:
                try:
                    self.sse_thread.stop()
                    self.sse_thread = None
                except Exception as cleanup_error:
                    log.log_error(f"Error cleaning up failed SSE server: {cleanup_error}")
            raise

    def _start_streamablehttp_server(self):
        """Start the Streamable HTTP server thread"""
        if not self.mcp_server:
            raise RuntimeError("MCP server not created")

        try:
            # Create ASGI app for Streamable HTTP transport
            log.log_info("Creating Streamable HTTP ASGI app...")

            if hasattr(self.mcp_server, 'streamable_http_app'):
                log.log_info("Using streamable_http_app method")
                asgi_app = self.mcp_server.streamable_http_app()
            else:
                raise RuntimeError("FastMCP does not have streamable_http_app method")

            log.log_info(f"Created Streamable HTTP ASGI app: {asgi_app}")

            # Wrap the ASGI app with resource management
            wrapped_asgi_app = ResourceManagedASGIApp(asgi_app)
            log.log_info("Wrapped Streamable HTTP ASGI app with resource management")

            self.streamablehttp_thread = SSEServerThread(wrapped_asgi_app, self.config)
            log.log_info(f"Created Streamable HTTP server thread for {self.config.server.host}:{self.config.server.port}")

            self.streamablehttp_thread.start()
            log.log_info("Streamable HTTP server thread started")

            # Give the thread a moment to start
            import time
            time.sleep(0.2)

            if self.streamablehttp_thread.is_alive():
                log.log_info("Streamable HTTP server thread is running")
            else:
                log.log_error("Streamable HTTP server thread failed to start")
                self.streamablehttp_thread = None
                raise RuntimeError("Streamable HTTP server thread failed to start")

        except Exception as e:
            log.log_error(f"Failed to start Streamable HTTP server: {e}")
            if hasattr(self, 'streamablehttp_thread') and self.streamablehttp_thread:
                try:
                    self.streamablehttp_thread.stop()
                    self.streamablehttp_thread = None
                except Exception as cleanup_error:
                    log.log_error(f"Error cleaning up failed Streamable HTTP server: {cleanup_error}")
            raise

    def stop(self):
        """Stop the MCP server"""
        if not self._running:
            log.log_warn("Server is not running")
            return
            
        log.log_info("Stopping BinAssistMCP server")
        
        try:
            # Stop SSE server with improved cleanup
            if self.sse_thread:
                log.log_info("Stopping SSE server thread")
                try:
                    self.sse_thread.stop()

                    # Wait for thread to finish with proper timeout
                    if self.sse_thread.is_alive():
                        self.sse_thread.join(timeout=10.0)

                    if self.sse_thread.is_alive():
                        log.log_warn("SSE server thread did not stop within 10 second timeout")
                    else:
                        log.log_info("SSE server thread stopped successfully")

                except Exception as stop_error:
                    log.log_error(f"Error stopping SSE server thread: {stop_error}")
                finally:
                    self.sse_thread = None

            # Stop Streamable HTTP server with improved cleanup
            if self.streamablehttp_thread:
                log.log_info("Stopping Streamable HTTP server thread")
                try:
                    self.streamablehttp_thread.stop()

                    # Wait for thread to finish with proper timeout
                    if self.streamablehttp_thread.is_alive():
                        self.streamablehttp_thread.join(timeout=10.0)

                    if self.streamablehttp_thread.is_alive():
                        log.log_warn("Streamable HTTP server thread did not stop within 10 second timeout")
                    else:
                        log.log_info("Streamable HTTP server thread stopped successfully")

                except Exception as stop_error:
                    log.log_error(f"Error stopping Streamable HTTP server thread: {stop_error}")
                finally:
                    self.streamablehttp_thread = None

            # Clear MCP server reference and force cleanup
            if self.mcp_server:
                log.log_info("Clearing MCP server reference")
                self.mcp_server = None
                
                # Force garbage collection to help clean up any lingering resources
                import gc
                gc.collect()
                log.log_debug("Forced garbage collection after MCP server cleanup")
                
        except Exception as e:
            log.log_error(f"Error during server shutdown: {e}")
        finally:
            self._running = False
            log.log_info("BinAssistMCP server stopped")
        
    def is_running(self):
        """Check if the server is running"""
        return self._running
        
        
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()
