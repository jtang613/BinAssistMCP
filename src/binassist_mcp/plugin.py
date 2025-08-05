"""
Binary Ninja plugin integration for BinAssistMCP

This module provides the Binary Ninja plugin interface with menu integration,
settings management, and automatic server lifecycle management.
"""

from typing import Optional

from .logging import log, disable_external_logging

# Disable external library logging early to reduce ScriptingProvider messages
disable_external_logging()

try:
    import binaryninja as bn
    BINJA_AVAILABLE = True
except ImportError:
    BINJA_AVAILABLE = False
    log.log_warn("Binary Ninja not available")

if BINJA_AVAILABLE:
    try:
        from .config import BinAssistMCPConfig
        from .server import BinAssistMCPServer
    except ImportError as e:
        log.log_error(f"Failed to import BinAssistMCP modules: {e}")
        BINJA_AVAILABLE = False


class BinAssistMCPPlugin:
    """Binary Ninja plugin for BinAssistMCP"""
    
    def __init__(self):
        """Initialize the plugin"""
        self.config: Optional[BinAssistMCPConfig] = None
        self.server: Optional[BinAssistMCPServer] = None
        self._settings_registered = False
        
        if BINJA_AVAILABLE:
            self._initialize_plugin()
        
    def _initialize_plugin(self):
        """Initialize the plugin with Binary Ninja"""
        try:
            # Load configuration
            self.config = BinAssistMCPConfig()
            
            # Register settings if not already done
            if not self._settings_registered:
                self._register_settings()
                self._settings_registered = True
                
            # Update config from Binary Ninja settings
            self.config.update_from_binja_settings()
            
            # Register plugin commands
            self._register_commands()
            
            # Auto-startup is handled via global event registration in __init__.py
                
            log.log_info("BinAssistMCP plugin initialized successfully")
            
        except Exception as e:
            log.log_error(f"Failed to initialize plugin: {e}")
            if self.config and self.config.plugin.show_notifications:
                log.log_error(f"BinAssistMCP initialization failed: {e}")
                
    def _register_settings(self):
        """Register plugin settings with Binary Ninja"""
        try:
            settings = bn.Settings()
            
            # Server settings
            settings.register_setting(
                "binassistmcp.server.host",
                '{"description": "BinAssistMCP server host address", "title": "Server Host", "default": "localhost", "type": "string"}'
            )
            settings.register_setting(
                "binassistmcp.server.port",
                '{"description": "BinAssistMCP server port", "title": "Server Port", "default": 8000, "type": "number", "minValue": 1024, "maxValue": 65535}'
            )
            settings.register_setting(
                "binassistmcp.server.transport",
                '{"description": "MCP transport type", "title": "Transport Type", "default": "both", "type": "string", "enum": ["sse", "stdio", "both"]}'
            )
            
            # Plugin settings
            settings.register_setting(
                "binassistmcp.plugin.auto_startup",
                '{"description": "Automatically start server when Binary Ninja loads a file", "title": "Auto Startup", "default": true, "type": "boolean"}'
            )
            settings.register_setting(
                "binassistmcp.plugin.show_notifications",
                '{"description": "Show status notifications", "title": "Show Notifications", "default": true, "type": "boolean"}'
            )
            
            # Binary analysis settings
            settings.register_setting(
                "binassistmcp.binary.max_binaries",
                '{"description": "Maximum number of concurrent binaries", "title": "Max Binaries", "default": 10, "type": "number", "minValue": 1, "maxValue": 50}'
            )
            settings.register_setting(
                "binassistmcp.binary.auto_analysis",
                '{"description": "Enable automatic binary analysis", "title": "Auto Analysis", "default": true, "type": "boolean"}'
            )
            
            log.log_info("Registered BinAssistMCP settings")
            
        except Exception as e:
            log.log_error(f"Failed to register settings: {e}")
            
    def _register_commands(self):
        """Register plugin menu commands"""
        try:
            # Start server command
            bn.PluginCommand.register(
                "BinAssistMCP\\Start Server",
                "Start the BinAssistMCP server",
                self._start_server_command
            )
            
            # Stop server command
            bn.PluginCommand.register(
                "BinAssistMCP\\Stop Server", 
                "Stop the BinAssistMCP server",
                self._stop_server_command
            )
            
            # Restart server command
            bn.PluginCommand.register(
                "BinAssistMCP\\Restart Server",
                "Restart the BinAssistMCP server",
                self._restart_server_command
            )
            
            # Server status command
            bn.PluginCommand.register(
                "BinAssistMCP\\Server Status",
                "Show BinAssistMCP server status",
                self._server_status_command
            )
            
            # Configuration command
            bn.PluginCommand.register(
                "BinAssistMCP\\Open Settings",
                "Open BinAssistMCP settings",
                self._open_settings_command
            )
            
            log.log_info("Registered BinAssistMCP menu commands")
            
        except Exception as e:
            log.log_error(f"Failed to register commands: {e}")
            
    def handle_auto_startup(self, binary_view):
        """Handle auto-startup when a binary is analyzed"""
        try:
            if not self.config or not self.config.plugin.auto_startup:
                return
                
            # Start server if not running
            if not self.server or not self.server.is_running():
                log.log_info("Auto-startup triggered: starting BinAssistMCP server")
                self._start_server_command(binary_view)
            else:
                # Add binary to existing server
                self.add_binary_to_server(binary_view)
                log.log_info("Auto-startup triggered: added binary to running server")
                
        except Exception as e:
            log.log_error(f"Error in auto-startup: {e}")
            
    def _start_server_command(self, bv):
        """Start server command handler"""
        try:
            if self.server and self.server.is_running():
                message = "BinAssistMCP server is already running"
                if self.config.plugin.show_notifications:
                    log.log_info(message)
                return
                
            # Reload configuration
            log.log_info("Reloading configuration...")
            self.config = BinAssistMCPConfig()
            self.config.update_from_binja_settings()
            log.log_info("Configuration reloaded successfully")
            
            # Create and start server
            log.log_info("Creating BinAssistMCP server instance...")
            self.server = BinAssistMCPServer(self.config)
            log.log_info("Server instance created successfully")
            
            # Add current binary if available
            if bv:
                self.server.add_initial_binary(bv)
                
            # Add detailed logging for server startup
            log.log_info(f"Attempting to start BinAssistMCP server...")
            log.log_info(f"  Configuration: {self.config.server.host}:{self.config.server.port}")
            log.log_info(f"  Transport: {self.config.server.transport.value}")
            
            log.log_info("Calling server.start()...")
            start_result = self.server.start()
            log.log_info(f"Server.start() returned: {start_result}")
            
            if start_result:
                message = f"BinAssistMCP server started on {self.config.get_server_url()}"
                if self.config.plugin.show_notifications:
                    log.log_info(message)
                    
                # Show transport information
                if self.config.server.transport.value == "both":
                    log.log_info(f"SSE endpoint: {self.config.get_sse_url()}")
                    log.log_info("STDIO transport: Available via CLI")
                elif self.config.server.transport.value == "sse":
                    log.log_info(f"SSE endpoint: {self.config.get_sse_url()}")
                elif self.config.server.transport.value == "stdio":
                    log.log_info("STDIO transport: Available via CLI")
                    
                # Try to verify server is listening
                import socket
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                        sock.settimeout(1)
                        result = sock.connect_ex((self.config.server.host, self.config.server.port))
                        if result == 0:
                            log.log_info("✓ Server is listening and accepting connections")
                        else:
                            log.log_warn(f"⚠ Server started but not accepting connections (connect result: {result})")
                except Exception as e:
                    log.log_warn(f"⚠ Could not verify server connectivity: {e}")
            else:
                message = "Failed to start BinAssistMCP server"
                log.log_error(message)
                
        except Exception as e:
            error_msg = f"Error starting server: {e}"
            log.log_error(error_msg)
            log.log_error(error_msg)
            # Also log the full traceback for debugging
            import traceback
            traceback_msg = traceback.format_exc()
            log.log_error(f"Server startup traceback: {traceback_msg}")
            log.log_error(f"Full error details: {traceback_msg}")
            
    def _stop_server_command(self, bv):
        """Stop server command handler"""
        try:
            if not self.server or not self.server.is_running():
                message = "BinAssistMCP server is not running"
                if self.config and self.config.plugin.show_notifications:
                    log.log_info(message)
                return
                
            self.server.stop()
            message = "BinAssistMCP server stopped"
            if self.config and self.config.plugin.show_notifications:
                log.log_info(message)
                
        except Exception as e:
            error_msg = f"Error stopping server: {e}"
            log.log_error(error_msg)
            log.log_error(error_msg)
            
    def _restart_server_command(self, bv):
        """Restart server command handler"""
        try:
            self._stop_server_command(bv)
            # Small delay to ensure clean shutdown
            import time
            time.sleep(0.5)
            self._start_server_command(bv)
            
        except Exception as e:
            error_msg = f"Error restarting server: {e}"
            log.log_error(error_msg)
            log.log_error(error_msg)
            
    def _server_status_command(self, bv):
        """Server status command handler"""
        try:
            if not self.server:
                log.log_info("BinAssistMCP server: Not initialized")
                return
                
            if self.server.is_running():
                log.log_info("BinAssistMCP server: Running")
                if self.config:
                    log.log_info(f"  Host: {self.config.server.host}")
                    log.log_info(f"  Port: {self.config.server.port}")
                    log.log_info(f"  Transport: {self.config.server.transport.value}")
                    from .config import TransportType
                    if self.config.is_transport_enabled(TransportType.SSE):
                        log.log_info(f"  SSE URL: {self.config.get_sse_url()}")
            else:
                log.log_info("BinAssistMCP server: Stopped")
                
        except Exception as e:
            error_msg = f"Error getting server status: {e}"
            log.log_error(error_msg)
            log.log_error(error_msg)
            
    def _open_settings_command(self, bv):
        """Open settings command handler"""
        try:
            # This will open the Binary Ninja settings dialog
            # Users can navigate to the BinAssist section
            log.log_info("BinAssistMCP settings can be found in Binary Ninja Settings under 'binassist' section")
            log.log_info("Use Ctrl+, (Cmd+, on Mac) to open settings")
            
        except Exception as e:
            error_msg = f"Error opening settings: {e}"
            log.log_error(error_msg)
            log.log_error(error_msg)
            
    def add_binary_to_server(self, binary_view):
        """Add a binary view to the running server"""
        if self.server and self.server.is_running():
            try:
                # Get the context manager and add the binary
                context_manager = getattr(self.server.mcp_server, '_context_manager', None)
                if context_manager:
                    name = context_manager.add_binary(binary_view)
                    if self.config and self.config.plugin.show_notifications:
                        log.log_info(f"Added binary '{name}' to BinAssistMCP server")
                        
            except Exception as e:
                log.log_error(f"Failed to add binary to server: {e}")


# Global plugin instance reference for callbacks
_plugin_instance: Optional[BinAssistMCPPlugin] = None

def set_plugin_instance(plugin: BinAssistMCPPlugin):
    """Set the global plugin instance for callback access"""
    global _plugin_instance
    _plugin_instance = plugin
    
def get_plugin_instance() -> Optional[BinAssistMCPPlugin]:
    """Get the global plugin instance"""
    return _plugin_instance


