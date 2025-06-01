"""
BinAssist-MCP: Binary Ninja Plugin Entry Point

This file serves as the main entry point for the Binary Ninja plugin.
Binary Ninja requires this __init__.py file in the root directory to recognize the plugin.
"""

import logging

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

try:
    import binaryninja as bn
    
    # Try to import our plugin implementation
    try:
        from .src.binassist_mcp.plugin import BinAssistMCPPlugin
        
        # Initialize the plugin
        logger.info("Loading BinAssist-MCP plugin...")
        
        # The plugin will automatically register itself when imported
        plugin_instance = BinAssistMCPPlugin()
        
        # Set global plugin instance and register auto-startup callback
        from .src.binassist_mcp.plugin import set_plugin_instance
        set_plugin_instance(plugin_instance)
        
        # Register for binary analysis completion events (for auto-startup)
        def on_binaryview_analysis_completion(bv):
            """Handle binary analysis completion for auto-startup"""
            try:
                from .src.binassist_mcp.plugin import get_plugin_instance
                plugin = get_plugin_instance()
                if plugin:
                    plugin.handle_auto_startup(bv)
            except Exception as e:
                logger.error(f"Error in auto-startup callback: {e}")
                bn.log_error(f"BinAssist-MCP auto-startup error: {e}")
        
        # Register the callback with Binary Ninja
        bn.BinaryViewType.add_binaryview_initial_analysis_completion_event(
            on_binaryview_analysis_completion
        )
        
        logger.info("BinAssist-MCP plugin loaded successfully")
        bn.log_info("BinAssist-MCP plugin loaded successfully")
        bn.log_info("BinAssist-MCP auto-startup enabled - server will start automatically when analysis completes")
        
    except ImportError as import_err:
        logger.error(f"Failed to import BinAssist-MCP modules: {import_err}")
        bn.log_error(f"BinAssist-MCP plugin failed to load - missing dependencies: {import_err}")
        bn.log_info("To fix this, install BinAssist-MCP dependencies: pip install anyio hypercorn mcp trio pydantic pydantic-settings click")
        
    except Exception as plugin_err:
        logger.error(f"Failed to initialize BinAssist-MCP plugin: {plugin_err}")
        bn.log_error(f"BinAssist-MCP plugin initialization failed: {plugin_err}")
        
except ImportError:
    logger.error("Binary Ninja not available - this should only happen outside of Binary Ninja")
    
except Exception as e:
    logger.error(f"Unexpected error in BinAssist-MCP plugin loading: {e}")
    try:
        import binaryninja as bn
        bn.log_error(f"BinAssist-MCP plugin unexpected error: {e}")
    except:
        pass