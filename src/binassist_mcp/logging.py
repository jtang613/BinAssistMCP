"""
Centralized logging utilities for BinAssistMCP using Binary Ninja's Logger
"""

import logging

try:
    import binaryninja
    log = binaryninja.log.Logger(0, "BinAssistMCP")
    
    # Suppress logging from external libraries to avoid ScriptingProvider messages
    def setup_logging_filters():
        """Configure logging to suppress external library messages"""
        # Disable or redirect common external loggers
        external_loggers = [
            'hypercorn',
            'hypercorn.error', 
            'hypercorn.access',
            'uvicorn',
            'uvicorn.error',
            'uvicorn.access',
            'mcp',
            'httpx',
            'fastapi'
        ]
        
        for logger_name in external_loggers:
            ext_logger = logging.getLogger(logger_name)
            ext_logger.setLevel(logging.CRITICAL)  # Only show critical errors
            ext_logger.propagate = False
            
    # Setup filters when Binary Ninja is available
    setup_logging_filters()
    
    def disable_external_logging():
        """Completely disable external library logging"""
        # Set root logger to critical to suppress most messages
        logging.getLogger().setLevel(logging.CRITICAL)
        
        # Disable specific loggers completely
        external_loggers = [
            'hypercorn', 'hypercorn.error', 'hypercorn.access',
            'uvicorn', 'uvicorn.error', 'uvicorn.access', 
            'mcp', 'mcp.client', 'mcp.server',
            'httpx', 'fastapi', 'starlette'
        ]
        
        for logger_name in external_loggers:
            ext_logger = logging.getLogger(logger_name)
            ext_logger.disabled = True
            ext_logger.propagate = False
    
except ImportError:
    # Fallback for testing outside Binary Ninja
    class MockLog:
        @staticmethod
        def log_info(msg): print(f"[BinAssistMCP] INFO: {msg}")
        @staticmethod
        def log_error(msg): print(f"[BinAssistMCP] ERROR: {msg}")
        @staticmethod
        def log_warn(msg): print(f"[BinAssistMCP] WARN: {msg}")
        @staticmethod
        def log_debug(msg): print(f"[BinAssistMCP] DEBUG: {msg}")
    log = MockLog()
    
    def disable_external_logging():
        """Mock disable function for non-Binary Ninja environments"""
        pass