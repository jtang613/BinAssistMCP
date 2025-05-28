"""
Command-line interface for BinAssist-MCP

This module provides CLI commands for running the MCP server in standalone mode,
managing configurations, and providing STDIO transport for MCP clients.
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Optional

import click

from .config import BinAssistMCPConfig, TransportType
from .server import BinAssistMCPServer

logger = logging.getLogger(__name__)

try:
    import binaryninja as bn
    BINJA_AVAILABLE = True
except ImportError:
    BINJA_AVAILABLE = False


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.option("--config", type=click.Path(exists=True, path_type=Path), help="Configuration file path")
@click.pass_context
def cli(ctx, debug: bool, config: Optional[Path]):
    """BinAssist-MCP: Comprehensive MCP server for Binary Ninja"""
    # Set up logging
    log_level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Load configuration
    if config:
        from .config import load_config_from_file
        ctx.ensure_object(dict)
        ctx.obj['config'] = load_config_from_file(config)
    else:
        ctx.ensure_object(dict)
        ctx.obj['config'] = BinAssistMCPConfig()
        
    if debug:
        ctx.obj['config'].debug = True
        ctx.obj['config']._setup_logging()


@cli.command()
@click.argument("binary_files", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--host", default="localhost", help="Server host address")
@click.option("--port", default=8000, type=int, help="Server port")
@click.option("--transport", type=click.Choice(["sse", "stdio", "both"]), default="both", help="Transport type")
@click.option("--max-binaries", default=10, type=int, help="Maximum concurrent binaries")
@click.pass_context
def serve(ctx, binary_files: List[Path], host: str, port: int, transport: str, max_binaries: int):
    """Start the MCP server with optional binary files"""
    if not BINJA_AVAILABLE:
        click.echo("Error: Binary Ninja not available. Please install the Binary Ninja Python API.", err=True)
        sys.exit(1)
        
    config: BinAssistMCPConfig = ctx.obj['config']
    
    # Override config with CLI options
    config.server.host = host
    config.server.port = port
    config.server.transport = TransportType(transport)
    config.binary.max_binaries = max_binaries
    
    # Validate configuration
    errors = config.validate()
    if errors:
        for error in errors:
            click.echo(f"Configuration error: {error}", err=True)
        sys.exit(1)
        
    # Create server
    server = BinAssistMCPServer(config)
    
    # Load binary files
    if binary_files:
        click.echo(f"Loading {len(binary_files)} binary files...")
        for binary_file in binary_files:
            try:
                binary_view = bn.open_view(str(binary_file))
                if binary_view:
                    server.add_initial_binary(binary_view)
                    click.echo(f"  Loaded: {binary_file}")
                else:
                    click.echo(f"  Failed to load: {binary_file}", err=True)
            except Exception as e:
                click.echo(f"  Error loading {binary_file}: {e}", err=True)
                
    # Start server
    click.echo(f"Starting BinAssist-MCP server...")
    click.echo(f"  Host: {config.server.host}")
    click.echo(f"  Port: {config.server.port}")
    click.echo(f"  Transport: {config.server.transport.value}")
    
    if server.start():
        try:
            if config.is_transport_enabled(TransportType.SSE):
                click.echo(f"  SSE endpoint: {config.get_sse_url()}")
            if config.is_transport_enabled(TransportType.STDIO):
                click.echo("  STDIO transport: Available")
                
            click.echo("\nServer started successfully. Press Ctrl+C to stop.")
            
            # Keep server running
            try:
                while True:
                    import time
                    time.sleep(1)
            except KeyboardInterrupt:
                click.echo("\nShutting down server...")
                
        finally:
            server.stop()
            click.echo("Server stopped.")
    else:
        click.echo("Failed to start server.", err=True)
        sys.exit(1)


@cli.command()
@click.option("--host", default="localhost", help="Server host address")
@click.option("--port", default=8000, type=int, help="Server port")
@click.pass_context
def stdio(ctx, host: str, port: int):
    """Run STDIO transport for MCP clients"""
    if not BINJA_AVAILABLE:
        click.echo("Error: Binary Ninja not available. Please install the Binary Ninja Python API.", err=True)
        sys.exit(1)
        
    config: BinAssistMCPConfig = ctx.obj['config']
    config.server.host = host
    config.server.port = port
    config.server.transport = TransportType.STDIO
    
    # Create server with STDIO transport only
    server = BinAssistMCPServer(config)
    
    try:
        # Get STDIO server and run it
        mcp_server = server.get_stdio_server()
        
        # Run STDIO transport
        import trio
        from mcp.server.stdio import stdio_server
        
        async def run_stdio():
            async with stdio_server() as streams:
                await mcp_server.run(*streams)
                
        trio.run(run_stdio)
        
    except Exception as e:
        click.echo(f"STDIO transport error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def config(ctx):
    """Show current configuration"""
    config: BinAssistMCPConfig = ctx.obj['config']
    
    click.echo("BinAssist-MCP Configuration:")
    click.echo(f"  Debug: {config.debug}")
    click.echo(f"  Log Level: {config.log_level.value}")
    click.echo()
    click.echo("Server:")
    click.echo(f"  Host: {config.server.host}")
    click.echo(f"  Port: {config.server.port}")
    click.echo(f"  Transport: {config.server.transport.value}")
    click.echo(f"  Max Connections: {config.server.max_connections}")
    click.echo()
    click.echo("Binary Analysis:")
    click.echo(f"  Max Binaries: {config.binary.max_binaries}")
    click.echo(f"  Auto Analysis: {config.binary.auto_analysis}")
    click.echo(f"  Analysis Timeout: {config.binary.analysis_timeout}s")
    click.echo(f"  Cache Results: {config.binary.cache_results}")
    click.echo()
    click.echo("Plugin:")
    click.echo(f"  Auto Startup: {config.plugin.auto_startup}")
    click.echo(f"  Show Notifications: {config.plugin.show_notifications}")
    click.echo(f"  Menu Integration: {config.plugin.menu_integration}")


@cli.command()
@click.argument("binary_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def analyze(ctx, binary_file: Path):
    """Analyze a binary file and show basic information"""
    if not BINJA_AVAILABLE:
        click.echo("Error: Binary Ninja not available. Please install the Binary Ninja Python API.", err=True)
        sys.exit(1)
        
    try:
        click.echo(f"Analyzing {binary_file}...")
        
        # Open binary
        binary_view = bn.open_view(str(binary_file))
        if not binary_view:
            click.echo("Failed to open binary file.", err=True)
            sys.exit(1)
            
        # Create tools instance
        from .tools import BinAssistMCPTools
        tools = BinAssistMCPTools(binary_view)
        
        # Get triage summary
        summary = tools.get_triage_summary()
        
        click.echo("\nBinary Information:")
        file_info = summary["file_metadata"]
        click.echo(f"  Filename: {file_info['filename']}")
        click.echo(f"  File Size: {file_info['file_size']} bytes")
        click.echo(f"  View Type: {file_info['view_type']}")
        
        binary_info = summary["binary_info"]
        click.echo(f"  Platform: {binary_info['platform']}")
        click.echo(f"  Architecture: {binary_info.get('architecture', 'Unknown')}")
        click.echo(f"  Entry Point: {binary_info['entry_point']}")
        click.echo(f"  Base Address: {binary_info['base_address']}")
        click.echo(f"  End Address: {binary_info['end_address']}")
        
        stats = summary["statistics"]
        click.echo(f"\nStatistics:")
        click.echo(f"  Functions: {stats['function_count']}")
        click.echo(f"  Strings: {stats['string_count']}")
        click.echo(f"  Segments: {stats['segment_count']}")
        click.echo(f"  Sections: {stats['section_count']}")
        
    except Exception as e:
        click.echo(f"Analysis error: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def version(ctx):
    """Show version information"""
    from . import __version__
    click.echo(f"BinAssist-MCP version {__version__}")
    
    if BINJA_AVAILABLE:
        click.echo(f"Binary Ninja API: Available")
        try:
            click.echo(f"Binary Ninja Version: {bn.core_version}")
        except Exception as e:
            click.echo(f"Binary Ninja Version: Unknown (error: {e})")
    else:
        click.echo("Binary Ninja API: Not available")


@cli.command()
@click.pass_context  
def check(ctx):
    """Check system requirements and configuration"""
    click.echo("BinAssist-MCP System Check:")
    click.echo()
    
    # Check Binary Ninja
    if BINJA_AVAILABLE:
        click.echo("✓ Binary Ninja API: Available")
        try:
            click.echo(f"  Version: {bn.core_version}")
        except Exception as e:
            click.echo(f"  Version: Unknown (error: {e})")
    else:
        click.echo("✗ Binary Ninja API: Not available")
        click.echo("  Please install the Binary Ninja Python API")
    
    # Check dependencies
    click.echo()
    click.echo("Dependencies:")
    
    deps = [
        ("anyio", "anyio"),
        ("hypercorn", "hypercorn"),
        ("mcp", "mcp"),
        ("trio", "trio"),
        ("pydantic", "pydantic"),
    ]
    
    for name, module in deps:
        try:
            __import__(module)
            click.echo(f"✓ {name}: Available")
        except ImportError:
            click.echo(f"✗ {name}: Missing")
            
    # Check configuration
    click.echo()
    config: BinAssistMCPConfig = ctx.obj['config']
    errors = config.validate()
    if errors:
        click.echo("Configuration Issues:")
        for error in errors:
            click.echo(f"✗ {error}")
    else:
        click.echo("✓ Configuration: Valid")


def main():
    """Main entry point for the CLI"""
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nInterrupted by user.")
        sys.exit(1)
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()