"""
Comprehensive tool implementations for BinAssist-MCP

This module provides all the Binary Ninja integration tools.
"""

import functools
import logging
import re
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import binaryninja as bn
    BINJA_AVAILABLE = True
except ImportError:
    BINJA_AVAILABLE = False
    logger.warning("Binary Ninja not available")


def handle_exceptions(func):
    """Decorator to handle exceptions in tool methods"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper


def require_binja(func):
    """Decorator to ensure Binary Ninja is available"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not BINJA_AVAILABLE:
            raise RuntimeError("Binary Ninja is not available")
        return func(*args, **kwargs)
    return wrapper


class BinAssistMCPTools:
    """Comprehensive tool handler for Binary Ninja MCP tools"""
    
    def __init__(self, binary_view):
        """Initialize with a Binary Ninja BinaryView
        
        Args:
            binary_view: Binary Ninja BinaryView object
        """
        if not BINJA_AVAILABLE:
            raise RuntimeError("Binary Ninja is not available")
            
        self.bv = binary_view
        if not self.bv:
            raise ValueError("Binary view cannot be None")
            
    def _resolve_symbol(self, address_or_name: str) -> Optional[int]:
        """Resolve a symbol name or address to a numeric address
        
        Args:
            address_or_name: Either a hex address string or symbol name
            
        Returns:
            Numeric address if found, None otherwise
        """
        # Try to parse as hex address
        try:
            if isinstance(address_or_name, str) and address_or_name.startswith("0x"):
                return int(address_or_name, 16)
            return int(address_or_name, 16)
        except ValueError:
            pass
            
        # Try to parse as decimal address
        try:
            addr = int(address_or_name)
            if addr >= 0:
                return addr
        except ValueError:
            pass
            
        # Search by function name
        for func in self.bv.functions:
            if func.name == address_or_name:
                return func.start
                
        # Search by data variable name
        for addr, var in self.bv.data_vars.items():
            if hasattr(var, 'symbol') and var.symbol and var.symbol.name == address_or_name:
                return addr
                
        # Search by symbol name
        symbol = self.bv.get_symbol_by_raw_name(str(address_or_name))
        if symbol:
            return symbol.address
            
        return None
        
    def _get_function_by_name_or_address(self, identifier: Union[str, int]):
        """Get a function by name or address"""
        # Handle address-based lookup
        try:
            if isinstance(identifier, str) and identifier.startswith("0x"):
                addr = int(identifier, 16)
            elif isinstance(identifier, (int, str)):
                addr = int(identifier) if isinstance(identifier, str) else identifier
                
            func = self.bv.get_function_at(addr)
            if func:
                return func
        except ValueError:
            pass
            
        # Handle name-based lookup
        for func in self.bv.functions:
            if func.name == identifier:
                return func
                
        # Try case-insensitive match
        for func in self.bv.functions:
            if func.name.lower() == str(identifier).lower():
                return func
                
        # Try symbol lookup
        symbol = self.bv.get_symbol_by_raw_name(str(identifier))
        if symbol and symbol.address:
            func = self.bv.get_function_at(symbol.address)
            if func:
                return func
                
        return None
        
    # Core analysis tools
    @handle_exceptions
    @require_binja
    def rename_symbol(self, address_or_name: str, new_name: str) -> str:
        """Rename a function or data variable
        
        Args:
            address_or_name: Address (hex string) or name of the symbol
            new_name: New name for the symbol
            
        Returns:
            Success message string
        """
        addr = self._resolve_symbol(address_or_name)
        if addr is None:
            raise ValueError(f"No function or data variable found with name/address '{address_or_name}'")
            
        # Try to rename function
        func = self.bv.get_function_at(addr)
        if func:
            old_name = func.name
            func.name = new_name
            return f"Successfully renamed function at {hex(addr)} from '{old_name}' to '{new_name}'"
            
        # Try to rename data variable
        if addr in self.bv.data_vars:
            var = self.bv.data_vars[addr]
            old_name = var.symbol.name if var.symbol else 'unnamed'
            
            # Create a symbol at this address with the new name
            self.bv.define_user_symbol(bn.Symbol(bn.SymbolType.DataSymbol, addr, new_name))
            return f"Successfully renamed data variable at {hex(addr)} from '{old_name}' to '{new_name}'"
            
        raise ValueError(f"No function or data variable found at address {hex(addr)}")
        
    @handle_exceptions
    @require_binja
    def decompile_function(self, address_or_name: str) -> str:
        """Decompile a function to high-level representation
        
        Args:
            address_or_name: Function name or address
            
        Returns:
            Decompiled function code
        """
        func = self._get_function_by_name_or_address(address_or_name)
        if not func:
            raise ValueError(f"Function not found: {address_or_name}")
            
        # Ensure analysis is complete
        func.analysis_skipped = False
        self.bv.update_analysis_and_wait()
        
        # Try High Level IL first
        if hasattr(func, 'hlil') and func.hlil:
            return str(func.hlil)
        # Fall back to Medium Level IL
        elif hasattr(func, 'mlil') and func.mlil:
            return str(func.mlil)
        # Last resort: basic function representation
        else:
            return str(func)
            
    @handle_exceptions
    @require_binja
    def get_function_pseudo_c(self, address_or_name: str) -> str:
        """Get pseudo C code for a function
        
        Args:
            address_or_name: Function name or address
            
        Returns:
            Pseudo C code as string
        """
        addr = self._resolve_symbol(address_or_name)
        if addr is None:
            raise ValueError(f"No function found with name/address '{address_or_name}'")
            
        func = self.bv.get_function_at(addr)
        if not func:
            raise ValueError(f"No function found at address {hex(addr)}")
            
        lines = []
        settings = bn.DisassemblySettings()
        settings.set_option(bn.DisassemblyOption.ShowAddress, False)
        settings.set_option(bn.DisassemblyOption.WaitForIL, True)
        
        obj = bn.LinearViewObject.language_representation(self.bv, settings)
        cursor_end = bn.LinearViewCursor(obj)
        cursor_end.seek_to_address(func.highest_address)
        
        body = self.bv.get_next_linear_disassembly_lines(cursor_end)
        cursor_end.seek_to_address(func.highest_address)
        header = self.bv.get_previous_linear_disassembly_lines(cursor_end)
        
        for line in header:
            lines.append(f"{str(line)}\n")
        for line in body:
            lines.append(f"{str(line)}\n")
            
        return ''.join(lines)
        
    @handle_exceptions
    @require_binja
    def get_function_high_level_il(self, address_or_name: str) -> str:
        """Get High Level IL for a function
        
        Args:
            address_or_name: Function name or address
            
        Returns:
            HLIL as string
        """
        addr = self._resolve_symbol(address_or_name)
        if addr is None:
            raise ValueError(f"No function found with name/address '{address_or_name}'")
            
        func = self.bv.get_function_at(addr)
        if not func:
            raise ValueError(f"No function found at address {hex(addr)}")
            
        hlil = func.hlil
        if not hlil:
            raise ValueError(f"Failed to get HLIL for function at {hex(addr)}")
            
        lines = []
        for instruction in hlil.instructions:
            lines.append(f"{instruction.address:#x}: {instruction}\n")
            
        return ''.join(lines)
        
    @handle_exceptions
    @require_binja
    def get_function_medium_level_il(self, address_or_name: str) -> str:
        """Get Medium Level IL for a function
        
        Args:
            address_or_name: Function name or address
            
        Returns:
            MLIL as string
        """
        addr = self._resolve_symbol(address_or_name)
        if addr is None:
            raise ValueError(f"No function found with name/address '{address_or_name}'")
            
        func = self.bv.get_function_at(addr)
        if not func:
            raise ValueError(f"No function found at address {hex(addr)}")
            
        mlil = func.mlil
        if not mlil:
            raise ValueError(f"Failed to get MLIL for function at {hex(addr)}")
            
        lines = []
        for instruction in mlil.instructions:
            lines.append(f"{instruction.address:#x}: {instruction}\n")
            
        return ''.join(lines)
        
    @handle_exceptions
    @require_binja
    def get_disassembly(self, address_or_name: str, length: Optional[int] = None) -> str:
        """Get disassembly for a function or address range
        
        Args:
            address_or_name: Function name or start address
            length: Optional length in bytes for range disassembly
            
        Returns:
            Disassembly as string
        """
        addr = self._resolve_symbol(address_or_name)
        if addr is None:
            raise ValueError(f"No symbol found with name/address '{address_or_name}'")
            
        # Range disassembly if length specified
        if length is not None:
            disasm = []
            current_addr = addr
            remaining_length = length
            
            while remaining_length > 0 and current_addr < self.bv.end:
                instr_length = self.bv.get_instruction_length(current_addr)
                if instr_length == 0:
                    instr_length = 1
                    
                tokens = self.bv.get_disassembly(current_addr)
                if tokens:
                    disasm.append(f"{hex(current_addr)}: {tokens}")
                    
                current_addr += instr_length
                remaining_length -= instr_length
                
            if not disasm:
                raise ValueError(f"Failed to disassemble at address {hex(addr)} with length {length}")
            return '\n'.join(disasm)
            
        # Function disassembly
        func = self.bv.get_function_at(addr)
        if not func:
            raise ValueError(f"No function found at address {hex(addr)}")
            
        result_lines = []
        settings = bn.DisassemblySettings()
        settings.set_option(bn.DisassemblyOption.ShowAddress, True)
        
        obj = bn.LinearViewObject.single_function_disassembly(func, settings)
        cursor = bn.LinearViewCursor(obj)
        cursor.seek_to_begin()
        
        while not cursor.after_end:
            lines = self.bv.get_next_linear_disassembly_lines(cursor)
            if not lines:
                break
            for line in lines:
                result_lines.append(str(line))
                
        if not result_lines:
            raise ValueError(f"Failed to disassemble function at {hex(addr)}")
            
        return '\n'.join(result_lines)
        
    def _get_annotated_instruction(self, addr: int, instr_len: int) -> Optional[str]:
        """Get a single instruction with annotations"""
        try:
            # Get raw bytes
            raw_bytes = self.bv.read(addr, instr_len)
            hex_bytes = ' '.join(f'{b:02x}' for b in raw_bytes)
            
            # Get disassembly
            disasm_text = self.bv.get_disassembly(addr)
            if not disasm_text:
                disasm_text = hex_bytes + " ; [Raw bytes]"
                
            # Annotate call instructions
            if "call" in disasm_text.lower():
                addr_pattern = r'0x[0-9a-fA-F]+'
                match = re.search(addr_pattern, disasm_text)
                if match:
                    call_addr_str = match.group(0)
                    call_addr = int(call_addr_str, 16)
                    sym = self.bv.get_symbol_at(call_addr)
                    if sym and hasattr(sym, "name"):
                        disasm_text = disasm_text.replace(call_addr_str, sym.name)
                        
            # Get comment if any
            comment = self.bv.get_comment_at(addr)
            
            # Format final line
            line = f"0x{addr:08x}  {disasm_text}"
            if comment:
                line += f"  ; {comment}"
                
            return line
            
        except Exception as e:
            logger.debug(f"Error annotating instruction at {hex(addr)}: {e}")
            return f"0x{addr:08x}  {hex_bytes} ; [Error: {str(e)}]"
            
    # Information retrieval tools
    @handle_exceptions
    @require_binja
    def get_functions(self) -> List[Dict[str, Any]]:
        """Get list of all functions"""
        functions = []
        for func in self.bv.functions:
            functions.append({
                "name": func.name,
                "address": hex(func.start),
                "size": func.total_bytes,
                "symbol_type": str(func.symbol.type) if func.symbol else None,
                "parameter_count": len(func.parameter_vars),
                "return_type": str(func.return_type) if func.return_type else None,
                "basic_block_count": len(list(func.basic_blocks))
            })
        return functions
        
    @handle_exceptions
    @require_binja
    def search_functions_by_name(self, search_term: str) -> List[Dict[str, Any]]:
        """Search functions by name substring
        
        Args:
            search_term: Substring to search for
            
        Returns:
            List of matching functions
        """
        if not search_term:
            return []
            
        matches = []
        for func in self.bv.functions:
            if search_term.lower() in func.name.lower():
                matches.append({
                    "name": func.name,
                    "address": hex(func.start),
                    "symbol_type": str(func.symbol.type) if func.symbol else None
                })
                
        matches.sort(key=lambda x: x["name"])
        return matches
        
    @handle_exceptions
    @require_binja
    def get_imports(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get imported symbols grouped by module"""
        imports = {}
        
        for sym in self.bv.get_symbols_of_type(bn.SymbolType.ImportedFunctionSymbol):
            module = sym.namespace or 'unknown'
            if module not in imports:
                imports[module] = []
                
            imports[module].append({
                "name": sym.name,
                "address": hex(sym.address),
                "type": str(sym.type),
                "ordinal": getattr(sym, 'ordinal', None)
            })
            
        for sym in self.bv.get_symbols_of_type(bn.SymbolType.ImportedDataSymbol):
            module = sym.namespace or 'unknown'
            if module not in imports:
                imports[module] = []
                
            imports[module].append({
                "name": sym.name,
                "address": hex(sym.address),
                "type": str(sym.type),
                "ordinal": getattr(sym, 'ordinal', None)
            })
            
        return imports
        
    @handle_exceptions
    @require_binja
    def get_exports(self) -> List[Dict[str, Any]]:
        """Get exported symbols"""
        exports = []
        
        for sym in self.bv.get_symbols_of_type(bn.SymbolType.FunctionSymbol):
            if sym.binding == bn.SymbolBinding.GlobalBinding:
                exports.append({
                    "name": sym.name,
                    "address": hex(sym.address),
                    "type": str(sym.type),
                    "ordinal": getattr(sym, 'ordinal', None)
                })
                
        for sym in self.bv.get_symbols_of_type(bn.SymbolType.DataSymbol):
            if sym.binding == bn.SymbolBinding.GlobalBinding:
                exports.append({
                    "name": sym.name,
                    "address": hex(sym.address),
                    "type": str(sym.type),
                    "ordinal": getattr(sym, 'ordinal', None)
                })
                
        return exports
        
    @handle_exceptions
    @require_binja
    def get_strings(self) -> List[Dict[str, Any]]:
        """Get strings found in the binary"""
        strings = []
        for string in self.bv.strings:
            strings.append({
                "value": string.value,
                "address": hex(string.start),
                "length": string.length,
                "type": str(string.type)
            })
        return strings
        
    @handle_exceptions
    @require_binja
    def get_segments(self) -> List[Dict[str, Any]]:
        """Get memory segments"""
        segments = []
        for segment in self.bv.segments:
            segments.append({
                "start": hex(segment.start),
                "end": hex(segment.end),
                "length": segment.length,
                "readable": segment.readable,
                "writable": segment.writable,
                "executable": segment.executable,
                "data_offset": segment.data_offset,
                "data_length": segment.data_length
            })
        return segments
        
    @handle_exceptions
    @require_binja
    def get_sections(self) -> List[Dict[str, Any]]:
        """Get binary sections"""
        sections = []
        for section in self.bv.sections.values():
            sections.append({
                "name": section.name,
                "start": hex(section.start),
                "end": hex(section.end),
                "length": section.length,
                "type": section.type,
                "align": section.align,
                "entry_size": section.entry_size
            })
        return sections
        
    @handle_exceptions
    @require_binja
    def get_triage_summary(self) -> Dict[str, Any]:
        """Get binary triage summary"""
        return {
            "file_metadata": {
                "filename": self.bv.file.filename,
                "file_size": self.bv.length,
                "view_type": self.bv.view_type
            },
            "binary_info": {
                "platform": str(self.bv.platform),
                "architecture": self.bv.arch.name if self.bv.arch else None,
                "entry_point": hex(self.bv.entry_point),
                "base_address": hex(self.bv.start),
                "end_address": hex(self.bv.end),
                "endianness": self.bv.endianness.name,
                "address_size": self.bv.address_size
            },
            "statistics": {
                "function_count": len(list(self.bv.functions)),
                "string_count": len(list(self.bv.strings)),
                "segment_count": len(self.bv.segments),
                "section_count": len(self.bv.sections)
            }
        }
        
    @handle_exceptions
    @require_binja
    def update_analysis_and_wait(self) -> str:
        """Update analysis and wait for completion"""
        self.bv.update_analysis_and_wait()
        return f"Analysis updated successfully for {self.bv.file.filename}"
        
    # Class and namespace management tools
    @handle_exceptions
    @require_binja
    def get_classes(self) -> List[Dict[str, Any]]:
        """Get all classes/structs/types in the binary"""
        classes = []
        
        # Get all user-defined types
        for type_name, type_obj in self.bv.types.items():
            if isinstance(type_obj, (bn.StructureType, bn.ClassType)):
                members = []
                for member in type_obj.members:
                    members.append({
                        "name": member.name,
                        "type": str(member.type),
                        "offset": member.offset
                    })
                    
                classes.append({
                    "name": type_name,
                    "type": "class" if isinstance(type_obj, bn.ClassType) else "struct",
                    "size": type_obj.width,
                    "members": members,
                    "member_count": len(members)
                })
                
        return classes
        
    @handle_exceptions
    @require_binja
    def create_class(self, name: str, size: int) -> str:
        """Create a new class/struct type
        
        Args:
            name: Name of the class/struct
            size: Size in bytes
            
        Returns:
            Success message
        """
        if name in self.bv.types:
            raise ValueError(f"Type '{name}' already exists")
            
        # Create empty structure
        struct = bn.StructureBuilder.create()
        struct.width = size
        
        # Define the type
        self.bv.define_user_type(name, struct)
        return f"Successfully created class/struct '{name}' with size {size} bytes"
        
    @handle_exceptions
    @require_binja
    def add_class_member(self, class_name: str, member_name: str, member_type: str, offset: int) -> str:
        """Add a member to an existing class/struct
        
        Args:
            class_name: Name of the class/struct
            member_name: Name of the member
            member_type: Type of the member (e.g., 'int32_t', 'char*')
            offset: Offset within the struct
            
        Returns:
            Success message
        """
        if class_name not in self.bv.types:
            raise ValueError(f"Class/struct '{class_name}' not found")
            
        struct_type = self.bv.types[class_name]
        if not isinstance(struct_type, (bn.StructureType, bn.ClassType)):
            raise ValueError(f"'{class_name}' is not a class or struct")
            
        # Parse the member type
        try:
            parsed_type = self.bv.parse_type_string(member_type)[0]
        except Exception as e:
            raise ValueError(f"Invalid type '{member_type}': {str(e)}")
            
        # Create new structure with the added member
        struct = bn.StructureBuilder.create(struct_type)
        struct.insert(offset, parsed_type, member_name)
        
        # Update the type
        self.bv.define_user_type(class_name, struct)
        return f"Successfully added member '{member_name}' to '{class_name}' at offset {offset}"
        
    @handle_exceptions
    @require_binja
    def get_namespaces(self) -> List[Dict[str, Any]]:
        """Get all namespaces in the binary"""
        namespaces = {}
        
        # Collect all symbols and group by namespace
        for sym in self.bv.symbols.values():
            for symbol in sym:
                ns = symbol.namespace if symbol.namespace else "global"
                if ns not in namespaces:
                    namespaces[ns] = []
                    
                namespaces[ns].append({
                    "name": symbol.name,
                    "address": hex(symbol.address),
                    "type": str(symbol.type)
                })
                
        # Convert to list format
        result = []
        for ns_name, symbols in namespaces.items():
            result.append({
                "namespace": ns_name,
                "symbol_count": len(symbols),
                "symbols": symbols
            })
            
        return result
        
    # Advanced data management tools
    @handle_exceptions
    @require_binja
    def create_data_var(self, address: str, var_type: str, name: Optional[str] = None) -> str:
        """Create a data variable at the specified address
        
        Args:
            address: Address in hex format (e.g., '0x401000')
            var_type: Type of the variable (e.g., 'int32_t', 'char*')
            name: Optional name for the variable
            
        Returns:
            Success message
        """
        addr = self._resolve_symbol(address)
        if addr is None:
            raise ValueError(f"Invalid address: {address}")
            
        # Parse the type
        try:
            parsed_type = self.bv.parse_type_string(var_type)[0]
        except Exception as e:
            raise ValueError(f"Invalid type '{var_type}': {str(e)}")
            
        # Define the data variable
        self.bv.define_user_data_var(addr, parsed_type)
        
        # Set name if provided
        if name:
            symbol = bn.Symbol(bn.SymbolType.DataSymbol, addr, name)
            self.bv.define_user_symbol(symbol)
            
        return f"Successfully created data variable at {hex(addr)} with type '{var_type}'" + (f" named '{name}'" if name else "")
        
    @handle_exceptions
    @require_binja
    def get_data_vars(self) -> List[Dict[str, Any]]:
        """Get all data variables in the binary"""
        data_vars = []
        
        for addr, var in self.bv.data_vars.items():
            var_info = {
                "address": hex(addr),
                "type": str(var.type),
                "size": var.type.width if var.type else 0,
                "name": None
            }
            
            # Try to get symbol name
            symbol = self.bv.get_symbol_at(addr)
            if symbol:
                var_info["name"] = symbol.name
                
            data_vars.append(var_info)
            
        # Sort by address
        data_vars.sort(key=lambda x: int(x["address"], 16))
        return data_vars
        
    @handle_exceptions
    @require_binja
    def get_data_at_address(self, address: str, size: Optional[int] = None) -> Dict[str, Any]:
        """Get data at a specific address
        
        Args:
            address: Address in hex format
            size: Optional size to read (if not specified, uses data var size or default 16)
            
        Returns:
            Dictionary with data information
        """
        addr = self._resolve_symbol(address)
        if addr is None:
            raise ValueError(f"Invalid address: {address}")
            
        # Determine size to read
        read_size = size
        if not read_size:
            # Check if there's a data variable at this address
            if addr in self.bv.data_vars:
                var = self.bv.data_vars[addr]
                read_size = var.type.width if var.type else 16
            else:
                read_size = 16  # Default size
                
        # Read raw data
        try:
            raw_data = self.bv.read(addr, read_size)
        except Exception as e:
            raise ValueError(f"Failed to read data at {hex(addr)}: {str(e)}")
            
        # Get hex representation
        hex_data = ' '.join(f'{b:02x}' for b in raw_data)
        
        result = {
            "address": hex(addr),
            "size": read_size,
            "raw_hex": hex_data,
            "raw_bytes": list(raw_data)
        }
        
        # Try to interpret as different types
        if len(raw_data) >= 4:
            try:
                result["as_uint32"] = int.from_bytes(raw_data[:4], byteorder='little')
                result["as_int32"] = int.from_bytes(raw_data[:4], byteorder='little', signed=True)
            except:
                pass
                
        if len(raw_data) >= 8:
            try:
                result["as_uint64"] = int.from_bytes(raw_data[:8], byteorder='little')
                result["as_int64"] = int.from_bytes(raw_data[:8], byteorder='little', signed=True)
            except:
                pass
                
        # Try to interpret as string
        try:
            # Find null terminator or use all data
            null_pos = raw_data.find(0)
            str_data = raw_data[:null_pos] if null_pos != -1 else raw_data
            result["as_string"] = str_data.decode('utf-8', errors='replace')
        except:
            pass
            
        # Check if there's a defined data variable
        if addr in self.bv.data_vars:
            var = self.bv.data_vars[addr]
            result["defined_type"] = str(var.type)
            symbol = self.bv.get_symbol_at(addr)
            if symbol:
                result["symbol_name"] = symbol.name
                
        return result
        
    # Comment management tools
    @handle_exceptions
    @require_binja
    def set_comment(self, address: str, comment: str) -> str:
        """Set a comment at the specified address
        
        Args:
            address: Address in hex format
            comment: Comment text
            
        Returns:
            Success message
        """
        addr = self._resolve_symbol(address)
        if addr is None:
            raise ValueError(f"Invalid address: {address}")
            
        self.bv.set_comment_at(addr, comment)
        return f"Successfully set comment at {hex(addr)}: '{comment}'"
        
    @handle_exceptions
    @require_binja
    def get_comment(self, address: str) -> Optional[str]:
        """Get comment at the specified address
        
        Args:
            address: Address in hex format
            
        Returns:
            Comment text or None if no comment exists
        """
        addr = self._resolve_symbol(address)
        if addr is None:
            raise ValueError(f"Invalid address: {address}")
            
        return self.bv.get_comment_at(addr)
        
    @handle_exceptions
    @require_binja
    def get_all_comments(self) -> List[Dict[str, Any]]:
        """Get all comments in the binary"""
        comments = []
        
        # Get function-level comments
        for func in self.bv.functions:
            if func.comment:
                comments.append({
                    "address": hex(func.start),
                    "type": "function",
                    "comment": func.comment,
                    "function_name": func.name
                })
                
        # Get instruction-level comments (this is more complex as we need to iterate through all addresses)
        # We'll check comments in function ranges to be more efficient
        for func in self.bv.functions:
            addr = func.start
            while addr < func.highest_address:
                comment = self.bv.get_comment_at(addr)
                if comment:
                    comments.append({
                        "address": hex(addr),
                        "type": "instruction",
                        "comment": comment,
                        "function_name": func.name
                    })
                addr += self.bv.get_instruction_length(addr) or 1
                
        # Sort by address
        comments.sort(key=lambda x: int(x["address"], 16))
        return comments
        
    @handle_exceptions
    @require_binja
    def remove_comment(self, address: str) -> str:
        """Remove comment at the specified address
        
        Args:
            address: Address in hex format
            
        Returns:
            Success message
        """
        addr = self._resolve_symbol(address)
        if addr is None:
            raise ValueError(f"Invalid address: {address}")
            
        # Check if comment exists
        existing_comment = self.bv.get_comment_at(addr)
        if not existing_comment:
            return f"No comment found at {hex(addr)}"
            
        self.bv.set_comment_at(addr, "")
        return f"Successfully removed comment at {hex(addr)}"
        
    @handle_exceptions
    @require_binja
    def set_function_comment(self, function_name_or_address: str, comment: str) -> str:
        """Set a comment for an entire function
        
        Args:
            function_name_or_address: Function name or address
            comment: Comment text
            
        Returns:
            Success message
        """
        func = self._get_function_by_name_or_address(function_name_or_address)
        if not func:
            raise ValueError(f"Function not found: {function_name_or_address}")
            
        func.comment = comment
        return f"Successfully set comment for function '{func.name}': '{comment}'"
        
    # Variable management tools
    @handle_exceptions
    @require_binja
    def create_variable(self, function_name_or_address: str, var_name: str, var_type: str, storage: str = "auto") -> str:
        """Create a local variable in a function
        
        Args:
            function_name_or_address: Function name or address
            var_name: Variable name
            var_type: Variable type (e.g., 'int32_t', 'char*')
            storage: Storage type ('auto', 'register', etc.)
            
        Returns:
            Success message
        """
        func = self._get_function_by_name_or_address(function_name_or_address)
        if not func:
            raise ValueError(f"Function not found: {function_name_or_address}")
            
        # Parse the type
        try:
            parsed_type = self.bv.parse_type_string(var_type)[0]
        except Exception as e:
            raise ValueError(f"Invalid type '{var_type}': {str(e)}")
            
        # Create the variable (this is simplified - Binary Ninja's variable management is complex)
        # In practice, you might need to analyze the function's IL to determine proper variable placement
        var = bn.Variable.from_identifier(self.bv.arch, 0, var_name)  # Simplified approach
        
        # Try to set the variable type in the function
        try:
            func.create_user_var(var, parsed_type, var_name)
            return f"Successfully created variable '{var_name}' with type '{var_type}' in function '{func.name}'"
        except Exception as e:
            raise ValueError(f"Failed to create variable: {str(e)}")
            
    @handle_exceptions
    @require_binja
    def get_variables(self, function_name_or_address: str) -> List[Dict[str, Any]]:
        """Get all variables in a function
        
        Args:
            function_name_or_address: Function name or address
            
        Returns:
            List of variables with their information
        """
        func = self._get_function_by_name_or_address(function_name_or_address)
        if not func:
            raise ValueError(f"Function not found: {function_name_or_address}")
            
        variables = []
        
        # Get parameter variables
        for param in func.parameter_vars:
            variables.append({
                "name": param.name,
                "type": str(func.get_variable_type(param)) if func.get_variable_type(param) else "unknown",
                "category": "parameter",
                "storage": str(param.storage),
                "identifier": str(param.identifier)
            })
            
        # Get local variables
        for var in func.vars:
            if var not in func.parameter_vars:
                variables.append({
                    "name": var.name,
                    "type": str(func.get_variable_type(var)) if func.get_variable_type(var) else "unknown", 
                    "category": "local",
                    "storage": str(var.storage),
                    "identifier": str(var.identifier)
                })
                
        return variables
        
    @handle_exceptions
    @require_binja
    def rename_variable(self, function_name_or_address: str, old_name: str, new_name: str) -> str:
        """Rename a variable in a function
        
        Args:
            function_name_or_address: Function name or address
            old_name: Current variable name
            new_name: New variable name
            
        Returns:
            Success message
        """
        func = self._get_function_by_name_or_address(function_name_or_address)
        if not func:
            raise ValueError(f"Function not found: {function_name_or_address}")
            
        # Find the variable
        target_var = None
        for var in func.vars:
            if var.name == old_name:
                target_var = var
                break
                
        if not target_var:
            raise ValueError(f"Variable '{old_name}' not found in function '{func.name}'")
            
        # Rename the variable
        target_var.name = new_name
        return f"Successfully renamed variable from '{old_name}' to '{new_name}' in function '{func.name}'"
        
    @handle_exceptions
    @require_binja
    def set_variable_type(self, function_name_or_address: str, var_name: str, var_type: str) -> str:
        """Set the type of a variable in a function
        
        Args:
            function_name_or_address: Function name or address
            var_name: Variable name
            var_type: New variable type (e.g., 'int32_t', 'char*')
            
        Returns:
            Success message
        """
        func = self._get_function_by_name_or_address(function_name_or_address)
        if not func:
            raise ValueError(f"Function not found: {function_name_or_address}")
            
        # Find the variable
        target_var = None
        for var in func.vars:
            if var.name == var_name:
                target_var = var
                break
                
        if not target_var:
            raise ValueError(f"Variable '{var_name}' not found in function '{func.name}'")
            
        # Parse the type
        try:
            parsed_type = self.bv.parse_type_string(var_type)[0]
        except Exception as e:
            raise ValueError(f"Invalid type '{var_type}': {str(e)}")
            
        # Set the variable type
        func.create_user_var(target_var, parsed_type, var_name)
        return f"Successfully set type of variable '{var_name}' to '{var_type}' in function '{func.name}'"
        
    # Type system tools
    @handle_exceptions
    @require_binja
    def create_type(self, name: str, definition: str) -> str:
        """Create a new type definition
        
        Args:
            name: Name of the type
            definition: Type definition (e.g., 'struct { int x; int y; }', 'int*')
            
        Returns:
            Success message
        """
        if name in self.bv.types:
            raise ValueError(f"Type '{name}' already exists")
            
        # Parse the type definition
        try:
            parsed_type = self.bv.parse_type_string(definition)[0]
        except Exception as e:
            raise ValueError(f"Invalid type definition '{definition}': {str(e)}")
            
        # Define the type
        self.bv.define_user_type(name, parsed_type)
        return f"Successfully created type '{name}' with definition '{definition}'"
        
    @handle_exceptions
    @require_binja
    def get_types(self) -> List[Dict[str, Any]]:
        """Get all user-defined types"""
        types = []
        
        for type_name, type_obj in self.bv.types.items():
            type_info = {
                "name": type_name,
                "size": type_obj.width if hasattr(type_obj, 'width') else None,
                "category": self._get_type_category(type_obj),
                "definition": str(type_obj)
            }
            
            # Add additional info for complex types
            if isinstance(type_obj, (bn.StructureType, bn.ClassType)):
                type_info["member_count"] = len(type_obj.members) if hasattr(type_obj, 'members') else 0
            elif isinstance(type_obj, bn.EnumerationType):
                type_info["member_count"] = len(type_obj.members) if hasattr(type_obj, 'members') else 0
            elif isinstance(type_obj, bn.ArrayType):
                type_info["element_type"] = str(type_obj.element_type)
                type_info["count"] = type_obj.count
                
            types.append(type_info)
            
        return types
        
    def _get_type_category(self, type_obj) -> str:
        """Get the category of a type object"""
        if isinstance(type_obj, bn.StructureType):
            return "struct"
        elif isinstance(type_obj, bn.ClassType):
            return "class"
        elif isinstance(type_obj, bn.EnumerationType):
            return "enum"
        elif isinstance(type_obj, bn.ArrayType):
            return "array"
        elif isinstance(type_obj, bn.PointerType):
            return "pointer"
        elif isinstance(type_obj, bn.FunctionType):
            return "function"
        else:
            return "primitive"
            
    @handle_exceptions
    @require_binja
    def create_enum(self, name: str, members: Dict[str, int]) -> str:
        """Create an enumeration type
        
        Args:
            name: Name of the enum
            members: Dictionary of member names to values
            
        Returns:
            Success message
        """
        if name in self.bv.types:
            raise ValueError(f"Type '{name}' already exists")
            
        # Create enumeration
        enum_builder = bn.EnumerationBuilder.create()
        for member_name, value in members.items():
            enum_builder.append(member_name, value)
            
        # Define the type
        enum_type = bn.Type.enumeration_type(self.bv.arch, enum_builder, 4)  # 4-byte enum
        self.bv.define_user_type(name, enum_type)
        
        member_list = ', '.join(f"{k}={v}" for k, v in members.items())
        return f"Successfully created enum '{name}' with members: {member_list}"
        
    @handle_exceptions
    @require_binja
    def create_typedef(self, name: str, base_type: str) -> str:
        """Create a type alias (typedef)
        
        Args:
            name: Name of the typedef
            base_type: Base type to alias
            
        Returns:
            Success message
        """
        if name in self.bv.types:
            raise ValueError(f"Type '{name}' already exists")
            
        # Parse the base type
        try:
            parsed_type = self.bv.parse_type_string(base_type)[0]
        except Exception as e:
            raise ValueError(f"Invalid base type '{base_type}': {str(e)}")
            
        # Create named type
        named_type = bn.Type.named_type_from_type(name, parsed_type)
        self.bv.define_user_type(name, named_type)
        
        return f"Successfully created typedef '{name}' for type '{base_type}'"
        
    @handle_exceptions
    @require_binja
    def get_type_info(self, type_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific type
        
        Args:
            type_name: Name of the type
            
        Returns:
            Dictionary with type information
        """
        if type_name not in self.bv.types:
            raise ValueError(f"Type '{type_name}' not found")
            
        type_obj = self.bv.types[type_name]
        
        info = {
            "name": type_name,
            "category": self._get_type_category(type_obj),
            "size": type_obj.width if hasattr(type_obj, 'width') else None,
            "definition": str(type_obj)
        }
        
        # Add specific information based on type
        if isinstance(type_obj, (bn.StructureType, bn.ClassType)):
            info["members"] = []
            if hasattr(type_obj, 'members'):
                for member in type_obj.members:
                    info["members"].append({
                        "name": member.name,
                        "type": str(member.type),
                        "offset": member.offset,
                        "size": member.type.width if member.type else 0
                    })
                    
        elif isinstance(type_obj, bn.EnumerationType):
            info["members"] = []
            if hasattr(type_obj, 'members'):
                for member in type_obj.members:
                    info["members"].append({
                        "name": member.name,
                        "value": member.value
                    })
                    
        elif isinstance(type_obj, bn.ArrayType):
            info["element_type"] = str(type_obj.element_type)
            info["count"] = type_obj.count
            info["element_size"] = type_obj.element_type.width if type_obj.element_type else 0
            
        elif isinstance(type_obj, bn.PointerType):
            info["target_type"] = str(type_obj.target)
            info["pointer_size"] = type_obj.width
            
        elif isinstance(type_obj, bn.FunctionType):
            info["return_type"] = str(type_obj.return_value)
            info["parameters"] = []
            if hasattr(type_obj, 'parameters'):
                for param in type_obj.parameters:
                    info["parameters"].append({
                        "type": str(param.type),
                        "name": param.name if hasattr(param, 'name') else None
                    })
                    
        return info
        
    # Function analysis tools
    @handle_exceptions
    @require_binja
    def get_call_graph(self, function_name_or_address: Optional[str] = None) -> Dict[str, Any]:
        """Get call graph information for a function or entire binary
        
        Args:
            function_name_or_address: Optional function name or address (if None, returns global call graph)
            
        Returns:
            Call graph information
        """
        if function_name_or_address:
            # Single function call graph
            func = self._get_function_by_name_or_address(function_name_or_address)
            if not func:
                raise ValueError(f"Function not found: {function_name_or_address}")
                
            calls_to = []
            calls_from = []
            
            # Get functions this function calls
            for call_site in func.call_sites:
                called_func = self.bv.get_function_at(call_site.address)
                if called_func:
                    calls_to.append({
                        "function": called_func.name,
                        "address": hex(called_func.start),
                        "call_site": hex(call_site.address)
                    })
                    
            # Get functions that call this function
            for caller in func.callers:
                calls_from.append({
                    "function": caller.name,
                    "address": hex(caller.start)
                })
                
            return {
                "function": func.name,
                "address": hex(func.start),
                "calls_to": calls_to,
                "calls_from": calls_from,
                "call_count_out": len(calls_to),
                "call_count_in": len(calls_from)
            }
        else:
            # Global call graph
            call_graph = {}
            for func in self.bv.functions:
                calls = []
                for call_site in func.call_sites:
                    called_func = self.bv.get_function_at(call_site.address)
                    if called_func:
                        calls.append({
                            "target": called_func.name,
                            "address": hex(called_func.start)
                        })
                        
                call_graph[func.name] = {
                    "address": hex(func.start),
                    "calls": calls,
                    "call_count": len(calls)
                }
                
            return {"call_graph": call_graph, "function_count": len(call_graph)}
            
    @handle_exceptions
    @require_binja
    def analyze_function(self, function_name_or_address: str) -> Dict[str, Any]:
        """Perform comprehensive analysis of a function
        
        Args:
            function_name_or_address: Function name or address
            
        Returns:
            Comprehensive function analysis
        """
        func = self._get_function_by_name_or_address(function_name_or_address)
        if not func:
            raise ValueError(f"Function not found: {function_name_or_address}")
            
        # Basic function info
        analysis = {
            "name": func.name,
            "address": hex(func.start),
            "size": func.total_bytes,
            "basic_block_count": len(list(func.basic_blocks)),
            "instruction_count": sum(len(list(bb.instructions)) for bb in func.basic_blocks),
            "parameter_count": len(func.parameter_vars),
            "local_variable_count": len(func.vars) - len(func.parameter_vars),
            "complexity": {
                "cyclomatic": self._calculate_cyclomatic_complexity(func),
                "call_depth": len(list(func.call_sites))
            }
        }
        
        # Control flow analysis
        analysis["control_flow"] = {
            "entry_point": hex(func.start),
            "exit_points": [hex(bb.end) for bb in func.basic_blocks if len(bb.outgoing_edges) == 0],
            "branch_count": sum(1 for bb in func.basic_blocks if len(bb.outgoing_edges) > 1),
            "loop_count": self._count_loops(func)
        }
        
        # Call analysis
        calls_to = []
        for call_site in func.call_sites:
            called_func = self.bv.get_function_at(call_site.address)
            if called_func:
                calls_to.append(called_func.name)
                
        analysis["calls"] = {
            "outgoing": calls_to,
            "incoming": [caller.name for caller in func.callers],
            "external_calls": [call for call in calls_to if call.startswith("sub_") or "@" in call]
        }
        
        # Type information
        analysis["types"] = {
            "return_type": str(func.return_type) if func.return_type else "void",
            "parameters": [
                {
                    "name": param.name,
                    "type": str(func.get_variable_type(param)) if func.get_variable_type(param) else "unknown"
                }
                for param in func.parameter_vars
            ]
        }
        
        return analysis
        
    def _calculate_cyclomatic_complexity(self, func) -> int:
        """Calculate cyclomatic complexity for a function"""
        # Cyclomatic complexity = E - N + 2P
        # Where E = edges, N = nodes, P = connected components (usually 1)
        edges = sum(len(bb.outgoing_edges) for bb in func.basic_blocks)
        nodes = len(list(func.basic_blocks))
        return edges - nodes + 2
        
    def _count_loops(self, func) -> int:
        """Count the number of loops in a function"""
        # Simple heuristic: count back edges
        loop_count = 0
        visited = set()
        
        for bb in func.basic_blocks:
            for edge in bb.outgoing_edges:
                if edge.target.start <= bb.start and edge.target.start not in visited:
                    loop_count += 1
                visited.add(bb.start)
                
        return loop_count
        
    @handle_exceptions
    @require_binja
    def get_cross_references(self, address_or_name: str) -> Dict[str, Any]:
        """Get cross-references for a function or address
        
        Args:
            address_or_name: Function name or address
            
        Returns:
            Cross-reference information
        """
        addr = self._resolve_symbol(address_or_name)
        if addr is None:
            raise ValueError(f"Invalid address or symbol: {address_or_name}")
            
        xrefs_to = []
        xrefs_from = []
        
        # Get references TO this address
        for ref in self.bv.get_code_refs(addr):
            ref_func = self.bv.get_function_at(ref)
            xrefs_to.append({
                "address": hex(ref),
                "function": ref_func.name if ref_func else "unknown",
                "type": "code"
            })
            
        for ref in self.bv.get_data_refs(addr):
            ref_func = self.bv.get_function_at(ref)
            xrefs_to.append({
                "address": hex(ref),
                "function": ref_func.name if ref_func else "unknown", 
                "type": "data"
            })
            
        # Get references FROM this address (if it's a function)
        func = self.bv.get_function_at(addr)
        if func:
            for call_site in func.call_sites:
                called_func = self.bv.get_function_at(call_site.address)
                xrefs_from.append({
                    "address": hex(call_site.address),
                    "target": called_func.name if called_func else "unknown",
                    "type": "call"
                })
                
        return {
            "address": hex(addr),
            "symbol_name": address_or_name if not address_or_name.startswith("0x") else None,
            "references_to": xrefs_to,
            "references_from": xrefs_from,
            "total_refs_to": len(xrefs_to),
            "total_refs_from": len(xrefs_from)
        }
        
    # Enhanced function listing tools
    @handle_exceptions
    @require_binja
    def get_functions_advanced(self, 
                               name_filter: Optional[str] = None,
                               min_size: Optional[int] = None,
                               max_size: Optional[int] = None,
                               has_parameters: Optional[bool] = None,
                               sort_by: str = "address",
                               limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get functions with advanced filtering and search capabilities
        
        Args:
            name_filter: Filter by function name (substring match)
            min_size: Minimum function size in bytes
            max_size: Maximum function size in bytes
            has_parameters: Filter by whether function has parameters
            sort_by: Sort by 'address', 'name', 'size', or 'complexity'
            limit: Maximum number of results
            
        Returns:
            Filtered and sorted list of functions
        """
        functions = []
        
        for func in self.bv.functions:
            # Apply filters
            if name_filter and name_filter.lower() not in func.name.lower():
                continue
                
            if min_size is not None and func.total_bytes < min_size:
                continue
                
            if max_size is not None and func.total_bytes > max_size:
                continue
                
            if has_parameters is not None:
                func_has_params = len(func.parameter_vars) > 0
                if has_parameters != func_has_params:
                    continue
                    
            func_info = {
                "name": func.name,
                "address": hex(func.start),
                "size": func.total_bytes,
                "parameter_count": len(func.parameter_vars),
                "basic_block_count": len(list(func.basic_blocks)),
                "complexity": self._calculate_cyclomatic_complexity(func),
                "call_count": len(list(func.call_sites)),
                "caller_count": len(list(func.callers)),
                "return_type": str(func.return_type) if func.return_type else "void"
            }
            
            functions.append(func_info)
            
        # Sort functions
        if sort_by == "name":
            functions.sort(key=lambda x: x["name"].lower())
        elif sort_by == "size":
            functions.sort(key=lambda x: x["size"], reverse=True)
        elif sort_by == "complexity":
            functions.sort(key=lambda x: x["complexity"], reverse=True)
        else:  # default to address
            functions.sort(key=lambda x: int(x["address"], 16))
            
        # Apply limit
        if limit is not None:
            functions = functions[:limit]
            
        return functions
        
    @handle_exceptions
    @require_binja
    def search_functions_advanced(self, 
                                  search_term: str,
                                  search_in: str = "name",
                                  case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """Advanced function search with multiple search targets
        
        Args:
            search_term: Term to search for
            search_in: Where to search ('name', 'comment', 'calls', 'variables')
            case_sensitive: Whether search should be case sensitive
            
        Returns:
            List of matching functions
        """
        if not search_term:
            return []
            
        matches = []
        search_lower = search_term.lower() if not case_sensitive else search_term
        
        for func in self.bv.functions:
            match_found = False
            match_reason = []
            
            if search_in in ["name", "all"]:
                func_name = func.name if case_sensitive else func.name.lower()
                if search_lower in func_name:
                    match_found = True
                    match_reason.append("name")
                    
            if search_in in ["comment", "all"]:
                if func.comment:
                    comment = func.comment if case_sensitive else func.comment.lower()
                    if search_lower in comment:
                        match_found = True
                        match_reason.append("comment")
                        
            if search_in in ["calls", "all"]:
                for call_site in func.call_sites:
                    called_func = self.bv.get_function_at(call_site.address)
                    if called_func:
                        called_name = called_func.name if case_sensitive else called_func.name.lower()
                        if search_lower in called_name:
                            match_found = True
                            match_reason.append("calls")
                            break
                            
            if search_in in ["variables", "all"]:
                for var in func.vars:
                    var_name = var.name if case_sensitive else var.name.lower()
                    if search_lower in var_name:
                        match_found = True
                        match_reason.append("variables")
                        break
                        
            if match_found:
                matches.append({
                    "name": func.name,
                    "address": hex(func.start),
                    "size": func.total_bytes,
                    "match_reason": match_reason,
                    "comment": func.comment if func.comment else None
                })
                
        # Sort by relevance (name matches first, then others)
        matches.sort(key=lambda x: (
            0 if "name" in x["match_reason"] else 1,
            x["name"].lower()
        ))
        
        return matches
        
    @handle_exceptions
    @require_binja
    def get_function_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about all functions in the binary"""
        if not self.bv.functions:
            return {"error": "No functions found in binary"}
            
        sizes = [func.total_bytes for func in self.bv.functions]
        complexities = [self._calculate_cyclomatic_complexity(func) for func in self.bv.functions]
        param_counts = [len(func.parameter_vars) for func in self.bv.functions]
        bb_counts = [len(list(func.basic_blocks)) for func in self.bv.functions]
        
        return {
            "total_functions": len(list(self.bv.functions)),
            "size_statistics": {
                "min": min(sizes),
                "max": max(sizes),
                "average": sum(sizes) / len(sizes),
                "total": sum(sizes)
            },
            "complexity_statistics": {
                "min": min(complexities),
                "max": max(complexities),
                "average": sum(complexities) / len(complexities)
            },
            "parameter_statistics": {
                "min": min(param_counts),
                "max": max(param_counts),
                "average": sum(param_counts) / len(param_counts),
                "functions_with_params": sum(1 for count in param_counts if count > 0)
            },
            "basic_block_statistics": {
                "min": min(bb_counts),
                "max": max(bb_counts),
                "average": sum(bb_counts) / len(bb_counts),
                "total": sum(bb_counts)
            },
            "top_largest_functions": [
                {"name": func.name, "address": hex(func.start), "size": func.total_bytes}
                for func in sorted(self.bv.functions, key=lambda f: f.total_bytes, reverse=True)[:10]
            ],
            "top_most_complex_functions": [
                {"name": func.name, "address": hex(func.start), "complexity": self._calculate_cyclomatic_complexity(func)}
                for func in sorted(self.bv.functions, key=lambda f: self._calculate_cyclomatic_complexity(f), reverse=True)[:10]
            ]
        }