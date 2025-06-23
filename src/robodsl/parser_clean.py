"""Clean implementation of the RoboDSL parser without duplicates."""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Literal, Union
import re

# Keep all the dataclass definitions as they are
# [Previous dataclass definitions remain the same...]

def extract_section(content: str, section_name: str) -> Optional[str]:
    """Extract a section from node content.
    
    Args:
        content: The full node content
        section_name: Name of the section to extract (e.g., 'methods')
        
    Returns:
        Extracted section content or None if not found
    """
    pattern = rf'{section_name}\s*=\s*\[([\s\S]*?)\]'
    match = re.search(pattern, content, re.DOTALL)
    return match.group(1).strip() if match else None

def parse_methods_section(node: NodeConfig, methods_content: str) -> None:
    """Parse the methods section of a node configuration.
    
    Args:
        node: The node to add methods to
        methods_content: Content of the methods section
    """
    # Remove outer braces if present
    methods_content = methods_content.strip()
    if methods_content.startswith('{') and methods_content.endswith('}'):
        methods_content = methods_content[1:-1].strip()
    
    # Split into individual method blocks
    method_blocks = []
    current_block = []
    in_block = False
    brace_count = 0
    
    # Process each line to find method blocks
    for line in methods_content.split('\n'):
        line = line.strip()
        if not line:
            continue
            
        if not in_block and '{' in line:
            # Start of a new method block
            in_block = True
            current_block = []
            brace_count = 1
            # Add content after the opening brace if any
            after_brace = line[line.index('{')+1:].strip()
            if after_brace:
                current_block.append(after_brace)
        elif in_block:
            # Inside a method block
            current_block.append(line)
            brace_count += line.count('{')
            brace_count -= line.count('}')
            
            if brace_count <= 0:
                # End of method block
                block_content = '\n'.join(current_block)
                if block_content.endswith('}'):
                    block_content = block_content[:-1].strip()
                method_blocks.append(block_content)
                current_block = []
                in_block = False
    
    # Process each method block
    for block in method_blocks:
        try:
            method = parse_method_block(block)
            if method:
                node.methods.append(method)
        except Exception as e:
            print(f"Error parsing method block: {e}")
            import traceback
            traceback.print_exc()

def parse_method_block(block: str) -> Optional[CppMethodConfig]:
    """Parse a single method block into a CppMethodConfig.
    
    Args:
        block: The method block content
        
    Returns:
        CppMethodConfig if successful, None otherwise
    """
    # Implementation of parse_method_block
    # [Previous implementation remains the same...]
    pass

def parse_node(node_name: str, node_content: str) -> Optional[NodeConfig]:
    """Parse a single node configuration.
    
    Args:
        node_name: Name of the node
        node_content: Content of the node configuration
        
    Returns:
        NodeConfig if successful, None otherwise
    """
    try:
        node = NodeConfig(name=node_name)
        
        # Parse publishers (format: publisher /topic type)
        publisher_matches = re.finditer(r'publisher\s+([^\s]+)\s+([^\s\n]+)', node_content)
        for match in publisher_matches:
            topic = match.group(1).strip()
            msg_type = match.group(2).strip()
            node.publishers.append(PublisherConfig(topic=topic, msg_type=msg_type))
        
        # Parse subscribers (format: subscriber /topic type)
        subscriber_matches = re.finditer(r'subscriber\s+([^\s]+)\s+([^\s\n]+)', node_content)
        for match in subscriber_matches:
            topic = match.group(1).strip()
            msg_type = match.group(2).strip()
            node.subscribers.append(SubscriberConfig(topic=topic, msg_type=msg_type))
        
        # Parse services (format: service /service_name type)
        service_matches = re.finditer(r'service\s+([^\s]+)\s+([^\s\n]+)', node_content)
        for match in service_matches:
            service = match.group(1).strip()
            srv_type = match.group(2).strip()
            node.services.append(ServiceConfig(service=service, srv_type=srv_type))
        
        # Parse methods array if present
        methods_match = re.search(r'methods\s*=\s*\[([\s\S]*?)\](?:\s*;)?', node_content, re.DOTALL)
        if methods_match:
            methods_content = methods_match.group(1).strip()
            parse_methods_section(node, methods_content)
        
        return node
        
    except Exception as e:
        print(f"Error parsing node '{node_name}': {e}")
        import traceback
        traceback.print_exc()
        return None

# [Rest of the implementation remains the same...]
