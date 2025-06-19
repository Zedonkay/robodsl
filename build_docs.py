#!/usr/bin/env python3
"""
RoboDSL Documentation Builder

This script converts Markdown documentation files into HTML pages
using the existing website template.
"""
import os
import re
import markdown
from pathlib import Path
from bs4 import BeautifulSoup

# Configuration
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / 'docs'
TEMPLATE_FILE = DOCS_DIR / 'index.html'
OUTPUT_DIR = DOCS_DIR

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Files to convert (in order of appearance in navigation)
DOC_FILES = [
    'getting-started.md',  # Will be created from parts of developer_guide.md
    'developer-guide.md',
    'dsl-specification.md',
    'faq.md',             # Will be created from parts of dsl_specification.md
    'contributing.md',
    'code-of-conduct.md'
]

# Extract sections from existing files
SECTION_REGEX = {
    'getting-started': r'(?s)## Project Overview.*?(?=## Architecture)',
    'faq': r'(?s)## FAQ.*?(?=## Version)'
}

def load_template():
    """Load the HTML template."""
    with open(TEMPLATE_FILE, 'r', encoding='utf-8') as f:
        return f.read()

def extract_section(content, section_name):
    """Extract a specific section from markdown content."""
    if section_name in SECTION_REGEX:
        match = re.search(SECTION_REGEX[section_name], content)
        if match:
            return match.group(0).strip()
    return content

def convert_markdown(md_content):
    """Convert markdown to HTML."""
    # Convert markdown to HTML
    html = markdown.markdown(
        md_content,
        extensions=[
            'fenced_code',
            'tables',
            'toc',
            'codehilite',
            'md_in_html'
        ],
        extension_configs={
            'codehilite': {
                'linenums': True,
                'use_pygments': False
            }
        }
    )
    
    # Post-process HTML
    soup = BeautifulSoup(html, 'html.parser')
    
    # Add classes to tables
    for table in soup.find_all('table'):
        table['class'] = table.get('class', []) + ['table-auto', 'w-full', 'my-4']
    
    # Add classes to code blocks
    for pre in soup.find_all('pre'):
        pre['class'] = pre.get('class', []) + ['bg-gray-100', 'p-4', 'rounded', 'overflow-x-auto', 'my-4']
    
    return str(soup)

def generate_page(template, title, content, current_page=None):
    """Generate a complete HTML page."""
    # Update title
    page = template.replace('<title>RoboDSL Documentation</title>', f'<title>{title} | RoboDSL</title>')
    
    # Update active menu item
    if current_page:
        page = page.replace(f'<a href="{current_page}.html">', f'<a href="{current_page}.html" class="active">')
    
    # Update main content
    soup = BeautifulSoup(page, 'html.parser')
    main_content = soup.find('main')
    if main_content:
        main_content.clear()
        content_div = soup.new_tag('div')
        content_div['class'] = 'prose max-w-none'
        content_div.append(BeautifulSoup(content, 'html.parser'))
        main_content.append(content_div)
    
    # Add breadcrumbs
    breadcrumb = soup.new_tag('nav')
    breadcrumb['class'] = 'breadcrumb mb-6'
    breadcrumb.append(BeautifulSoup(
        f'<a href="index.html">Home</a> &gt; {title}',
        'html.parser'
    ))
    main_content.insert_before(breadcrumb)
    
    return str(soup)

def process_documentation():
    """Process all documentation files."""
    # Load template
    template = load_template()
    
    # Process each documentation file
    for doc_file in DOC_FILES:
        input_path = DOCS_DIR / doc_file
        output_path = OUTPUT_DIR / f"{Path(doc_file).stem}.html"
        
        # Special handling for generated files
        if doc_file == 'getting-started.md':
            # Use the beginning of developer-guide.md for getting started
            with open(DOCS_DIR / 'developer-guide.md', 'r', encoding='utf-8') as f:
                content = f.read()
                content = content.split('## Architecture')[0]  # Get content before Architecture section
        elif doc_file == 'faq.md':
            # Create a simple FAQ page
            content = """# Frequently Asked Questions

## General

### What is RoboDSL?
RoboDSL is a domain-specific language for building GPU-accelerated robotics applications.

### How do I get started?
Check out our [Getting Started](getting-started.html) guide.

## Development

### How do I contribute?
Please see our [Contributing Guidelines](contributing.html).

### What's the code of conduct?
We follow the [Contributor Covenant](code-of-conduct.html)."""
        elif input_path.exists():
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            print(f"Warning: {input_path} not found, skipping...")
            continue
        
        # Convert markdown to HTML
        html_content = convert_markdown(content)
        
        # Generate complete page
        title = ' '.join(word.capitalize() for word in Path(doc_file).stem.split('-'))
        page_html = generate_page(template, title, html_content, Path(doc_file).stem)
        
        # Write output file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(page_html)
        
        print(f"Generated: {output_path}")

if __name__ == '__main__':
    process_documentation()
