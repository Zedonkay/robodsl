#!/usr/bin/env python3

def fix_brackets():
    with open('examples/example1.robodsl', 'r') as f:
        lines = f.readlines()
    
    # Find where the comprehensive_processor node should end
    # It should end after the cuda_kernels block
    for i, line in enumerate(lines):
        if line.strip() == '}' and i > 0:
            # Check if this closes a cuda_kernels block
            if lines[i-1].strip() == '}':
                # This is the closing brace of cuda_kernels
                # The next line should be the pipeline definition
                if i+1 < len(lines) and '// Pipeline definition' in lines[i+1]:
                    # Insert the missing closing brace for comprehensive_processor
                    lines.insert(i+1, '}\n')
                    break
    
    # Write the fixed file
    with open('examples/example1.robodsl', 'w') as f:
        f.writelines(lines)
    
    print("Fixed the missing closing brace for comprehensive_processor node")

if __name__ == '__main__':
    fix_brackets() 