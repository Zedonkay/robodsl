import re

content = '''cuda_kernel multi_gpu_advanced {
    kernel: |
        __global__ void multi_gpu_advanced(float* input, float* output, int size, int device_id) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < size) {
                // Device-specific computation
                float val = input[idx];
                val = val * (device_id + 1) + __sinf(val);
                output[idx] = val;
            }
        }
    block_size: 256
    grid_size: "(size + 255) / 256"
    inputs: ["input", "output", "size", "device_id"]
    outputs: ["output"]
    multi_gpu: true
    gpu_count: 1
}'''

print("Original content:")
print(repr(content))
print()

# Test the pattern
pattern = r'(kernel:\s*\|)(\s*\n(\s+)[^\n]*(\n\3[^\n]*)*)'
matches = list(re.finditer(pattern, content, re.MULTILINE))
print(f'Matches: {len(matches)}')

for i, match in enumerate(matches):
    print(f'Match {i}:')
    print(f'  Groups: {match.groups()}')
    print(f'  Start: {match.start()}, End: {match.end()}')
    print(f'  Full match: {repr(match.group(0))}')
    print() 