# CUDA and Thrust Examples

This directory contains example projects demonstrating the use of CUDA and Thrust in RoboDSL.

## Vector Operations Example

### Overview
This example demonstrates how to use CUDA kernels and Thrust algorithms for vector operations.

### Files
- `vector_ops.robodsl`: Main DSL file defining the vector operations
- `CMakeLists.txt`: Build configuration
- `README.md`: This file

### Building and Running

1. Navigate to the example directory:
   ```bash
   cd examples/vector_ops
   ```

2. Generate the code:
   ```bash
   robodsl generate vector_ops.robodsl
   ```

3. Build the project:
   ```bash
   mkdir -p build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   ```

4. Run the example:
   ```bash
   ./vector_ops
   ```

### Example Output
```
[INFO] [vector_ops]: Initializing vector operations...
[INFO] [vector_ops]: Running vector addition...
[INFO] [vector_ops]: Result: [3.0, 5.0, 7.0, 9.0, 11.0]
[INFO] [vector_ops]: Running vector multiplication...
[INFO] [vector_ops]: Result: [2.0, 4.0, 6.0, 8.0, 10.0]
[INFO] [vector_ops]: Running vector reduction with Thrust...
[INFO] [vector_ops]: Sum: 15.0
```

## License
This example is part of the RoboDSL project and is licensed under the MIT License.
