"""
Code generation for RoboDSL.

This module handles the generation of C++ and CUDA source files from the parsed DSL configuration.
"""

import dataclasses
import os
import jinja2
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, fields, asdict

from .parser import RoboDSLConfig, NodeConfig, CudaKernelConfig

# Set up Jinja2 environment with the templates directory
template_dir = Path(__file__).parent / 'templates'
if template_dir.exists():
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        trim_blocks=True,
        lstrip_blocks=True
    )
    env.filters['indent'] = lambda text, n: '\n'.join(' ' * n + line if line.strip() else line for line in text.split('\n'))


@dataclass
class CodeGenerator:
    """Generates C++ and CUDA source code from RoboDSL configuration."""
    
    def __init__(self, config: RoboDSLConfig, output_dir: str = "."):
        """Initialize the code generator with the parsed DSL configuration.
        
        Args:
            config: The parsed RoboDSL configuration
            output_dir: Base directory for generated files (default: current directory)
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate(self) -> List[Path]:
        """Generate all source files from the DSL configuration.
        
        Returns:
            List of Path objects for all generated files
        """
        generated_files = []
        
        # Create output directories
        (self.output_dir / 'include' / 'robodsl').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'src').mkdir(exist_ok=True)
        
        # Generate node source and header files
        for node in self.config.nodes:
            header_path = self._generate_node_header(node)
            source_path = self._generate_node_source(node)
            if header_path:
                generated_files.append(header_path)
            if source_path:
                generated_files.append(source_path)
        
        # Generate CUDA kernel files
        for kernel in self.config.cuda_kernels:
            kernel_files = self._generate_cuda_kernel(kernel)
            if kernel_files:
                generated_files.extend(kernel_files)
        
        # Generate CMake configuration
        cmake_file = self._generate_cmakelists()
        if cmake_file:
            generated_files.append(cmake_file)
            
        return generated_files
    
    def _generate_node_source(self, node: NodeConfig) -> Path:
        """Generate a C++ source file for a ROS2 node.
        
        Args:
            node: The node configuration
            
        Returns:
            Path to the generated source file
        """
        # Convert node name to a valid C++ class name (e.g., 'my_node' -> 'MyNode')
        class_name = ''.join(word.capitalize() for word in node.name.split('_'))
        
        # Generate includes
        includes = [
            f'#include "robodsl/{node.name}_node.hpp"',
            '#include <functional>',
            '#include <memory>',
            '#include <string>',
            '#include <vector>',
            '#include <map>',
            '#include <utility>',
            '#include "rclcpp_components/register_node_macro.hpp"',
        ]
        
        # Add message includes for publishers and subscribers
        msg_includes = set()
        for pub in node.publishers:
            msg_includes.add(f'#include "{pub["msg_type"].replace(".", "/")}.hpp"')
        for sub in node.subscribers:
            msg_includes.add(f'#include "{sub["msg_type"].replace(".", "/")}.hpp"')
        
        # Add service includes
        for srv in node.services:
            msg_includes.add(f'#include "{srv["srv_type"].replace(".", "/")}.hpp"')
        
        # Sort includes for consistent output
        includes.extend(sorted(msg_includes))
        
        # Generate constructor implementation
        constructor = f"""{class_name}::{class_name}(const rclcpp::NodeOptions & options)
  : Node("{node.name}", options)
{{"""
        
        # Add parameter declarations
        if node.parameters:
            constructor += '\n    // Declare parameters\n'
            for param_name, default_value in node.parameters.items():
                # Determine parameter type from default value
                if default_value.lower() in ('true', 'false'):
                    param_type = 'bool'
                    default = 'true' if default_value.lower() == 'true' else 'false'
                    decl = f'this->declare_parameter<{param_type}>("{param_name}", {default});'
                elif default_value.replace('.', '', 1).isdigit():
                    if '.' in default_value:
                        param_type = 'double'
                        decl = f'this->declare_parameter<{param_type}>("{param_name}", {default_value});'
                    else:
                        param_type = 'int'
                        decl = f'this->declare_parameter<{param_type}>("{param_name}", {default_value});'
                else:
                    decl = f'this->declare_parameter<std::string>("{param_name}", "{default_value}");'
                
                constructor += f'    {decl}\n'
        
        # Initialize publishers
        if node.publishers:
            constructor += '\n    // Initialize publishers\n'
            for pub in node.publishers:
                topic = pub['topic']
                msg_type = pub['msg_type'].replace('/', '::')
                qos = '10'  # Default queue size
                var_name = f"{topic.lstrip('/').replace('/', '_')}_pub_"
                decl = f'{var_name} = this->create_publisher<{msg_type}>("{topic}", {qos});'
                constructor += f'    {decl}\n'
        
        # Initialize subscribers
        if node.subscribers:
            constructor += '\n    // Initialize subscribers\n'
            for sub in node.subscribers:
                topic = sub['topic']
                msg_type = sub['msg_type'].replace('/', '::')
                callback_name = f"{topic.lstrip('/').replace('/', '_')}_callback"
                var_name = f"{topic.lstrip('/').replace('/', '_')}_sub_"
                
                # Create callback
                callback = f'''void {class_name}::{callback_name}(const {msg_type}::SharedPtr msg) const
{{
    // Process incoming message
    RCLCPP_DEBUG(this->get_logger(), "Received message on topic %s", "{topic}");
    
    // TODO: Implement message processing
}}
'''
                
                # Create subscription
                qos = '10'  # Default queue size
                decl = f'''{var_name} = this->create_subscription<{msg_type}>(
        "{topic}", {qos}, std::bind(&{class_name}::{callback_name}, this, std::placeholders::_1));'''
                
                constructor += f'    {decl}\n'
        
        # Initialize services
        if node.services:
            constructor += '\n    // Initialize services\n'
            for srv in node.services:
                service_name = srv['service']
                srv_type = srv['srv_type'].replace('/', '::')
                callback_name = f"{service_name.lstrip('/').replace('/', '_')}_callback"
                var_name = f"{service_name.lstrip('/').replace('/', '_')}_srv_"
                
                # Create callback
                callback = f'''void {class_name}::{callback_name}(
    const std::shared_ptr<{srv_type}::Request> request,
    std::shared_ptr<{srv_type}::Response> response)
{{
    // Process service request
    RCLCPP_DEBUG(this->get_logger(), "Received service call on %s", "{service_name}");
    
    // TODO: Implement service logic
    (void)request;  // Avoid unused parameter warning
    (void)response; // Avoid unused parameter warning
}}
'''
                
                # Create service
                decl = f'''{var_name} = this->create_service<{srv_type}>(
        "{service_name}",
        std::bind(&{class_name}::{callback_name}, this, std::placeholders::_1, std::placeholders::_2));'''
                
                constructor += f'    {decl}\n'
        
        # Close constructor
        constructor += '}'
        
        # Generate destructor
        destructor = f"""{class_name}::~{class_name}()
{{
    // Cleanup resources if needed
}}"""
        
        # Generate register macro
        register_macro = f'RCLCPP_COMPONENTS_REGISTER_NODE(robodsl::{class_name})'
        
        # Combine all parts
        source_content = f"""// Generated by RoboDSL - DO NOT EDIT

{"\n".join(includes)}

namespace robodsl {{

{constructor}

{destructor}

}}  // namespace robodsl

{register_macro}
"""
        
        # Add callbacks after the class definition
        if node.subscribers or node.services:
            source_content += '\n// Callback implementations\n'
            if node.subscribers:
                for sub in node.subscribers:
                    topic = sub['topic']
                    msg_type = sub['msg_type'].replace('/', '::')
                    callback_name = f"{topic.lstrip('/').replace('/', '_')}_callback"
                    source_content += f'''
void {class_name}::{callback_name}(const {msg_type}::SharedPtr msg) const
{{
    // Process incoming message
    RCLCPP_DEBUG(this->get_logger(), "Received message on topic %s", "{topic}");
    
    // TODO: Implement message processing
    (void)msg;  // Avoid unused parameter warning
}}
'''
            
            if node.services:
                for srv in node.services:
                    service_name = srv['service']
                    srv_type = srv['srv_type'].replace('/', '::')
                    callback_name = f"{service_name.lstrip('/').replace('/', '_')}_callback"
                    source_content += f'''
void {class_name}::{callback_name}(
    const std::shared_ptr<{srv_type}::Request> request,
    std::shared_ptr<{srv_type}::Response> response)
{{
    // Process service request
    RCLCPP_DEBUG(this->get_logger(), "Received service call on %s", "{service_name}");
    
    // TODO: Implement service logic
    (void)request;  // Avoid unused parameter warning
    (void)response; // Avoid unused parameter warning
}}
'''
        
        # Create output directory if it doesn't exist
        source_path = self.output_dir / 'src' / f"{node.name}_node.cpp"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(source_path, 'w') as f:
            f.write(source_content)
    
    def _generate_node_header(self, node: NodeConfig) -> Path:
        """Generate a C++ header file for a ROS2 node.
        
        Args:
            node: The node configuration
            
        Returns:
            Path to the generated header file
        """
        # Convert node name to a valid C++ class name (e.g., 'my_node' -> 'MyNode')
        class_name = ''.join(word.capitalize() for word in node.name.split('_'))
        
        # Generate include guard
        include_guard = f"ROBODSL_{node.name.upper()}_NODE_HPP_"
        
        # Generate includes
        includes = [
            '#include "rclcpp/rclcpp.hpp"',
            '#include <string>',
            '#include <vector>',
            '#include <map>',
            '',  # For better formatting
        ]
        
        # Add message includes for publishers and subscribers
        msg_includes = set()
        for pub in node.publishers:
            msg_includes.add(f'#include "{pub["msg_type"].replace(".", "/")}.hpp"')
        for sub in node.subscribers:
            msg_includes.add(f'#include "{sub["msg_type"].replace(".", "/")}.hpp"')
        
        # Add service includes
        for srv in node.services:
            msg_includes.add(f'#include "{srv["srv_type"].replace(".", "/")}.hpp"')
        
        # Sort includes for consistent output
        includes.extend(sorted(msg_includes))
        
        # Generate the class declaration parts
        ros_publishers = self._generate_publisher_declarations(node.publishers)
        ros_subscribers = self._generate_subscriber_declarations(node.subscribers)
        ros_services = self._generate_service_declarations(node.services)
        parameters = self._generate_parameter_declarations(node.parameters)
        cuda_kernels = self._generate_cuda_kernel_declarations(node.name, self.config.cuda_kernels)
        
        # Generate class declaration using f-strings to avoid format() conflicts
        class_declaration = f"""class {class_name} : public rclcpp::Node {{
public:
    {class_name}();
    virtual ~{class_name}();

private:
    // ROS2 Publishers
{ros_publishers}
    
    // ROS2 Subscribers
{ros_subscribers}
    
    // ROS2 Services
{ros_services}
    
    // Parameters
{parameters}

    // CUDA Kernels
{cuda_kernels}
}};
"""
        
        # Write the header file
        header_content = f"""// Generated by RoboDSL - DO NOT EDIT

#ifndef {include_guard}
#define {include_guard}

{"\n".join(includes)}

namespace robodsl {{

{class_declaration}

}}  // namespace robodsl

#endif  // {include_guard}
"""
        
        # Create output directory if it doesn't exist
        header_path = self.output_dir / 'include' / 'robodsl' / f"{node.name}_node.hpp"
        header_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(header_path, 'w') as f:
            f.write(header_content)
    
    def _generate_publisher_declarations(self, publishers: List[Dict[str, str]]) -> str:
        """Generate C++ declarations for ROS2 publishers."""
        if not publishers:
            return '    // No publishers declared\n'
            
        result = []
        for pub in publishers:
            # Convert message type to C++ type (e.g., 'std_msgs/msg/String' -> 'std_msgs::msg::String')
            msg_type = pub['msg_type'].replace('/', '::')
            var_name = f"{pub['topic'].lstrip('/').replace('/', '_')}_pub_"
            result.append(f'    rclcpp::Publisher<{msg_type}>::SharedPtr {var_name};')
        
        return '\n'.join(result) + '\n'
    
    def _generate_subscriber_declarations(self, subscribers: List[Dict[str, str]]) -> str:
        """Generate C++ declarations for ROS2 subscribers."""
        if not subscribers:
            return '    // No subscribers declared\n'
            
        result = []
        for sub in subscribers:
            # Convert message type to C++ type
            msg_type = sub['msg_type'].replace('/', '::')
            var_name = f"{sub['topic'].lstrip('/').replace('/', '_')}_sub_"
            callback_name = f"{sub['topic'].lstrip('/').replace('/', '_')}_callback"
            result.append(f'    rclcpp::Subscription<{msg_type}>::SharedPtr {var_name};')
            result.append(f'    void {callback_name}(const {msg_type}::SharedPtr msg) const;')
        
        return '\n'.join(result) + '\n'
    
    def _generate_service_declarations(self, services: List[Dict[str, str]]) -> str:
        """Generate C++ declarations for ROS2 services."""
        if not services:
            return '    // No services declared\n'
            
        result = []
        for srv in services:
            # Convert service type to C++ type
            srv_type = srv['srv_type'].replace('/', '::')
            var_name = f"{srv['service'].lstrip('/').replace('/', '_')}_srv_"
            callback_name = f"{srv['service'].lstrip('/').replace('/', '_')}_callback"
            result.append(f'    rclcpp::Service<{srv_type}>::SharedPtr {var_name};')
            result.append(f'    void {callback_name}(')
            result.append(f'        const std::shared_ptr<{srv_type}::Request> request,')
            result.append(f'        std::shared_ptr<{srv_type}::Response> response);')
        
        return '\n'.join(result) + '\n'
    
    def _generate_parameter_declarations(self, parameters: Dict[str, str]) -> str:
        """Generate C++ declarations for node parameters."""
        if not parameters:
            return '    // No parameters declared\n'
            
        result = ['    // Parameters']
        for name, value in parameters.items():
            # Determine parameter type from value
            if value.lower() in ('true', 'false'):
                param_type = 'bool'
            elif value.replace('.', '', 1).isdigit():
                param_type = 'double' if '.' in value else 'int'
            else:
                param_type = 'std::string'
            
            result.append(f'    {param_type} {name}_;')
        
        return '\n'.join(result) + '\n'
    
    def _generate_cuda_kernel_declarations(self, node_name: str, kernels: List[CudaKernelConfig]) -> str:
        """Generate C++ declarations for CUDA kernels used by this node."""
        node_kernels = [k for k in kernels if k.name.startswith(node_name)]
        
        if not node_kernels:
            return '    // No CUDA kernels for this node\n'
            
        result = ['    // CUDA Kernels']
        for kernel in node_kernels:
            # Generate a C++ class for the CUDA kernel
            class_name = f"{kernel.name.capitalize()}Kernel"
            result.extend([
                f'    class {class_name} {{',
                '    public:',
                f'        {class_name}();',
                '        ~CudaKernel();',
                '        void configure(const std::map<std::string, std::string>& params);',
                '        void cleanup();',
                '        void process();',
                '    private:',
                '        // Kernel-specific data members',
                '    };',
                ''
            ])
        
        return '\n'.join(result)
    
    def _generate_cuda_kernel(self, kernel: CudaKernelConfig) -> List[Path]:
        """Generate CUDA kernel implementation files.
        
        Args:
            kernel: The CUDA kernel configuration
            
        Returns:
            List of Path objects for the generated files
        """
        # Create output directories if they don't exist
        cuda_include_dir = self.output_dir / 'include' / 'robodsl' / 'cuda'
        cuda_src_dir = self.output_dir / 'cuda'
        cuda_include_dir.mkdir(parents=True, exist_ok=True)
        cuda_src_dir.mkdir(parents=True, exist_ok=True)
        
        generated_files = []
        
        # Generate CUDA header file (.cuh)
        header_path = self._generate_cuda_header(kernel, cuda_include_dir)
        if header_path and isinstance(header_path, Path):
            generated_files.append(header_path)
        
        # Generate CUDA source file (.cu)
        source_path = self._generate_cuda_source(kernel, cuda_src_dir)
        if source_path and isinstance(source_path, Path):
            generated_files.append(source_path)
            
        return generated_files
    
    def _generate_cuda_header(self, kernel: CudaKernelConfig, output_dir: Path) -> Path:
        """Generate CUDA kernel header file with Thrust and memory management support.
        
        Args:
            kernel: The CUDA kernel configuration
            output_dir: Directory to write the header file to
            
        Returns:
            Path to the generated header file
        """
        class_name = f"{kernel.name.capitalize()}Kernel"
        header_guard = f"{kernel.name.upper()}_KERNEL_H_"
            
        # Generate includes
        includes = [
            '#include <cuda_runtime.h>',
            '#include <string>',
            '#include <map>',
            '#include <vector>',
            '#include <stdexcept>',
            '#include <memory>',
            '#include "cuda_utils.h"'  # For CUDA error checking
        ]
            
        # Add Thrust includes if needed
        if kernel.use_thrust:
            includes.extend([
                '#include <thrust/device_vector.h>',
                '#include <thrust/execution_policy.h>',
                '#include <thrust/transform.h>',
                '#include <thrust/functional.h>',
            ])
            
        # Add custom includes
        for include in kernel.includes:
            includes.append(f'#include <{include}>' if '/' in include else f'#include "{include}"')
            
        # Generate input/output parameter declarations
        input_params = []
        output_params = []
            
        for input_def in kernel.inputs:
            param_name = input_def['name']
            input_params.append(f"const {input_def['type']}& {param_name}")
            
        for output_def in kernel.outputs:
            param_name = output_def['name']
            output_params.append(f"{output_def['type']}& {output_def['name']}")
            
        # Generate function declarations
        kernel_launch_decl = f"void launch_kernel("
        kernel_launch_decl += ", ".join([f"const {p}" for p in input_params] + output_params)
        kernel_launch_decl += ");"
            
        # Generate header content
        header_content = f"""// Generated by RoboDSL - DO NOT EDIT
// CUDA Kernel: {kernel.name}

#ifndef {header_guard}
#define {header_guard}

{"\n".join(sorted(set(includes), key=str.lower))}

// Kernel configuration
struct {class_name}Config {{
    dim3 block_size = {{{kernel.block_size[0]}, {kernel.block_size[1]}, {kernel.block_size[2]}}};
    dim3 grid_size = {{{kernel.grid_size[0] if kernel.grid_size else 1}, {kernel.grid_size[1] if kernel.grid_size else 1}, {kernel.grid_size[2] if kernel.grid_size else 1}}};
    size_t shared_mem_bytes = {kernel.shared_mem_bytes};
    bool use_managed_memory = false;
}};

class {class_name} {{
public:
    {class_name}();
    ~{class_name}();
    
    // Configure kernel with parameters
    void configure(const std::map<std::string, std::string>& params);
    
    // Process data using CUDA kernel
    void process({', '.join(input_params + output_params)});
    
    // Get configuration
    const {class_name}Config& get_config() const {{ return config_; }}
    
private:
    // Kernel launch function
    {kernel_launch_decl}
    
    // Configuration
    {class_name}Config config_;
    
    // Device memory pointers
    std::vector<void*> device_ptrs_;
    
    // Thrust device vectors (if using Thrust)
    {"std::vector<thrust::device_vector<char>> thrust_vectors_;" if kernel.use_thrust else "// No Thrust vectors"}
    
    // Allocate device memory
    template<typename T>
    T* allocate_device_memory(size_t count);
    
    // Free all device memory
    void free_device_memory();
    
    // CUDA error checking
    void check_cuda_error(cudaError_t err, const std::string& msg) const;
}};

// Kernel function declaration
extern "C" __global__ void {kernel.name}_kernel("""
        
        # Add kernel parameters
        kernel_params = []
        for input_def in kernel.inputs:
            kernel_params.append(f"const {input_def['type']}* {input_def['name']}")
        for output_def in kernel.outputs:
            kernel_params.append(f"{output_def['type']}* {output_def['name']}")
        
        header_content += ", ".join(kernel_params) + ");\n"
        
        # Add end of header
        header_content += f"""

// Helper macros for kernel launches
#define LAUNCH_{kernel.name.upper()}_KERNEL(\
    grid, block, shared_mem, stream, ...) \
    {kernel.name}_kernel<<<grid, block, shared_mem, stream>>>(__VA_ARGS__);

#endif // {header_guard}
"""
        
        # Write header file
        header_path = output_dir / f"{kernel.name}_kernel.cuh"
        header_path.parent.mkdir(parents=True, exist_ok=True)
        with open(header_path, 'w') as f:
            f.write(header_content)
            
        return header_path
    
    def _generate_cuda_source(self, kernel: CudaKernelConfig, output_dir: Path) -> Path:
        """Generate CUDA kernel source file with memory management and kernel launch.
        
        Args:
            kernel: The CUDA kernel configuration
            output_dir: Directory to write the source file to
            
        Returns:
            Path to the generated source file
        """
        # Set up Jinja2 environment
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(Path(__file__).parent / 'templates'),
            autoescape=jinja2.select_autoescape(['cu', 'cuh']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        env.filters['dataclass_to_dict'] = self._dataclass_to_dict
        
        # Get block dimensions from kernel config (default: 256, 1, 1)
        block_x = kernel.block_size[0] if len(kernel.block_size) > 0 else 256
        block_y = kernel.block_size[1] if len(kernel.block_size) > 1 else 1
        block_z = kernel.block_size[2] if len(kernel.block_size) > 2 else 1
        
        # Generate input/output parameter declarations for the process method
        input_params = []
        output_params = []
        for i, input_def in enumerate(kernel.inputs):
            input_params.append(f"const std::vector<{input_def['type']}>& input_{i}")
        for i, output_def in enumerate(kernel.outputs):
            output_params.append(f"std::vector<{output_def['type']}>& output_{i}")
        
        # Generate kernel parameters for the device function
        kernel_params = []
        for i, input_def in enumerate(kernel.inputs):
            kernel_params.append(f"const {input_def['type']}* {input_def['name']}")
        for output_def in kernel.outputs:
            kernel_params.append(f"{output_def['type']}* {output_def['name']}")
        kernel_params_str = ", ".join(kernel_params)
        
        # Generate input/output size calculations
        size_calculations = []
        for i, (input_def, output_def) in enumerate(zip(kernel.inputs, kernel.outputs)):
            size_calculations.append(f"    size_t size_{i} = input_{i}.size() * sizeof({input_def['type']});")
        
        # Generate memory allocations
        memory_allocations = []
        for i, (input_def, output_def) in enumerate(zip(kernel.inputs, kernel.outputs)):
            memory_allocations.append(f"    {input_def['type']}* d_input_{i} = nullptr;")
            memory_allocations.append(f"    {output_def['type']}* d_output_{i} = nullptr;")
            memory_allocations.append(f"    CUDA_CHECK(cudaMalloc(&d_input_{i}, size_{i}));")
            memory_allocations.append(f"    CUDA_CHECK(cudaMalloc(&d_output_{i}, size_{i}));")
        
        mem_copies_h2d = []
        mem_copies_d2h = []
        
        # Input memory copies (host to device)
        for i, input_def in enumerate(kernel.inputs):
            mem_copies_h2d.append(
                f"// Copy input {i} to device\n"
                f"{input_def['type']}* d_input_{i} = allocate_device_memory<{input_def['type']}>(input_{i}.size());\n"
                f"CUDA_CHECK(cudaMemcpy(d_input_{i}, input_{i}.data(), input_{i}.size() * sizeof({input_def['type']}), cudaMemcpyHostToDevice));"
            )
        
        # Output memory allocations and copies (device to host)
        for i, output_def in enumerate(kernel.outputs):
            mem_copies_h2d.append(
                f"// Allocate device memory for output {i}\n"
                f"{output_def['type']}* d_output_{i} = allocate_device_memory<{output_def['type']}>(output_{i}.size());"
            )
            mem_copies_d2h.append(
                f"// Copy output {i} back to host\n"
                f"CUDA_CHECK(cudaMemcpy(output_{i}.data(), d_output_{i}, output_{i}.size() * sizeof({output_def['type']}), cudaMemcpyDeviceToHost));"
            )
        
        # Class implementation
        class_impl = """
// Allocate device memory
template<typename T>
T* {{ class_name }}::allocate_device_memory(size_t count) {
    T* ptr = nullptr;
    if (config_.use_managed_memory) {
        CUDA_CHECK(cudaMallocManaged(&ptr, count * sizeof(T)));
    } else {
        CUDA_CHECK(cudaMalloc(&ptr, count * sizeof(T)));
    }
    device_ptrs_.push_back(ptr);
    return ptr;
}

// Free all device memory
void {{ class_name }}::free_device_memory() {
    for (void* ptr : device_ptrs_) {
        if (ptr) cudaFree(ptr);
    }
    device_ptrs_.clear();
    
    // Clear Thrust vectors if used
#ifdef THRUST_DEVICE_SYSTEM
    if (config_.use_thrust) {
        thrust_vectors_.clear();
    }
#endif
}

// Check CUDA errors
void {{ class_name }}::check_cuda_error(cudaError_t err, const std::string& msg) const {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            msg + ": " + cudaGetErrorString(err));
    }
}"""
        
        # Generate the source content using the template
        template = env.get_template('cuda_source.cu.jinja2')
        
        # Prepare the context for the template
        context = {
            'kernel': self._dataclass_to_dict(kernel),
            'block_x': block_x,
            'block_y': block_y,
            'block_z': block_z,
            'class_name': f"{kernel.name.capitalize()}Kernel",
            'class_impl': class_impl,
            'mem_copies_h2d': '\n        '.join(mem_copies_h2d),
            'mem_copies_d2h': '\n        '.join(mem_copies_d2h),
            'input_params': [f'const std::vector<{input_def["type"]}>& input_{i}' for i, input_def in enumerate(kernel.inputs)],
            'output_params': [f'std::vector<{output_def["type"]}>& output_{i}' for i, output_def in enumerate(kernel.outputs)]
        }
        
        # Render the template
        source_content = template.render(**context)
        
        # Write the generated source to file
        source_path = output_dir / f"{kernel.name}_kernel.cu"
        source_path.parent.mkdir(parents=True, exist_ok=True)
        with open(source_path, 'w') as f:
            f.write(source_content)
            
        return source_path
    

    
    def _dataclass_to_dict(self, obj):
        """Convert a dataclass object to a dictionary for Jinja2 templating."""
        if hasattr(obj, '__dataclass_fields__'):
            # Handle dataclasses
            result = {}
            for field in dataclasses.fields(obj):
                value = getattr(obj, field.name)
                result[field.name] = self._dataclass_to_dict(value)
            return result
        elif isinstance(obj, (list, tuple)):
            # Handle lists and tuples
            return [self._dataclass_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            # Handle dictionaries
            return {k: self._dataclass_to_dict(v) for k, v in obj.items()}
        else:
            # Handle primitive types
            return obj

    def _generate_cmakelists(self) -> Optional[Path]:
        """Generate CMakeLists.txt for the project.
        
        Returns:
            Path to the generated CMakeLists.txt file, or None if not generated
        """
        # Get all node names
        node_names = [node.name for node in self.config.nodes]
        
        # Get all CUDA kernel names
        cuda_kernel_names = [kernel.name for kernel in self.config.cuda_kernels]
        
        cmake_content = f"""# Generated by RoboDSL - DO NOT EDIT
cmake_minimum_required(VERSION 3.8)
project({self.config.project_name} LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find required packages
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)

# Set include directories
include_directories(
  include
  ${{CMAKE_CURRENT_SOURCE_DIR}}/include
)

# Add CUDA specific flags
set(CMAKE_CUDA_FLAGS "${{CMAKE_CUDA_FLAGS}} -std=c++17 --expt-relaxed-constexpr")

# Add library for CUDA kernels
add_library(cuda_kernels
  src/cuda/cuda_kernels.cu
)
target_include_directories(cuda_kernels PUBLIC
  ${{CMAKE_CURRENT_SOURCE_DIR}}/include
)
target_link_libraries(cuda_kernels
  CUDA::cudart
  CUDA::cuda_driver
)

# Add node executables
"""

        # Add node targets
        for node_name in node_names:
            cmake_content += f"""
# {node_name} node
add_executable({node_name}_node
  src/{node_name}_node.cpp
)
target_include_directories({node_name}_node PUBLIC
  ${{CMAKE_CURRENT_SOURCE_DIR}}/include
)
target_link_libraries({node_name}_node
  rclcpp::rclcpp
  rclcpp_components::component
  cuda_kernels
)
"""

        # Add install directives
        cmake_content += """
# Install include files
install(
  DIRECTORY include/
  DESTINATION include/
)

# Install launch files
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install config files
install(
  DIRECTORY config/
  DESTINATION share/${PROJECT_NAME}/config
)

# Export dependencies
ament_export_dependencies(ament_cmake rclcpp rclcpp_components)

# Export includes
ament_export_include_directories(include)

# Export libraries
ament_export_libraries(cuda_kernels)

# Generate package configuration
ament_package()
"""
        
        # Write CMakeLists.txt
        cmake_path = self.output_dir / 'CMakeLists.txt'
        with open(cmake_path, 'w') as f:
            f.write(cmake_content)
            
        return cmake_path
