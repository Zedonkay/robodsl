# RoboDSL Examples

This directory contains example projects demonstrating RoboDSL's capabilities, from basic ROS2 integration to advanced CUDA-accelerated robotics applications. Each example is designed to be self-contained and includes all necessary configuration files.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Example Categories](#example-categories)
   - [Basic Examples](#basic-examples)
     - [Simple Publisher/Subscriber](#1-simple-publishersubscriber)
     - [ROS2 Parameters](#2-ros2-parameters)
   - [Intermediate Examples](#intermediate-examples)
     - [Lifecycle Node](#1-lifecycle-node)
     - [QoS Configuration](#2-qos-configuration)
   - [Advanced Examples](#advanced-examples)
     - [CUDA Acceleration](#1-cuda-acceleration)
     - [Action Server with CUDA](#2-action-server-with-cuda)
     - [Namespaced Components](#3-namespaced-components)
3. [Building and Running](#building-and-running)
4. [Troubleshooting](#troubleshooting)
5. [Contributing](#contributing)

## Quick Start

1. **Prerequisites**
   - ROS2 Humble or newer
   - CUDA Toolkit (for GPU examples)
   - Python 3.8+

2. **Build and Run an Example**
   ```bash
   # Navigate to the example directory
   cd examples/basic_pubsub
   
   # Build the example
   colcon build --symlink-install
   
   # Source the setup file
   source install/setup.bash
   
   # Run the example
   ros2 run basic_pubsub basic_pubsub_node
   ```

3. **Explore More Examples**
   - Check out the [example categories](#example-categories) below
   - Look for `README.md` in each example directory for specific instructions

## Example Categories

## Basic Examples

### 1. Simple Publisher/Subscriber

#### Overview
Demonstrates basic ROS2 communication patterns with configurable QoS settings.

#### Key Features
- Topic-based pub/sub communication
- Configurable QoS profiles
- Namespace and remapping support
- Parameterized message publishing rate

#### Files
```
basic_pubsub/
├── pubsub.robodsl     # DSL definition
├── config/            # Parameter files
│   └── params.yaml
├── launch/            # Launch files
│   └── pubsub.launch.py
├── CMakeLists.txt     # Build configuration
└── package.xml        # Package manifest
```

#### Example Snippet
```python
# pubsub.robodsl
node basic_pubsub {
    # Configure node properties
    namespace = "demo"
    executable = true
    
    # Define a publisher with custom QoS
    publishers = [{
        name = "chatter"
        type = "std_msgs/msg/String"
        qos = {
            reliability = "reliable"
            depth = 10
            durability = "volatile"
        }
    }]
    
    # Define a subscriber
    subscribers = [{
        name = "chatter"
        type = "std_msgs/msg/String"
        callback = "chatter_callback"
        qos = "sensor_data"  # Using a predefined QoS profile
    }]
    
    # Node parameters
    parameters = {
        "publish_rate" = 1.0
    }
    
    # Callback implementation
    callbacks = """
    void chatter_callback(const std_msgs::msg::String::SharedPtr msg) {
        RCLCPP_INFO(this->get_logger(), "I heard: '%s'", msg->data.c_str());
    }
    """
}
```

### 2. ROS2 Parameters

#### Overview
Demonstrates parameter handling with type safety, validation, and dynamic updates.

#### Key Features
- Strongly-typed parameters
- Runtime parameter updates
- Validation and constraints
- Parameter file loading
- Dynamic parameter reconfigure

#### Example Snippet
```python
# parameters.robodsl
node parameter_demo {
    # Declare parameters with types and defaults
    parameters = {
        "int_param" = {
            type = "integer"
            default = 42
            description = "An integer parameter"
            constraints = {
                min = 0
                max = 100
                step = 1
            }
        },
        "str_param" = {
            type = "string"
            default = "default_value"
            description = "A string parameter"
        },
        "bool_param" = {
            type = "boolean"
            default = true
            description = "A boolean parameter"
        },
        "double_array" = {
            type = "double[]"
            default = [1.0, 2.0, 3.0]
            description = "Array of doubles"
        }
    }
    
    # Dynamic parameter update callback
    callbacks = """
    rcl_interfaces::msg::SetParametersResult 
    on_parameter_change(const std::vector<rclcpp::Parameter> &parameters) {
        auto result = rcl_interfaces::msg::SetParametersResult();
        result.successful = true;
        
        for (const auto &param : parameters) {
            if (param.get_name() == "int_param") {
                // Custom validation
                if (param.as_int() < 0) {
                    result.successful = false;
                    result.reason = "Value must be non-negative";
                }
            }
        }
        return result;
    }
    """
}
```

## Intermediate Examples

### 1. Lifecycle Node

#### Overview
Demonstrates a managed lifecycle node with state transitions and error recovery.

#### Key Features
- Full lifecycle state management
- State transition callbacks
- Error handling and recovery
- System integration patterns
- Component lifecycle management

#### Example Snippet
```python
# lifecycle.robodsl
node lifecycle_demo : rclcpp_lifecycle::LifecycleNode {
    # Node configuration
    namespace = "lifecycle"
    executable = true
    
    # Lifecycle publisher (note the type prefix)
    publishers = [{
        name = "status"
        type = "std_msgs/msg/String"
        lifecycle = true  # Special lifecycle publisher
    }]
    
    # Override lifecycle callbacks
    callbacks = """
    // Configure callback
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_configure(const rclcpp_lifecycle::State &) override {
        RCLCPP_INFO(get_logger(), "Configuring...");
        // Initialize resources
        status_publisher_->on_activate();
        return LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }
    
    // Activate callback
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_activate(const rclcpp_lifecycle::State &) override {
        RCLCPP_INFO(get_logger(), "Activating...");
        // Start timers, subscriptions, etc.
        timer_ = create_wall_timer(
            std::chrono::seconds(1),
            std::bind(&LifecycleDemo::timer_callback, this)
        );
        return LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }
    
    // Deactivate callback
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_deactivate(const rclcpp_lifecycle::State &) override {
        RCLCPP_INFO(get_logger(), "Deactivating...");
        // Clean up resources
        timer_.reset();
        return LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }
    
    // Cleanup callback
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_cleanup(const rclcpp_lifecycle::State &) override {
        RCLCPP_INFO(get_logger(), "Cleaning up...");
        // Release all resources
        return LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }
    
    // Shutdown callback
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_shutdown(const rclcpp_lifecycle::State &state) override {
        RCLCPP_INFO(get_logger(), "Shutting down from state %s", state.label().c_str());
        // Final cleanup
        return LifecycleNodeInterface::CallbackReturn::SUCCESS;
    }
    
    // Error handling
    rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn
    on_error(const rclcpp_lifecycle::State &) override {
        RCLCPP_ERROR(get_logger(), "Error occurred!");
        // Handle error and decide next state
        return LifecycleNodeInterface::CallbackReturn::FAILURE;
    }
    
    // Timer callback
    void timer_callback() {
        auto msg = std_msgs::msg::String();
        msg.data = "Lifecycle node active";
        status_publisher_->publish(msg);
    }
    
    // Members
    rclcpp::TimerBase::SharedPtr timer_;
    rclcpp_lifecycle::LifecyclePublisher<std_msgs::msg::String>::SharedPtr status_publisher_;
    """
}
```

### 2. QoS Configuration

#### Overview
Demonstrates advanced Quality of Service (QoS) configuration for ROS2 communication.

#### Key Features
- Custom QoS profiles
- Liveliness and deadline configuration
- History depth and reliability settings
- Best practices for different communication patterns

#### Example Snippet
```python
# qos_demo.robodsl
node qos_demo {
    # Define custom QoS profiles
    qos_profiles = {
        # Best effort with small history for high-frequency sensor data
        sensor_data = {
            reliability = "best_effort"
            durability = "volatile"
            depth = 10
            deadline = "100ms"
            liveliness = "automatic"
            liveliness_lease_duration = "1s"
        },
        # Reliable with large history for commands
        command = {
            reliability = "reliable"
            durability = "transient_local"
            depth = 100
            deadline = "1s"
        }
    }
    
    # Use the profiles in publishers/subscribers
    publishers = [{
        name = "sensor_data"
        type = "sensor_msgs/msg/LaserScan"
        qos = "sensor_data"  # Reference the profile
    }]
    
    subscribers = [{
        name = "command"
        type = "std_msgs/msg/String"
        qos = "command"
        callback = "command_callback"
    }]
    
    # Service with custom QoS
    services = [{
        name = "process_data"
        type = "example_interfaces/srv/ProcessData"
        qos = {
            reliability = "reliable"
            history = "keep_last"
            depth = 5
        }
        callback = "process_data"
    }]
    
    callbacks = """
    void command_callback(const std_msgs::msg::String::SharedPtr msg) {
        RCLCPP_INFO(get_logger(), "Received command: %s", msg->data.c_str());
    }
    
    void process_data(
        const std::shared_ptr<rmw_request_id_t> request_header,
        const std::shared_ptr<example_interfaces::srv::ProcessData::Request> request,
        std::shared_ptr<example_interfaces::srv::ProcessData::Response> response
    ) {
        (void)request_header;  // Unused
        // Process data here
        response->success = true;
    }
    """
}
```

## Advanced Examples

### 1. CUDA Acceleration

#### Overview
Demonstrates GPU-accelerated computation with CUDA in a ROS2 node, including memory management and kernel optimization.

#### Key Features
- CUDA kernel definition and compilation
- Zero-copy memory management
- Asynchronous execution with CUDA streams
- Thrust algorithm integration
- Profiling with Nsight Systems

#### Project Structure
```
cuda_acceleration/
├── vector_ops.robodsl     # Main DSL definition
├── include/               # Header files
│   └── vector_ops.hpp
├── src/                   # CUDA source files
│   ├── vector_ops.cu     # CUDA kernels
│   └── vector_ops.cpp    # Host code
├── launch/                # Launch files
│   └── cuda_demo.launch.py
├── config/                # Configuration files
│   └── params.yaml
├── CMakeLists.txt         # Build configuration
└── package.xml            # Package manifest
```

#### Example Snippet
```python
# vector_ops.robodsl
node cuda_vector_ops {
    # Enable CUDA support
    cuda = {
        architectures = ["75", "80"]  # Compute capabilities
        flags = ["-O3", "--use_fast_math"]
    }
    
    # Node configuration
    namespace = "cuda_demo"
    executable = true
    
    # Define parameters
    parameters = {
        "vector_size" = 1000000
        "block_size" = 256
    }
    
    # CUDA kernel definition
    cuda_kernels = """
    // Vector addition kernel
    __global__ void vector_add(
        const float* a,
        const float* b,
        float* result,
        int n
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            result[idx] = a[idx] + b[idx];
        }
    }
    
    // Wrapper function to call from C++
    void launch_vector_add(
        const float* a,
        const float* b,
        float* result,
        int n,
        int block_size,
        cudaStream_t stream = 0
    ) {
        int grid_size = (n + block_size - 1) / block_size;
        vector_add<<<grid_size, block_size, 0, stream>>>(a, b, result, n);
    }
    """
    
    # Main node class
    callbacks = """
    class CudaVectorOps : public rclcpp::Node {
    public:
        explicit CudaVectorOps(const rclcpp::NodeOptions& options)
        : Node("cuda_vector_ops", options) {
            // Get parameters
            vector_size_ = this->declare_parameter<int>("vector_size", 1000000);
            block_size_ = this->declare_parameter<int>("block_size", 256);
            
            // Initialize CUDA stream
            cudaStreamCreate(&stream_);
            
            // Allocate host and device memory
            allocate_memory();
            
            // Create a timer for the processing loop
            timer_ = this->create_wall_timer(
                std::chrono::milliseconds(100),
                std::bind(&CudaVectorOps::timer_callback, this)
            );
        }
        
        ~CudaVectorOps() {
            // Cleanup CUDA resources
            cudaFree(device_a_);
            cudaFree(device_b_);
            cudaFree(device_result_);
            cudaStreamDestroy(stream_);
        }
        
    private:
        void allocate_memory() {
            // Allocate pinned host memory for better transfer performance
            cudaMallocHost(&host_a_, vector_size_ * sizeof(float));
            cudaMallocHost(&host_b_, vector_size_ * sizeof(float));
            cudaMallocHost(&host_result_, vector_size_ * sizeof(float));
            
            // Allocate device memory
            cudaMalloc(&device_a_, vector_size_ * sizeof(float));
            cudaMalloc(&device_b_, vector_size_ * sizeof(float));
            cudaMalloc(&device_result_, vector_size_ * sizeof(float));
            
            // Initialize host data
            for (int i = 0; i < vector_size_; ++i) {
                host_a_[i] = static_cast<float>(i);
                host_b_[i] = static_cast<float>(i * 2);
            }
        }
        
        void timer_callback() {
            auto start = std::chrono::high_resolution_clock::now();
            
            // Copy data to device (asynchronous)
            cudaMemcpyAsync(device_a_, host_a_, vector_size_ * sizeof(float),
                          cudaMemcpyHostToDevice, stream_);
            cudaMemcpyAsync(device_b_, host_b_, vector_size_ * sizeof(float),
                          cudaMemcpyHostToDevice, stream_);
            
            // Launch kernel
            launch_vector_add(device_a_, device_b_, device_result_, 
                            vector_size_, block_size_, stream_);
            
            // Copy result back to host (asynchronous)
            cudaMemcpyAsync(host_result_, device_result_, vector_size_ * sizeof(float),
                          cudaMemcpyDeviceToHost, stream_);
            
            // Wait for all operations to complete
            cudaStreamSynchronize(stream_);
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                end - start).count();
                
            RCLCPP_INFO(get_logger(), "Processed %d elements in %ld µs", 
                      vector_size_, duration);
        }
        
        // Member variables
        int vector_size_;
        int block_size_;
        cudaStream_t stream_;
        float *host_a_, *host_b_, *host_result_;
        float *device_a_, *device_b_, *device_result_;
        rclcpp::TimerBase::SharedPtr timer_;
    };
    """
}

### 2. Action Server with CUDA

#### Overview
Demonstrates a ROS2 action server that performs GPU-accelerated image processing, including progress feedback and cancellation support.

#### Key Features
- Action definition with custom interfaces
- Asynchronous CUDA processing
- Progress feedback and cancellation
- Thread-safe CUDA operations
- QoS configuration for action servers

#### Project Structure
```
action_server/
├── action/                    # Action definitions
│   └── ProcessImage.action
├── include/                   # C++ headers
│   └── image_processor.hpp
├── src/                       # Source files
│   ├── cuda/                 # CUDA kernels
│   │   └── image_processing.cu
│   └── image_processor.cpp   # Action server implementation
├── launch/                    # Launch files
│   └── image_processor.launch.py
├── config/                   # Configuration files
│   └── params.yaml
├── CMakeLists.txt            # Build configuration
└── package.xml               # Package manifest
```

#### Example Snippet
```python
# image_processor.robodsl
node image_processor {
    # Enable CUDA support
    cuda = {
        architectures = ["75", "80"]
        flags = ["-O3", "--use_fast_math"]
    }
    
    # Node configuration
    namespace = "vision"
    executable = true
    
    # Define parameters
    parameters = {
        "max_processing_threads" = 4
        "gpu_memory_limit_mb" = 2048
    }
    
    # Action server definition
    actions = [{
        name = "process_image"
        type = "vision_msgs/action/ProcessImage"
        callback = "execute_process_image"
        qos = {
            reliability = "reliable"
            depth = 10
        }
    }]
    
    # CUDA kernels for image processing
    cuda_kernels = """
    // Image processing kernel (simplified)
    __global__ void process_image_kernel(
        const uchar4* input,
        uchar4* output,
        int width,
        int height,
        float threshold
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (x < width && y < height) {
            int idx = y * width + x;
            uchar4 pixel = input[idx];
            
            // Simple thresholding as an example
            uchar gray = (pixel.x + pixel.y + pixel.z) / 3;
            uchar result = (gray > threshold * 255) ? 255 : 0;
            
            output[idx] = make_uchar4(result, result, result, 255);
        }
    }
    
    // Wrapper function
    void launch_process_image(
        const uchar4* d_input,
        uchar4* d_output,
        int width,
        int height,
        float threshold,
        cudaStream_t stream = 0
    ) {
        dim3 block(16, 16);
        dim3 grid((width + block.x - 1) / block.x,
                 (height + block.y - 1) / block.y);
        
        process_image_kernel<<<grid, block, 0, stream>>>(
            d_input, d_output, width, height, threshold);
    }
    """
    
    # Main node implementation
    callbacks = """
    class ImageProcessor : public rclcpp::Node {
    public:
        using ProcessImage = vision_msgs::action::ProcessImage;
        using GoalHandle = rclcpp_action::ServerGoalHandle<ProcessImage>;
        
        explicit ImageProcessor(const rclcpp::NodeOptions& options)
        : Node("image_processor", options) {
            // Initialize CUDA
            cudaSetDevice(0);
            cudaStreamCreate(&stream_);
            
            // Create action server
            action_server_ = rclcpp_action::create_server<ProcessImage>(
                this->get_node_base_interface(),
                this->get_node_clock_interface(),
                this->get_node_logging_interface(),
                this->get_node_waitables_interface(),
                "process_image",
                std::bind(&ImageProcessor::handle_goal, this, _1, _2),
                std::bind(&ImageProcessor::handle_cancel, this, _1),
                std::bind(&ImageProcessor::handle_accepted, this, _1)
            );
            
            RCLCPP_INFO(get_logger(), "Image processor ready");
        }
        
        ~ImageProcessor() {
            cudaStreamDestroy(stream_);
        }
        
    private:
        rclcpp_action::GoalResponse handle_goal(
            const rclcpp_action::GoalUUID& uuid,
            std::shared_ptr<const ProcessImage::Goal> goal
        ) {
            (void)uuid;
            // Validate the goal
            if (goal->threshold < 0.0 || goal->threshold > 1.0) {
                RCLCPP_WARN(get_logger(), "Invalid threshold: %f", goal->threshold);
                return rclcpp_action::GoalResponse::REJECT;
            }
            return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        }
        
        rclcpp_action::CancelResponse handle_cancel(
            const std::shared_ptr<GoalHandle> goal_handle
        ) {
            RCLCPP_INFO(get_logger(), "Received request to cancel goal");
            (void)goal_handle;
            return rclcpp_action::CancelResponse::ACCEPT;
        }
        
        void handle_accepted(const std::shared_ptr<GoalHandle> goal_handle) {
            std::thread{std::bind(&ImageProcessor::execute, this, _1), goal_handle}.detach();
        }
        
        void execute(const std::shared_ptr<GoalHandle> goal_handle) {
            const auto goal = goal_handle->get_goal();
            auto result = std::make_shared<ProcessImage::Result>();
            auto feedback = std::make_shared<ProcessImage::Feedback>();
            
            try {
                // Convert ROS image to CUDA format (simplified)
                // In a real implementation, you'd use cv_bridge or similar
                int width = goal->input_image.width;
                int height = goal->input_image.height;
                size_t image_size = width * height * 4;  // RGBA8
                
                // Allocate device memory
                uchar4* d_input = nullptr;
                uchar4* d_output = nullptr;
                cudaMalloc(&d_input, image_size);
                cudaMalloc(&d_output, image_size);
                
                // Copy input to device (with proper format conversion in real code)
                cudaMemcpyAsync(d_input, goal->input_image.data.data(),
                              image_size, cudaMemcpyHostToDevice, stream_);
                
                // Process in chunks with feedback
                const int num_chunks = 10;
                for (int i = 0; i < num_chunks; ++i) {
                    if (goal_handle->is_canceling()) {
                        result->success = false;
                        result->message = "Processing cancelled";
                        goal_handle->canceled(result);
                        RCLCPP_INFO(get_logger(), "Processing cancelled");
                        return;
                    }
                    
                    // Process a chunk (simplified)
                    launch_process_image(
                        d_input + (i * width * height / num_chunks),
                        d_output + (i * width * height / num_chunks),
                        width,
                        height / num_chunks,
                        goal->threshold,
                        stream_
                    );
                    
                    // Update feedback
                    feedback->progress = (i + 1.0) / num_chunks;
                    goal_handle->publish_feedback(feedback);
                    
                    // Simulate processing time
                    std::this_thread::sleep_for(std::chrono::milliseconds(100));
                }
                
                // Copy result back to host
                result->output_image = goal->input_image;  // In real code, copy actual result
                result->success = true;
                result->message = "Processing completed";
                
                goal_handle->succeed(result);
                RCLCPP_INFO(get_logger(), "Processing completed");
                
                // Cleanup
                cudaFree(d_input);
                cudaFree(d_output);
                
            } catch (const std::exception& e) {
                result->success = false;
                result->message = std::string("Error: ") + e.what();
                goal_handle->abort(result);
                RCLCPP_ERROR(get_logger(), "Processing failed: %s", e.what());
            }
        }
        
        // Member variables
        rclcpp_action::Server<ProcessImage>::SharedPtr action_server_;
        cudaStream_t stream_;
    };
    """
}

### 3. Namespaced Components

#### Overview
Demonstrates component-based architecture with namespacing and remapping for complex systems.

#### Key Features
- Component lifecycle management
- Namespace isolation
- Parameter remapping
- Component composition
- Resource management

#### Example Snippet
```python
# component_demo.robodsl
# Sensor component
component sensor_component {
    namespace = "sensor"
    
    # Publishers
    publishers = [{
        name = "data"
        type = "sensor_msgs/msg/Imu"
        qos = "sensor_data"
    }]
    
    # Parameters
    parameters = {
        "publish_rate" = 100.0
        "noise_level" = 0.01
    }
    
    # Implementation
    callbacks = """
    class SensorComponent : public rclcpp::Node {
    public:
        explicit SensorComponent(const rclcpp::NodeOptions& options)
        : Node("sensor", options) {
            // Initialize sensor
            timer_ = create_wall_timer(
                std::chrono::duration<double>(1.0 / this->declare_parameter("publish_rate", 100.0)),
                std::bind(&SensorComponent::timer_callback, this)
            );
        }
        
    private:
        void timer_callback() {
            auto msg = sensor_msgs::msg::Imu();
            // Populate with sensor data
            publisher_->publish(msg);
        }
        
        rclcpp::TimerBase::SharedPtr timer_;
        rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr publisher_;
    };
    """
}

# Processing component
component processor_component {
    namespace = "processor"
    
    # Subscribers
    subscribers = [{
        name = "input"
        type = "sensor_msgs/msg/Imu"
        topic = "/sensor/data"  # Remap to sensor's topic
        callback = "process_data"
    }]
    
    # Publishers
    publishers = [{
        name = "result"
        type = "geometry_msgs/msg/Vector3"
    }]
    
    # Implementation
    callbacks = """
    class ProcessorComponent : public rclcpp::Node {
    public:
        explicit ProcessorComponent(const rclcpp::NodeOptions& options)
        : Node("processor", options) {
            // Initialization
        }
        
        void process_data(const sensor_msgs::msg::Imu::SharedPtr msg) {
            // Process data and publish result
            auto result = geometry_msgs::msg::Vector3();
            publisher_->publish(result);
        }
        
    private:
        rclcpp::Publisher<geometry_msgs::msg::Vector3>::SharedPtr publisher_;
    };
    """
}

# Main composition node
node component_demo {
    namespace = "demo_system"
    executable = true
    
    # Include components
    components = [
        "sensor_component",
        "processor_component"
    ]
    
    # Remap topics between components
    remappings = {
        "/sensor/data" = "/demo_system/sensor/data"
        "/processor/input" = "/demo_system/processor/input"
    }
    
    # Launch configuration
    launch = {
        "sensor_component": {
            "ros__parameters": {
                "publish_rate": 50.0,
                "noise_level": 0.02
            }
        },
        "processor_component": {
            "ros__parameters": {
                "processing.enabled": true
            }
        }
    }
}

#### Key Features
- Runtime composition
- Component lifecycle
- Inter-component communication

## Building and Running

### Prerequisites
- RoboDSL installed and in your PATH
- ROS2 Humble or newer
- CUDA Toolkit 11.0+ (for CUDA examples)
- CMake 3.15+

### Building an Example

1. Navigate to the example directory:
   ```bash
   cd examples/<example_name>
   ```

2. Generate the code:
   ```bash
   robodsl generate <example_name>.robodsl
   ```

3. Build the project:
   ```bash
   mkdir -p build && cd build
   cmake .. -DCMAKE_BUILD_TYPE=Release
   make -j$(nproc)
   ```

### Running an Example

1. Source your ROS2 environment:
   ```bash
   source /opt/ros/$ROS_DISTRO/setup.bash
   ```

2. Run the example:
   ```bash
   ./<example_name>
   ```

   Or for ROS2 nodes:
   ```bash
   ros2 run <package_name> <node_name>
   ```

## Example: CUDA Vector Operations

### Building and Running

```bash
cd examples/cuda_vector_ops
robodsl generate vector_ops.robodsl
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
./vector_ops
```

### Expected Output

```
[INFO] [vector_ops]: Initializing vector operations...
[INFO] [vector_ops]: Running vector addition...
[INFO] [vector_ops]: Result: [3.0, 5.0, 7.0, 9.0, 11.0]
[INFO] [vector_ops]: Running vector multiplication...
[INFO] [vector_ops]: Result: [2.0, 4.0, 6.0, 8.0, 10.0]
[INFO] [vector_ops]: Running vector reduction with Thrust...
[INFO] [vector_ops]: Sum: 15.0
```

## Conditional Compilation

Examples demonstrate how to use conditional compilation flags:

```python
# In your .robodsl file
build_options = {
    "ENABLE_ROS2": true,
    "ENABLE_CUDA": true,
    "CUDA_ARCH": "sm_75"  # Specify your GPU architecture
}

# Conditional code blocks
#ifdef ENABLE_CUDA
cuda_kernel my_kernel {
    # Kernel definition
}
#endif
```

## Contributing

We welcome contributions to expand our examples! Please follow these guidelines:

1. Create a new directory under `examples/` for your example
2. Include a detailed README.md with:
   - Example description
   - Prerequisites
   - Build and run instructions
   - Expected output
3. Keep the example focused and well-documented
4. Follow the existing code style

## License

All examples are part of the RoboDSL project and are licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.

## Troubleshooting

### Common Issues

1. **CUDA Not Found**
   - Ensure CUDA Toolkit is installed
   - Set `CUDA_TOOLKIT_ROOT_DIR` if not in default location

2. **ROS2 Packages Not Found**
   - Source your ROS2 environment
   - Install missing dependencies with `rosdep`

3. **Build Failures**
   - Check CMake output for missing dependencies
   - Ensure all submodules are initialized

For additional help, please open an issue on our [GitHub repository](https://github.com/yourusername/robodsl).
