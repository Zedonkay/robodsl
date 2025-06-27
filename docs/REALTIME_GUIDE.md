# Real-time Constraints Guide

## What are Real-time Constraints?

Real-time constraints ensure that operations complete within specified time limits. In robotics and ML systems, this is crucial for:
- Sensor data processing within deadlines
- Control loop timing requirements
- Safety-critical operations
- Predictable system behavior

## Real-time Platforms

### Linux with PREEMPT_RT
- **What**: Real-time Linux kernel patch
- **Use case**: General real-time applications
- **Priority range**: 1-99 (higher = more priority)
- **Deadline**: Nanosecond precision

### Xenomai
- **What**: Real-time framework for Linux
- **Use case**: Hard real-time requirements
- **Priority range**: 0-255
- **Deadline**: Microsecond precision

### QNX
- **What**: Real-time operating system
- **Use case**: Safety-critical systems
- **Priority range**: 0-255
- **Deadline**: Microsecond precision

### Windows with RTX
- **What**: Real-time extension for Windows
- **Use case**: Windows-based real-time systems
- **Priority range**: 0-255
- **Deadline**: Microsecond precision

## Implementation Approach

For RoboDSL, we'll start with **Linux PREEMPT_RT** as it's:
1. Most commonly used in robotics
2. Well-supported in ROS 2
3. Easier to set up and test
4. Sufficient for most real-time requirements

## Real-time Grammar Examples

```robodsl
// Basic real-time configuration
node sensor_processor {
    realtime {
        priority: 80
        deadline: 10ms
        cpu_affinity: [0, 1]  // Use cores 0 and 1
        memory_policy: "locked"  // Lock memory to prevent paging
    }
    
    subscriber "/sensor_data" "sensor_msgs/Image"
    publisher "/processed_data" "sensor_msgs/Image"
}

// Pipeline with real-time constraints
pipeline realtime_vision {
    stage sensor_input {
        realtime {
            priority: 90
            deadline: 5ms
        }
        topic: "/camera/image_raw"
    }
    
    stage processing {
        realtime {
            priority: 80
            deadline: 15ms
        }
        method: process_image
    }
    
    stage output {
        realtime {
            priority: 70
            deadline: 5ms
        }
        topic: "/processed_output"
    }
}
```

## Generated Code Structure

The real-time generator will create:

1. **Thread Configuration**
   ```cpp
   // Set thread priority
   sched_param param;
   param.sched_priority = 80;
   pthread_setschedparam(pthread_self(), SCHED_FIFO, &param);
   
   // Set CPU affinity
   cpu_set_t cpuset;
   CPU_ZERO(&cpuset);
   CPU_SET(0, &cpuset);
   CPU_SET(1, &cpuset);
   pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
   ```

2. **Deadline Monitoring**
   ```cpp
   class DeadlineMonitor {
   public:
       DeadlineMonitor(std::chrono::microseconds deadline) 
           : deadline_(deadline), start_time_(std::chrono::steady_clock::now()) {}
       
       ~DeadlineMonitor() {
           auto elapsed = std::chrono::steady_clock::now() - start_time_;
           if (elapsed > deadline_) {
               RCLCPP_WARN(node_->get_logger(), "Deadline violation: %ld us", 
                          std::chrono::duration_cast<std::chrono::microseconds>(elapsed).count());
           }
       }
   private:
       std::chrono::microseconds deadline_;
       std::chrono::steady_clock::time_point start_time_;
   };
   ```

3. **Memory Management**
   ```cpp
   // Lock memory to prevent paging
   mlockall(MCL_CURRENT | MCL_FUTURE);
   
   // Pre-allocate memory pools
   std::vector<uint8_t> memory_pool(1024 * 1024);  // 1MB pool
   ```

## Testing Real-time Performance

### Tools
- **cyclictest**: Measure latency
- **ftrace**: Kernel tracing
- **perf**: Performance analysis
- **rt-tests**: Real-time test suite

### Metrics
- **Maximum latency**: Worst-case response time
- **Jitter**: Variation in response time
- **CPU utilization**: Real-time load
- **Memory usage**: Locked memory consumption

## Future Considerations

1. **Multi-platform support**: Extend to Xenomai, QNX, Windows RTX
2. **Advanced scheduling**: Earliest Deadline First (EDF), Rate Monotonic
3. **Resource reservation**: CPU bandwidth, memory bandwidth
4. **Fault tolerance**: Error recovery, redundancy
5. **Safety certification**: DO-178C, ISO 26262 compliance

## Getting Started

1. **Install PREEMPT_RT kernel**:
   ```bash
   sudo apt install linux-image-rt
   ```

2. **Configure real-time group**:
   ```bash
   sudo usermod -a -G realtime $USER
   ```

3. **Set real-time limits**:
   ```bash
   echo -1 > /proc/sys/kernel/sched_rt_runtime_us
   ```

4. **Test with cyclictest**:
   ```bash
   cyclictest -p 80 -t1 -n -a0 -l 100000
   ```

This guide provides the foundation for implementing real-time constraints in RoboDSL. The implementation will start with Linux PREEMPT_RT and can be extended to other platforms as needed. 