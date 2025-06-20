# ROS2 Integration

RoboDSL provides seamless integration with ROS2, allowing you to define nodes, topics, services, and actions with a clean, declarative syntax. This document covers the various ROS2 features supported by RoboDSL.

## Table of Contents
- [Lifecycle Nodes](#lifecycle-nodes)
- [QoS Configuration](#qos-configuration)
- [Namespaces and Remapping](#namespaces-and-remapping)
- [Parameters](#parameters)
- [Actions](#actions)
- [Services](#services)
- [Timers](#timers)

## Lifecycle Nodes

RoboDSL supports ROS2's managed nodes with full lifecycle management. To create a lifecycle node:

```robodsl
node my_lifecycle_node {
    lifecycle: true  // Enable lifecycle management
    
    // Define transitions
    on_configure {
        // Initialization code
        return SUCCESS
    }
    
    on_activate {
        // Activation code
        return SUCCESS
    }
    
    on_deactivate {
        // Deactivation code
        return SUCCESS
    }
    
    on_cleanup {
        // Cleanup code
        return SUCCESS
    }
    
    on_shutdown {
        // Shutdown code
        return SUCCESS
    }
}
```

## QoS Configuration

RoboDSL allows fine-grained control over Quality of Service (QoS) settings for publishers and subscribers:

```robodsl
node qos_example {
    // Publisher with custom QoS
    publisher /status std_msgs/msg/String {
        qos: {
            reliability: reliable      // or best_effort
            durability: transient_local // or volatile
            history: keep_last        // or keep_all
            depth: 10                 // Only for keep_last
            deadline: 100ms           // Optional
            lifespan: 1000ms          // Optional
            liveliness: automatic     // or manual_by_topic, manual_by_node
            liveliness_lease_duration: 1000ms
        }
    }
    
    // Subscriber with matching QoS
    subscriber /input sensor_msgs/msg/Image {
        qos: @reliable_transient_local
    }
}

// Define reusable QoS profiles
qos_profile reliable_transient_local {
    reliability: reliable
    durability: transient_local
    history: keep_last
    depth: 10
}
```

## Namespaces and Remapping

### Namespaces

You can organize nodes and topics into namespaces:

```robodsl
node my_node {
    namespace: /robot1
    
    // This will be published to /robot1/status
    publisher status std_msgs/msg/String
    
    // Nested namespaces
    namespace: /sensors {
        // This will be published to /robot1/sensors/lidar
        publisher lidar sensor_msgs/msg/PointCloud2
    }
}
```

### Remapping

Remap topic and service names at runtime:

```robodsl
node remap_example {
    // Remap topic names
    remap: {
        from: /old_topic
        to: /new_topic
    }
    
    // Multiple remaps
    remap: {
        from: /camera/image_raw
        to: /sensors/camera/left/image_raw
    }
    
    // This will publish to /new_topic
    publisher /old_topic std_msgs/msg/String
}
```

## Parameters

Define and use parameters with type safety:

```robodsl
node param_example {
    // Declare parameters
    parameters {
        // Required parameter with default value
        string my_string {
            default: "hello"
            description: "A string parameter"
        }
        
        // Integer parameter with constraints
        int my_int {
            default: 42
            min: 0
            max: 100
            description: "An integer parameter"
        }
        
        // Double parameter
        double my_double {
            default: 3.14
            description: "A double parameter"
        }
        
        // Boolean parameter
        bool my_bool {
            default: true
            description: "A boolean parameter"
        }
        
        // Array parameter
        double[] gains {
            default: [1.0, 2.0, 3.0]
            description: "Array of gains"
        }
    }
    
    // Use parameters in callbacks
    timer 1.0 {
        // Access parameters using the 'params' object
        let message = std_msgs::msg::String()
        message.data = params.my_string + " " + str(params.my_int)
        pub.publish(message)
    }
}
```

## Actions

Define and use ROS2 actions:

```robodsl
// Define a custom action
action Fibonacci {
    // Goal
    int32 order
    
    // Result
    int32[] sequence
    
    // Feedback
    int32[] partial_sequence
}

node action_server_example {
    // Action server
    action_server /fibonacci Fibonacci {
        // Called when a new goal is received
        on_goal(goal_handle) {
            // Accept the goal
            goal_handle.succeed()
            
            // Execute the action
            let sequence = [0, 1]
            
            // Publish feedback
            while (sequence.length() <= goal_handle.goal.order) {
                // Publish feedback
                let feedback = Fibonacci::Feedback()
                feedback.partial_sequence = sequence
                goal_handle.publish_feedback(feedback)
                
                // Calculate next number
                sequence.push_back(sequence[-1] + sequence[-2])
                
                // Sleep
                rclcpp::sleep_for(100ms)
            }
            
            // Set the result
            let result = Fibonacci::Result()
            result.sequence = sequence
            return result
        }
        
        // Handle goal cancellation
        on_cancel(goal_handle) {
            // Cleanup code
            return CancelResponse::ACCEPT
        }
    }
}

node action_client_example {
    // Action client
    action_client /fibonacci Fibonacci {
        // Called when the goal completes
        on_result(result) {
            // Handle the result
            print("Fibonacci sequence: ", result.sequence)
        }
        
        // Called when feedback is received
        on_feedback(feedback) {
            // Handle feedback
            print("Partial sequence: ", feedback.partial_sequence)
        }
    }
    
    // Send a goal
    timer 5.0 {
        let goal = Fibonacci::Goal()
        goal.order = 10
        action_client.send_goal(goal)
    }
}
```

## Services

Define and use ROS2 services:

```robodsl
// Define a custom service
srv AddTwoInts {
    int64 a
    int64 b
    ---
    int64 sum
}

node service_example {
    // Service server
    service /add_two_ints AddTwoInts (request, response) {
        response.sum = request.a + request.b
        return true
    }
    
    // Service client
    service_client /add_two_ints AddTwoInts {
        // Called when the service call completes
        on_response(future) {
            let response = future.get()
            print("Sum: ", response.sum)
        }
        
        // Called if the service call fails
        on_failure() {
            print("Service call failed")
        }
    }
    
    // Call the service
    timer 1.0 {
        let request = AddTwoInts::Request()
        request.a = 10
        request.b = 20
        service_client.call_async(request)
    }
}
```

## Timers

Create periodic callbacks with timers:

```robodsl
node timer_example {
    // Create a publisher
    publisher /timer_example std_msgs/msg/String
    
    // Create a timer that fires every second
    timer 1.0 {
        let msg = std_msgs::msg::String()
        msg.data = "Hello at " + str(now().seconds())
        pub.publish(msg)
    }
    
    // Timer with wall time
    wall_timer 100ms {
        // This runs every 100ms of wall time
    }
}
```

## Best Practices

1. **Use namespaces** to organize your nodes and topics
2. **Define QoS profiles** at the top level for reusability
3. **Use lifecycle nodes** for stateful nodes that need initialization and cleanup
4. **Validate parameters** with constraints
5. **Handle errors** in service and action callbacks
6. **Use wall timers** for real-time applications
7. **Document** your nodes, parameters, and interfaces
