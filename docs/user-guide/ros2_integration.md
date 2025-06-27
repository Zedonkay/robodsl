# ROS2 Integration

RoboDSL provides first-class support for ROS2 features. Here's a quick reference:

## Lifecycle Node

```robodsl
node my_node {
    enable_lifecycle = true
    on_configure = "configure_cb"
    on_activate = "activate_cb"
    on_deactivate = "deactivate_cb"
    on_cleanup = "cleanup_cb"
    on_shutdown = "shutdown_cb"
    on_error = "error_cb"
}
```

## QoS Configuration

```robodsl
// Publisher with QoS
publishers = [{
    name = "status"
    type = "std_msgs/msg/String"
    qos = {
        reliability = "reliable"  // or "best_effort"
        durability = "transient_local"  // or "volatile"
        depth = 10
    }
}]

// Reusable QoS profile
qos_profiles = [{
    name = "sensor_qos"
    reliability = "best_effort"
    durability = "volatile"
    depth = 1
}]
```

## Namespaces & Remapping

```robodsl
// Node with namespace
node my_node {
    namespace = "robot1"  // Topics will be /robot1/...
}

// Topic remapping
remappings = [
    {
        from = "/old_topic"
        to = "/new_topic"
    },
    {
        from = "/camera/image_raw"
        to = "/sensors/camera/left/image_raw"
    }
]
```

## Parameters

```robodsl
// Node with parameters
node param_node {
    parameters = [{
        name = "my_param"
        type = "string"
        value = "default"
        description = "Example parameter"
    }]
}

// Access in code:
// params.my_param
```

## Actions & Services

```robodsl
// Action definition
action Fibonacci {
    int32 order
    ---
    int32[] sequence
    ---
    int32[] partial_sequence
}

// Action server
actions = [{
    name = "fibonacci"
    type = "example_actions/action/Fibonacci"
    execute_callback = "fibonacci_cb"
}]

// Service definition
services = [{
    name = "add_two_ints"
    type = "example_interfaces/srv/AddTwoInts"
    callback = "add_cb"
}]
```
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
