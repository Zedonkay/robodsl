# RoboDSL Language Support

[![Version](https://vsmarketplacebadge.apphb.com/version/Zedonkay.robodsl.png)](https://marketplace.visualstudio.com/items?itemName=Zedonkay.robodsl)
[![Installs](https://vsmarketplacebadge.apphb.com/installs/Zedonkay.robodsl.png)](https://marketplace.visualstudio.com/items?itemName=Zedonkay.robodsl)
[![Rating](https://vsmarketplacebadge.apphb.com/rating/Zedonkay.robodsl.png)](https://marketplace.visualstudio.com/items?itemName=Zedonkay.robodsl)

Syntax highlighting and language support for RoboDSL files. RoboDSL is a domain-specific language for defining ROS 2 nodes with CUDA kernel support.

## Features

- Syntax highlighting for RoboDSL files (`.robodsl`)
- Support for ROS 2 node configuration
- CUDA kernel syntax highlighting
- Snippets for common patterns
- Bracket matching and auto-closing
- Comment toggling

## Installation

1. Open VS Code
2. Press `Ctrl+Shift+X` to open Extensions
3. Search for "RoboDSL"
4. Click Install

## Extension Settings

This extension contributes the following settings:

* `robodsl.enable`: Enable/disable the RoboDSL extension
* `robodsl.formatOnSave`: Enable/disable formatting on save

## Usage

Create a new file with a `.robodsl` extension to get started. The following features are supported:

### ROS 2 Node Configuration

```robodsl
node my_node {
    # QoS settings
    publisher /chatter std_msgs/msg/String qos=reliability=reliable durability=volatile
    
    # Timer with configuration
    timer my_timer: 1.0 {
        oneshot = false
        autostart = true
    }
    
    # Parameters
    parameter int rate = 10  # Hz
    parameter string frame_id = "base_link"
    
    # Lifecycle configuration
    lifecycle {
        autostart = true
        cleanup_on_shutdown = true
    }
}
```

### CUDA Kernel Support

```robodsl
cuda_kernels {
    kernel vector_add {
        block_size = (256, 1, 1)
        shared_memory = 0
        
        input float* a
        input float* b
        output float* c
        input int n
        
        code {
            __global__ void vector_add(const float* a, const float* b, float* c, int n) {
                int i = blockIdx.x * blockDim.x + threadIdx.x;
                if (i < n) {
                    c[i] = a[i] + b[i];
                }
            }
        }
    }
}
```

## Requirements

- Visual Studio Code 1.75.0 or higher

## Extension Commands

- `RoboDSL: Format Document`: Format the current RoboDSL document
- `RoboDSL: Show Syntax Tree`: Show the syntax tree for the current document

## Known Issues

- CUDA code blocks don't support all CUDA syntax highlighting features

## Release Notes

### 0.1.0

Initial release of RoboDSL language support.

## License

This extension is licensed under the [MIT License](LICENSE).