#!/bin/bash

# RoboDSL Installation Script
# This script installs RoboDSL with all its features and dependencies
# 
# Note: RoboDSL is currently in development and not yet published to PyPI.
# This script installs from source code.

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command_exists apt-get; then
            echo "ubuntu"
        elif command_exists yum; then
            echo "centos"
        elif command_exists dnf; then
            echo "fedora"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        echo "windows"
    else
        echo "unknown"
    fi
}

# Function to check Python version
check_python() {
    if command_exists python3; then
        PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_success "Python $PYTHON_VERSION found"
            return 0
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            return 1
        fi
    elif command_exists python; then
        PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            print_success "Python $PYTHON_VERSION found"
            return 0
        else
            print_error "Python 3.8+ required, found $PYTHON_VERSION"
            return 1
        fi
    else
        print_error "Python not found"
        return 1
    fi
}

# Function to check CUDA availability
check_cuda() {
    if command_exists nvidia-smi; then
        print_success "NVIDIA GPU detected"
        if command_exists nvcc; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
            print_success "CUDA $CUDA_VERSION found"
            return 0
        else
            print_warning "CUDA toolkit not found"
            return 1
        fi
    else
        print_warning "No NVIDIA GPU detected"
        return 1
    fi
}

# Function to install system dependencies
install_system_deps() {
    local os=$(detect_os)
    
    case $os in
        "ubuntu"|"debian")
            print_status "Installing system dependencies for Ubuntu/Debian..."
            sudo apt-get update
            sudo apt-get install -y python3-pip python3-venv build-essential cmake git
            ;;
        "centos"|"rhel")
            print_status "Installing system dependencies for CentOS/RHEL..."
            sudo yum install -y python3-pip python3-devel gcc gcc-c++ cmake git
            ;;
        "fedora")
            print_status "Installing system dependencies for Fedora..."
            sudo dnf install -y python3-pip python3-devel gcc gcc-c++ cmake git
            ;;
        "macos")
            print_status "Installing system dependencies for macOS..."
            if ! command_exists brew; then
                print_status "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew install python@3.10 cmake
            ;;
        *)
            print_warning "Unknown OS, please install dependencies manually"
            ;;
    esac
}

# Function to create virtual environment
create_venv() {
    print_status "Creating virtual environment..."
    python3 -m venv robodsl_env
    source robodsl_env/bin/activate
    print_success "Virtual environment created and activated"
}

# Function to upgrade pip
upgrade_pip() {
    print_status "Upgrading pip..."
    pip install --upgrade pip
}

# Function to install RoboDSL
install_robodsl() {
    local install_type=$1
    local os=$(detect_os)
    
    case $install_type in
        "basic")
            print_status "Installing RoboDSL (basic)..."
            pip install -e .
            ;;
        "cuda")
            print_status "Installing RoboDSL with CUDA support..."
            pip install -e ".[cuda]"
            ;;
        "full")
            print_status "Installing RoboDSL with all features..."
            pip install -e ".[cuda,tensorrt]"
            ;;
        "all")
            if [ "$os" = "macos" ]; then
                print_warning "The 'all' option includes TensorRT which is not available on macOS."
                print_warning "Installing with development tools instead..."
                pip install -e ".[dev,docs]"
            else
                print_status "Installing RoboDSL with EVERYTHING (all features + dev tools)..."
                pip install -e ".[all]"
            fi
            ;;
        "dev")
            print_status "Installing RoboDSL in development mode..."
            pip install -e ".[dev]"
            ;;
        *)
            print_error "Unknown installation type: $install_type"
            exit 1
            ;;
    esac
}

# Function to verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Test basic import
    if python -c "import robodsl; print('RoboDSL version:', robodsl.__version__)"; then
        print_success "RoboDSL imported successfully"
    else
        print_error "Failed to import RoboDSL"
        return 1
    fi
    
    # Test CLI
    if command_exists robodsl; then
        print_success "RoboDSL CLI available"
        robodsl --version
    else
        print_error "RoboDSL CLI not found"
        return 1
    fi
    
    # Test CUDA if available
    if check_cuda; then
        if python -c "import cupy; print('CUDA available:', cupy.cuda.is_available())"; then
            print_success "CUDA support verified"
        else
            print_warning "CUDA support not available"
        fi
    fi
    
    # Test ONNX Runtime
    if python -c "import onnxruntime as ort; print('ONNX Runtime version:', ort.__version__)"; then
        print_success "ONNX Runtime available"
    else
        print_warning "ONNX Runtime not available"
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE     Installation type (basic, cuda, full, all, dev) [default: basic]"
    echo "  -s, --system-deps   Install system dependencies"
    echo "  -v, --venv          Create virtual environment"
    echo "  -h, --help          Show this help message"
    echo ""
    echo "Installation types:"
    echo "  basic               Basic installation without CUDA/TensorRT"
    echo "  cuda                Installation with CUDA support"
    echo "  full                Installation with all features (CUDA + TensorRT)"
    echo "  all                 Installation with EVERYTHING (all features + dev tools)"
    echo "                      Note: On macOS, this installs dev tools only (TensorRT not available)"
    echo "  dev                 Development installation with all tools"
    echo ""
    echo "Examples:"
    echo "  $0                  # Basic installation"
    echo "  $0 -t cuda          # Install with CUDA support"
    echo "  $0 -t all -s -v     # Install everything with system deps and venv"
}

# Main installation function
main() {
    local install_type="basic"
    local install_system_deps=false
    local create_venv_flag=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                install_type="$2"
                shift 2
                ;;
            -s|--system-deps)
                install_system_deps=true
                shift
                ;;
            -v|--venv)
                create_venv_flag=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    print_status "Starting RoboDSL installation..."
    print_status "Installation type: $install_type"
    
    # Check Python
    if ! check_python; then
        print_error "Python 3.8+ is required"
        exit 1
    fi
    
    # Install system dependencies if requested
    if [ "$install_system_deps" = true ]; then
        install_system_deps
    fi
    
    # Create virtual environment if requested
    if [ "$create_venv_flag" = true ]; then
        create_venv
    fi
    
    # Upgrade pip
    upgrade_pip
    
    # Install RoboDSL
    install_robodsl "$install_type"
    
    # Verify installation
    if verify_installation; then
        print_success "RoboDSL installation completed successfully!"
        echo ""
        echo "Next steps:"
        echo "1. Try running: robodsl --help"
        echo "2. Check the documentation: https://robodsl.readthedocs.io"
        echo "3. Join the community: https://github.com/Zedonkay/robodsl"
    else
        print_error "Installation verification failed"
        exit 1
    fi
}

# Run main function with all arguments
main "$@" 