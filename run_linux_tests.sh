#!/bin/bash

# Comprehensive Linux Dependency Test Runner
# This script runs all comprehensive tests for Linux dependencies

set -e  # Exit on any error

echo "🚀 Starting Comprehensive Linux Dependency Tests"
echo "=================================================="

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check Python version
python_version=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "🐍 Python version: $python_version"

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Warning: Virtual environment not detected"
    echo "   Consider activating your virtual environment first"
fi

# Install test dependencies if needed
echo "📦 Installing test dependencies..."
pip install -r requirements-dev.txt

# Create test output directory
mkdir -p test_output

# Run the comprehensive test runner
echo "🧪 Running comprehensive tests..."
python tests/run_comprehensive_linux_tests.py \
    --verbose \
    --report \
    --timeout 600 \
    "$@"

# Check exit code
if [ $? -eq 0 ]; then
    echo "✅ All tests completed successfully!"
else
    echo "❌ Some tests failed. Check the output above for details."
    exit 1
fi

echo "📊 Test report generated: test_report.json"
echo "🎉 Test run completed!" 