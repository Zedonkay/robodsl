#!/bin/bash
set -e

# Create a temporary directory for testing
TEST_DIR=$(mktemp -d)
echo "Testing package installation in: $TEST_DIR"

# Create a virtual environment
python3 -m venv "$TEST_DIR/venv"
source "$TEST_DIR/venv/bin/activate"

# Install the package
pip install -e .

# Test the CLI
if command -v robodsl &> /dev/null; then
    echo "✅ robodsl command is available"
    robodsl --version
else
    echo "❌ robodsl command is not in PATH"
    exit 1
fi

# Clean up
deactivate
rm -rf "$TEST_DIR"
echo "✅ Test completed successfully"
