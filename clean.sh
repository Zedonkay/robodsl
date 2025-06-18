#!/bin/bash
set -e

echo "🧹 Cleaning up build artifacts..."

# Remove build directories
rm -rf deb-build/ debian-pkgs/ build/ dist/ *.egg-info/

# Remove Python cache files
find . -type d -name "__pycache__" -exec rm -rf {} +
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete
find . -name "*.pyd" -delete
find . -name "*.pyc" -delete

# Remove test directories
rm -rf test/__pycache__/

# Remove any .pyc files in the source directory
find src/ -name "*.pyc" -delete
find src/ -name "__pycache__" -exec rm -rf {} +

echo "✅ Cleanup complete!"
