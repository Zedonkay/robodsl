#!/bin/bash
set -ex

# Update package lists
apt-get update -o Acquire::Check-Valid-Until=false

# Install minimal build dependencies
apt-get install -y --no-install-recommends \
    build-essential \
    debhelper \
    dh-python \
    python3-all \
    python3-setuptools \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Ensure debian/changelog exists
if [ ! -f debian/changelog ]; then
    echo "Creating initial changelog..."
    dch --create --package robodsl --newversion 0.1.0-1 'Initial release'
fi

# Build the package
dpkg-buildpackage -us -uc

# Create output directory
mkdir -p /output

# Copy the built packages
cp ../*.deb /output/ 2>/dev/null || true
cp ../*.changes /output/ 2>/dev/null || true
cp ../*.buildinfo /output/ 2>/dev/null || true

# List the generated files
echo "Build complete. Generated files in /output:"
ls -la /output/
