#!/bin/bash
set -e

# Create output directory
mkdir -p debian-pkgs

# Create a temporary directory for the build
BUILD_DIR=$(mktemp -d)
trap 'rm -rf "$BUILD_DIR"' EXIT

echo "Setting up build environment in $BUILD_DIR..."

# Copy necessary files
cp -r debian setup.py src README.md "$BUILD_DIR/"

# Ensure debian/changelog exists
if [ ! -f "$BUILD_DIR/debian/changelog" ]; then
    echo "Creating initial changelog..."
    cd "$BUILD_DIR"
    dch --create --package robodsl --newversion 0.1.0-1 'Initial release'
    cd - >/dev/null
fi

echo "Building Debian package..."

docker run --rm \
    -v "$BUILD_DIR:/build" \
    -w /build \
    debian:bookworm-slim bash -c '
        set -ex && \
        apt-get update && \
        apt-get install -y build-essential debhelper dh-python python3-all python3-setuptools && \
        dpkg-buildpackage -us -uc && \
        mkdir -p /build/../debian-pkgs && \
        mv ../*.deb /build/../debian-pkgs/'

# Check if packages were created
if [ -d debian-pkgs ] && [ "$(ls -A debian-pkgs/ 2>/dev/null)" ]; then
    echo -e "\n✅ Success! Debian packages have been built in the debian-pkgs/ directory:"
    find debian-pkgs -name '*.deb' -exec ls -lh {} \;
    
    # Extract package info for verification
    echo -e "\nPackage information:"
    for pkg in debian-pkgs/*.deb; do
        echo -e "\n=== $(basename "$pkg") ==="
        dpkg-deb -I "$pkg" | grep -E 'Package|Version|Architecture|Depends' || true
    done
else
    echo -e "\n❌ Error: No Debian packages were generated. Check the build output for errors." >&2
    exit 1
fi
