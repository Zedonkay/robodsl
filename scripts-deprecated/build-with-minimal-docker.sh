#!/bin/bash
set -e

# Create output directory
mkdir -p debian-pkgs

# Create a temporary directory for the build context
BUILD_CTX=$(mktemp -d)
trap 'rm -rf "$BUILD_CTX"' EXIT

# Download minimal Debian rootfs
if [ ! -f "$BUILD_CTX/debian-bullseye-minimal.tar.xz" ]; then
    echo "Downloading minimal Debian rootfs..."
    curl -L -o "$BUILD_CTX/debian-bullseye-minimal.tar.xz" \
        https://github.com/debuerreotype/docker-debian-artifacts/raw/fe7f3c30d1bd0a7e461b195bc023149c330f0d5a/bullseye/slim/rootfs.tar.xz
fi

# Copy necessary files to build context
cp Dockerfile.minimal build-in-docker.sh "$BUILD_CTX/"
cp -r debian setup.py src README.md "$BUILD_CTX/"

# Build the minimal Docker image
echo "Building minimal Docker image..."
cd "$BUILD_CTX"
docker build --no-cache -t robodsl-builder -f Dockerfile.minimal .

# Run the build in the container
echo -e "\nBuilding Debian package..."
docker run --rm \
    -v "$PWD:/build" \
    -v "$PWD/../debian-pkgs:/output" \
    robodsl-builder

# Check if packages were created
cd ..
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
