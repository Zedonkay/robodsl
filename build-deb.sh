#!/bin/bash
set -e

# Create output directory
mkdir -p debian-pkgs

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "Docker is not running. Please start Docker Desktop and try again." >&2
    exit 1
fi

# Try to pull the image first with a public mirror
echo "Pulling base image from a public mirror..."
if ! docker pull docker.mirror.kku.at/library/python:3.9-slim; then
    echo "Warning: Failed to pull from mirror, trying default registry..."
    if ! docker pull python:3.9-slim; then
        echo "Failed to pull Python image. Please check your Docker setup and network connection." >&2
        exit 1
    fi
fi

echo -e "\nBuilding Docker image for Debian package creation..."
if ! docker build \
    --build-arg "http_proxy=${http_proxy}" \
    --build-arg "https_proxy=${https_proxy}" \
    --build-arg "no_proxy=${no_proxy}" \
    -t robodsl-deb \
    -f Dockerfile.debian .; then
    echo "Failed to build Docker image" >&2
    exit 1
fi

echo -e "\nBuilding Debian package (this may take a few minutes)..."
if ! docker run --rm \
    -e "http_proxy=${http_proxy}" \
    -e "https_proxy=${https_proxy}" \
    -e "no_proxy=${no_proxy}" \
    -v "$(pwd)/debian-pkgs:/build/debian-pkgs" \
    -v "$(pwd)/debian:/build/debian" \
    -v "$(pwd)/setup.py:/build/setup.py" \
    -v "$(pwd)/src:/build/src" \
    -v "$(pwd)/README.md:/build/README.md" \
    robodsl-deb; then
    echo "Failed to build Debian package" >&2
    exit 1
fi

# Check if packages were created
if [ -d debian-pkgs ] && [ "$(ls -A debian-pkgs/ 2>/dev/null)" ]; then
    echo -e "\n✅ Success! Debian packages have been built in the debian-pkgs/ directory:"
    find debian-pkgs -name '*.deb' -exec ls -lh {} \;
    
    # Extract package info for verification
    echo -e "\nPackage information:"
    for pkg in debian-pkgs/*.deb; do
        echo -e "\n=== $(basename "$pkg") ==="
        dpkg-deb -I "$pkg" | grep -E 'Package|Version|Architecture|Depends'
    done
else
    echo -e "\n❌ Error: No Debian packages were generated. Check the build output for errors." >&2
    echo "Contents of build directory:"
    docker run --rm -v "$(pwd):/mnt" python:3.9-slim ls -la /mnt/debian-pkgs/ 2>/dev/null || echo "Could not list package directory"
    exit 1
fi
