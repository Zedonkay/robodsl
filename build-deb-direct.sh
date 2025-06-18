#!/bin/bash
set -e

# Package information
PKG_NAME="robodsl"
VERSION="0.1.0"
ARCH="all"
MAINTAINER="Ishayu Shikhare <ishikhar@andrew.cmu.edu>"
DESCRIPTION="DSL for GPU-accelerated robotics applications with ROS2 and CUDA"

# Create build directory
BUILD_DIR="deb-build"
PKG_DIR="${BUILD_DIR}/${PKG_NAME}-${VERSION}"

# Clean previous build
rm -rf "${BUILD_DIR}"
mkdir -p "${PKG_DIR}/DEBIAN"

# Create control file
cat > "${PKG_DIR}/DEBIAN/control" << EOF
Package: ${PKG_NAME}
Version: ${VERSION}
Section: devel
Priority: optional
Architecture: ${ARCH}
Maintainer: ${MAINTAINER}
Description: ${DESCRIPTION}
Depends: python3, python3-click, python3-jinja2
EOF

# Install Python package
PKG_LIB_DIR="${PKG_DIR}/usr/local/lib/python3.9/dist-packages/${PKG_NAME}"
mkdir -p "${PKG_LIB_DIR}"
cp -r src/robodsl/* "${PKG_LIB_DIR}/"

# Install binary
BIN_DIR="${PKG_DIR}/usr/local/bin"
mkdir -p "${BIN_DIR}"
cat > "${BIN_DIR}/robodsl" << 'EOF'
#!/usr/bin/env python3
from robodsl.cli import main

if __name__ == "__main__":
    main()
EOF
chmod +x "${BIN_DIR}/robodsl"

# Build the package using Homebrew's dpkg
/opt/homebrew/opt/dpkg/bin/dpkg-deb --build --root-owner-group "${PKG_DIR}"

# Move the package to the current directory
mv "${BUILD_DIR}/${PKG_NAME}-${VERSION}.deb" "${PKG_NAME}_${VERSION}_${ARCH}.deb"

# Clean up
rm -rf "${BUILD_DIR}"

echo "✅ Successfully built package: ${PKG_NAME}_${VERSION}_${ARCH}.deb"
