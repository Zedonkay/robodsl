#!/bin/bash
set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <package.deb>"
    exit 1
fi

PACKAGE="$1"
REPO_DIR="apt-repo"
DISTRIBUTIONS="focal jammy bookworm"

# Verify package exists
if [ ! -f "$PACKAGE" ]; then
    echo "Error: Package $PACKAGE not found"
    exit 1
fi

# Create dists directory structure
for dist in $DISTRIBUTIONS; do
    mkdir -p "$REPO_DIR/dists/$dist/main/binary-{amd64,arm64}"
    
    # Copy package to repository
    cp "$PACKAGE" "$REPO_DIR"
    
    # Create Packages file
    cd "$REPO_DIR"
    dpkg-scanpackages -m . > Packages
    gzip -k -f Packages
    
    # Create Release file
    cat > Release <<EOF
Origin: RoboDSL
Label: RoboDSL
Codename: $dist
Architectures: amd64 arm64
Components: main
Description: RoboDSL APT Repository
Date: $(date -Ru)
MD5Sum:
 $(md5sum Packages | cut -d' ' -f1) $(stat -c %s Packages) Packages
 $(md5sum Packages.gz | cut -d' ' -f1) $(stat -c %s Packages.gz) Packages.gz
SHA256:
 $(sha256sum Packages | cut -d' ' -f1) $(stat -c %s Packages) Packages
 $(sha256sum Packages.gz | cut -d' ' -f1) $(stat -c %s Packages.gz) Packages.gz
EOF
    
    # Sign Release file
    gpg --default-key "packages@robodsl.org" -abs -o Release.gpg Release
    gpg --default-key "packages@robodsl.org" --clearsign -o InRelease Release
    
    cd ..
done

echo "Package added to repository. Next steps:"
echo "1. Host the apt-repo directory on a web server"
echo "2. Users can add the repository with:"
echo "   curl -sS https://your-server/robodsl.gpg.key | sudo gpg --dearmor -o /usr/share/keyrings/robodsl-archive-keyring.gpg"
echo "   echo \"deb [arch=amd64,arm64 signed-by=/usr/share/keyrings/robodsl-archive-keyring.gpg] https://your-server/apt-repo/ ./\" | sudo tee /etc/apt/sources.list.d/robodsl.list"
echo "   sudo apt update"
echo "   sudo apt install robodsl"
