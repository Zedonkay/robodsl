#!/bin/bash
set -e

# Configuration
REPO_NAME="robodsl"
GPG_KEY_NAME="RoboDSL Package Signing Key"
GPG_KEY_EMAIL="packages@robodsl.org"
DISTRIBUTIONS="focal jammy bookworm"  # Ubuntu 20.04, 22.04, Debian 12
COMPONENTS="main"
ARCHITECTURES="amd64 arm64"

# Create repository directory structure
REPO_DIR="apt-repo"
mkdir -p "$REPO_DIR/conf"

# Create distributions file
cat > "$REPO_DIR/conf/distributions" <<EOF
Origin: $REPO_NAME
Label: $REPO_NAME
Codename: {DISTRIBUTION}
Architectures: $ARCHITECTURES
Components: $COMPONENTS
Description: APT repository for RoboDSL
SignWith: yes
EOF

# Create options file
cat > "$REPO_DIR/conf/options" <<EOF
basedir .
ask-passphrase
EOF

# Generate GPG key if it doesn't exist
if ! gpg --list-secret-keys "$GPG_KEY_EMAIL" >/dev/null 2>&1; then
    echo "Generating GPG key..."
    gpg --batch --gen-key <<EOF
Key-Type: RSA
Key-Length: 4096
Subkey-Type: RSA
Subkey-Length: 4096
Name-Real: $GPG_KEY_NAME
Name-Email: $GPG_KEY_EMAIL
Expire-Date: 0
%no-protection
%commit
EOF
fi

# Export GPG public key
echo "Exporting GPG public key..."
gpg --armor --export "$GPG_KEY_EMAIL" > "$REPO_DIR/$REPO_NAME.gpg.key"

echo "APT repository structure created in $REPO_DIR"
