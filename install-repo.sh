#!/bin/bash
set -e

# Configuration
REPO_URL="https://your-server/apt-repo"  # Change this to your actual server URL

# Install required packages
echo "Installing required packages..."
sudo apt-get update
sudo apt-get install -y gnupg ca-certificates

# Download and install the GPG key
echo "Adding RoboDSL repository key..."
curl -fsSL "$REPO_URL/../robodsl.gpg.key" | sudo gpg --dearmor -o /usr/share/keyrings/robodsl-archive-keyring.gpg

# Add the repository
echo "Adding RoboDSL repository..."
echo "deb [arch=amd64,arm64 signed-by=/usr/share/keyrings/robodsl-archive-keyring.gpg] $REPO_URL/ ./" | sudo tee /etc/apt/sources.list.d/robodsl.list

# Update package lists
echo "Updating package lists..."
sudo apt-get update

echo "\nRepository setup complete! You can now install RoboDSL with:"
echo "  sudo apt install robodsl"
