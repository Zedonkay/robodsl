#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Setting up RoboDSL APT repository...${NC}"

# Create temporary directory
TEMP_DIR=$(mktemp -d)
trap 'rm -rf "$TEMP_DIR"' EXIT

# Install required packages
echo -e "${YELLOW}Installing required packages...${NC}"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    gnupg \
    lsb-release

# Add repository key
echo -e "${YELLOW}Adding repository key...${NC}"
curl -sSfL https://$GITHUB_USERNAME.github.io/robodsl/robodsl.gpg.key | \
    sudo gpg --dearmor -o /usr/share/keyrings/robodsl-archive-keyring.gpg

# Add repository
echo -e "${YELLOW}Adding repository...${NC}"
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/robodsl-archive-keyring.gpg] https://$GITHUB_USERNAME.github.io/robodsl/ ./" | \
    sudo tee /etc/apt/sources.list.d/robodsl.list > /dev/null

# Install package
echo -e "${YELLOW}Installing RoboDSL...${NC}"
sudo apt-get update
sudo apt-get install -y robodsl

echo -e "${GREEN}Installation complete! You can now run 'robodsl --version' to verify.${NC}"
