# Create a test directory
mkdir debian-test && cd debian-test

# Run Ubuntu container
docker run -it --rm -v $(pwd):/workspace ubuntu:latest bash

# Inside the container:
cd /workspace
apt update
apt install -y git

# Clone your repo
git clone <your-repo-url>
cd <your-repo-name>

# Follow the same steps as your GitHub Action
apt install -y software-properties-common
add-apt-repository ppa:deadsnakes/ppa -y
apt update

apt install -y \
  devscripts \
  debhelper \
  dh-python \
  build-essential \
  cmake \
  python3.10 \
  python3.10-dev \
  python3.10-venv \
  python3-all \
  python3-setuptools \
  python3-sphinx \
  python3-sphinx-rtd-theme \
  python3-sphinxcontrib.apidoc

# Build the package
dpkg-buildpackage -us -uc
