# Use official Python 3.9 slim image as base
FROM python:3.9-slim

# Set non-interactive frontend for apt
ENV DEBIAN_FRONTEND=noninteractive

# Set up Debian package mirror and configuration
RUN echo 'Acquire::Check-Valid-Until "false";' > /etc/apt/apt.conf.d/99no-check-valid && \
    echo 'APT::Get::Assume-Yes "true";' > /etc/apt/apt.conf.d/99force-confdef && \
    echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/99retry-limit && \
    echo 'Acquire::http::Timeout "120";' > /etc/apt/apt.conf.d/99timeout && \
    echo 'APT::Install-Recommends "false";' > /etc/apt/apt.conf.d/99no-recommends && \
    echo 'APT::Install-Suggests "false";' > /etc/apt/apt.conf.d/99no-suggests

# Update package lists and install build dependencies
RUN apt-get update -o Acquire::Check-Valid-Until=false && \
    apt-get install -y --no-install-recommends \
    build-essential \
    debhelper \
    dh-python \
    python3-all \
    python3-setuptools \
    python3-sphinx \
    python3-sphinx-rtd-theme \
    python3-sphinxcontrib.apidoc \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set up build environment
WORKDIR /build

# Copy only necessary files for build
COPY debian/ /build/debian/
COPY setup.py /build/
COPY src/ /build/src/
COPY README.md /build/

# Create output directory
RUN mkdir -p /build/debian-pkgs

# Build the package with retry logic
CMD ["/bin/bash", "-c", "\
    set -ex && \
    if [ ! -f debian/changelog ]; then \
        echo 'Creating initial changelog...' && \
        dch --create --package robodsl --newversion 0.1.0-1 'Initial release'; \
    fi && \
    for i in {1..3}; do \
        echo \"Attempt $i/3: Building package...\" && \
        if dpkg-buildpackage -us -uc; then \
            echo 'Package built successfully!' && \
            mv ../*.deb /build/debian-pkgs/ && \
            echo 'Generated packages:' && \
            ls -la /build/debian-pkgs/*.deb && \
            exit 0; \
        else \
            echo \"Attempt $i failed, retrying in 5 seconds...\" >&2; \
            sleep 5; \
        fi; \
    done"]
