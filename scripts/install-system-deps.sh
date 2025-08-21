#!/bin/bash
# TokenVM System Dependencies Installation Script
# This script installs system-level dependencies that require sudo

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}TokenVM System Dependencies Installer${NC}"
echo "======================================"
echo ""

# Detect OS
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
    # Detect distribution
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        DISTRO=$ID
        VERSION=$VERSION_ID
    fi
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
    DISTRO="macos"
else
    echo -e "${RED}Unsupported OS: $OSTYPE${NC}"
    exit 1
fi

echo "Detected: $OS ($DISTRO $VERSION)"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install packages based on distro
install_packages() {
    case "$DISTRO" in
        ubuntu|debian)
            echo -e "${YELLOW}Installing packages with apt-get...${NC}"
            sudo apt-get update
            sudo apt-get install -y \
                build-essential \
                g++ \
                make \
                cmake \
                git \
                curl \
                wget \
                python3 \
                python3-pip \
                python3-venv \
                python3-dev \
                libnuma-dev \
                libaio-dev \
                pkg-config
            ;;

        fedora|rhel|centos|rocky|almalinux)
            echo -e "${YELLOW}Installing packages with dnf/yum...${NC}"
            if command_exists dnf; then
                PKG_MGR="dnf"
            else
                PKG_MGR="yum"
            fi
            sudo $PKG_MGR install -y \
                gcc \
                gcc-c++ \
                make \
                cmake \
                git \
                curl \
                wget \
                python3 \
                python3-pip \
                python3-devel \
                numactl-devel \
                libaio-devel \
                pkgconfig
            ;;

        arch|manjaro)
            echo -e "${YELLOW}Installing packages with pacman...${NC}"
            sudo pacman -Syu --noconfirm \
                base-devel \
                gcc \
                make \
                cmake \
                git \
                curl \
                wget \
                python \
                python-pip \
                numactl \
                libaio \
                pkg-config
            ;;

        opensuse*)
            echo -e "${YELLOW}Installing packages with zypper...${NC}"
            sudo zypper install -y \
                gcc \
                gcc-c++ \
                make \
                cmake \
                git \
                curl \
                wget \
                python3 \
                python3-pip \
                python3-devel \
                libnuma-devel \
                libaio-devel \
                pkg-config
            ;;

        macos)
            echo -e "${YELLOW}Installing packages with Homebrew...${NC}"
            if ! command_exists brew; then
                echo -e "${YELLOW}Installing Homebrew...${NC}"
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew update
            brew install \
                gcc \
                make \
                cmake \
                git \
                curl \
                wget \
                python@3.11 \
                pkg-config
            ;;

        *)
            echo -e "${RED}Unsupported distribution: $DISTRO${NC}"
            echo "Please install the following packages manually:"
            echo "  - C++ compiler (g++ or clang++)"
            echo "  - make"
            echo "  - cmake"
            echo "  - git"
            echo "  - curl or wget"
            echo "  - Python 3.8+"
            echo "  - Python development headers"
            echo "  - NUMA library (optional)"
            echo "  - libaio (optional)"
            exit 1
            ;;
    esac
}

# Check for CUDA
check_cuda() {
    echo ""
    echo -e "${YELLOW}Checking for CUDA...${NC}"

    if command_exists nvcc; then
        CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')
        echo -e "${GREEN}✓ CUDA found: version $CUDA_VERSION${NC}"
    else
        echo -e "${YELLOW}⚠ CUDA not found${NC}"
        echo ""
        echo "To install CUDA (optional, for GPU support):"
        echo "1. Visit: https://developer.nvidia.com/cuda-downloads"
        echo "2. Select your OS and follow the installation instructions"
        echo "3. Add CUDA to your PATH:"
        echo "   export PATH=/usr/local/cuda/bin:\$PATH"
        echo "   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
        echo ""
        echo "TokenVM will work without CUDA using the CPU stub implementation."
    fi
}

# Check for Docker (optional)
check_docker() {
    echo ""
    echo -e "${YELLOW}Checking for Docker (optional)...${NC}"

    if command_exists docker; then
        echo -e "${GREEN}✓ Docker found${NC}"
    else
        echo -e "${YELLOW}⚠ Docker not found${NC}"
        echo "Docker is optional but recommended for containerized deployments."
        echo "To install: https://docs.docker.com/get-docker/"
    fi
}

# Main installation flow
main() {
    echo -e "${YELLOW}This script will install system-level dependencies for TokenVM.${NC}"
    echo -e "${YELLOW}You may be prompted for your sudo password.${NC}"
    echo ""
    read -p "Continue? (y/n) " -n 1 -r
    echo ""

    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        exit 0
    fi

    echo ""
    install_packages

    echo ""
    echo -e "${GREEN}✓ System packages installed successfully!${NC}"

    check_cuda
    check_docker

    echo ""
    echo -e "${GREEN}Installation complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Return to the TokenVM directory"
    echo "2. Run 'make all' to build TokenVM"
    echo ""
    echo "The Makefile will automatically:"
    echo "  - Install Go (if needed)"
    echo "  - Create a Python virtual environment"
    echo "  - Install Python dependencies"
    echo "  - Build the TokenVM library and tools"
}

# Run main function
main
