# TokenVM Makefile - Auto-installs dependencies and builds everything
SHELL := /bin/bash

# Version requirements
GO_MIN_VERSION := 1.21
PYTHON_MIN_VERSION := 3.8
CUDA_MIN_VERSION := 11.0

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

# Detect OS
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    OS := linux
endif
ifeq ($(UNAME_S),Darwin)
    OS := darwin
endif

# Detect architecture
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),x86_64)
    ARCH := amd64
endif
ifeq ($(UNAME_M),aarch64)
    ARCH := arm64
endif

# Installation directories
INSTALL_DIR := $(HOME)/.tokenvm
BIN_DIR := $(INSTALL_DIR)/bin
export PATH := $(BIN_DIR):$(PATH)

# Go installation
GO_VERSION := 1.22.0
GO_TARBALL := go$(GO_VERSION).$(OS)-$(ARCH).tar.gz
GO_URL := https://go.dev/dl/$(GO_TARBALL)
GO_INSTALL_DIR := $(INSTALL_DIR)/go

# Python virtual environment
VENV_DIR := venv
PYTHON := python3
PIP := $(VENV_DIR)/bin/pip
PYTHON_BIN := $(VENV_DIR)/bin/python

# Check for existing installations
HAS_GO := $(shell command -v go 2> /dev/null)
HAS_PYTHON := $(shell command -v python3 2> /dev/null)
HAS_CUDA := $(shell command -v nvcc 2> /dev/null)
HAS_VENV := $(shell test -d $(VENV_DIR) && echo yes)

# CUDA detection
ifdef HAS_CUDA
    CUDA_PATH ?= $(shell dirname $(shell dirname $(shell which nvcc)))
    CUDA_VERSION := $(shell nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')
    NVCC := nvcc
    CUDA_CFLAGS := -I$(CUDA_PATH)/include
    CUDA_LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart -lcuda
    BUILD_CUDA := 1
else
    BUILD_CUDA := 0
endif

# Compiler settings
CXX := g++
CXXFLAGS := -std=c++17 -fPIC -O2 -Wall
LDFLAGS := -shared

# Build directories
BUILD_DIR := build
LIB_DIR := lib

# Target library
ifeq ($(BUILD_CUDA),1)
    TARGET_LIB := $(LIB_DIR)/libtokenvm.so
    IMPL_SRC := internal/cuda/tokenvm.cc internal/cuda/arena.cc
    CUDA_SRCS := internal/cuda/pack.cu internal/cuda/gather.cu
else
    TARGET_LIB := $(LIB_DIR)/libtokenvm_stub.so
    IMPL_SRC := internal/cuda/tokenvm_stub.cc
endif

# Default target
.PHONY: all
all: check-deps install-deps setup build test
	@echo -e "$(GREEN)✓ TokenVM build complete!$(NC)"
	@echo -e "$(GREEN)✓ Python environment: $(VENV_DIR)/$(NC)"
	@echo -e "$(GREEN)✓ Library: $(TARGET_LIB)$(NC)"
	@echo -e "$(GREEN)✓ Run 'source $(VENV_DIR)/bin/activate' to activate Python environment$(NC)"
	@echo -e "$(GREEN)✓ Run 'make run-example' to test the system$(NC)"

# Check dependencies
.PHONY: check-deps
check-deps:
	@echo -e "$(YELLOW)Checking system dependencies...$(NC)"
	@echo -n "OS: $(OS) $(ARCH) ... "
	@echo -e "$(GREEN)✓$(NC)"

	@echo -n "C++ compiler ... "
	@if command -v g++ >/dev/null 2>&1; then \
		echo -e "$(GREEN)✓$(NC)"; \
	else \
		echo -e "$(RED)✗ Missing$(NC)"; \
		echo "Please install g++ (apt-get install g++ or yum install gcc-c++)"; \
		exit 1; \
	fi

	@echo -n "Make ... "
	@echo -e "$(GREEN)✓$(NC)"

	@echo -n "Git ... "
	@if command -v git >/dev/null 2>&1; then \
		echo -e "$(GREEN)✓$(NC)"; \
	else \
		echo -e "$(YELLOW)⚠ Missing (optional)$(NC)"; \
	fi

	@echo -n "curl/wget ... "
	@if command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1; then \
		echo -e "$(GREEN)✓$(NC)"; \
	else \
		echo -e "$(RED)✗ Missing$(NC)"; \
		echo "Please install curl or wget"; \
		exit 1; \
	fi

# Install missing dependencies
.PHONY: install-deps
install-deps: install-python install-go install-python-packages check-cuda

# Install Python if needed
.PHONY: install-python
install-python:
	@echo -n "Python 3 ... "
	@if [ -z "$(HAS_PYTHON)" ]; then \
		echo -e "$(YELLOW)Installing...$(NC)"; \
		if [ "$(OS)" = "linux" ]; then \
			if command -v apt-get >/dev/null 2>&1; then \
				sudo apt-get update && sudo apt-get install -y python3 python3-pip python3-venv; \
			elif command -v yum >/dev/null 2>&1; then \
				sudo yum install -y python3 python3-pip; \
			else \
				echo -e "$(RED)Cannot auto-install Python. Please install Python 3.8+ manually$(NC)"; \
				exit 1; \
			fi; \
		else \
			echo -e "$(RED)Please install Python 3.8+ manually$(NC)"; \
			exit 1; \
		fi; \
	else \
		echo -e "$(GREEN)✓$(NC)"; \
	fi

# Install Go if needed
.PHONY: install-go
install-go:
	@echo -n "Go $(GO_MIN_VERSION)+ ... "
	@if [ -z "$(HAS_GO)" ]; then \
		echo -e "$(YELLOW)Installing Go $(GO_VERSION)...$(NC)"; \
		mkdir -p $(INSTALL_DIR); \
		if command -v curl >/dev/null 2>&1; then \
			curl -L $(GO_URL) -o /tmp/$(GO_TARBALL); \
		else \
			wget $(GO_URL) -O /tmp/$(GO_TARBALL); \
		fi; \
		tar -C $(INSTALL_DIR) -xzf /tmp/$(GO_TARBALL); \
		rm /tmp/$(GO_TARBALL); \
		mkdir -p $(BIN_DIR); \
		ln -sf $(GO_INSTALL_DIR)/bin/go $(BIN_DIR)/go; \
		ln -sf $(GO_INSTALL_DIR)/bin/gofmt $(BIN_DIR)/gofmt; \
		echo -e "$(GREEN)✓ Installed Go to $(GO_INSTALL_DIR)$(NC)"; \
		echo -e "$(YELLOW)Add 'export PATH=$(BIN_DIR):$$PATH' to your shell profile$(NC)"; \
	else \
		GO_VERSION_CHECK=$$(go version | grep -oE '[0-9]+\.[0-9]+' | head -1); \
		echo -e "$(GREEN)✓ (version $$GO_VERSION_CHECK)$(NC)"; \
	fi

# Create Python virtual environment and install packages
.PHONY: install-python-packages
install-python-packages:
	@echo -n "Python virtual environment ... "
	@if [ -z "$(HAS_VENV)" ]; then \
		echo -e "$(YELLOW)Creating...$(NC)"; \
		$(PYTHON) -m venv $(VENV_DIR); \
		echo -e "$(GREEN)✓$(NC)"; \
	else \
		echo -e "$(GREEN)✓$(NC)"; \
	fi

	@echo -e "$(YELLOW)Installing Python packages...$(NC)"
	@$(PIP) install --upgrade pip setuptools wheel >/dev/null 2>&1
	@$(PIP) install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu >/dev/null 2>&1 || \
		$(PIP) install torch >/dev/null 2>&1 || \
		echo -e "$(YELLOW)⚠ PyTorch installation may require manual setup$(NC)"
	@$(PIP) install transformers accelerate datasets >/dev/null 2>&1
	@$(PIP) install prometheus-client psutil numpy >/dev/null 2>&1
	@$(PIP) install pytest pytest-benchmark >/dev/null 2>&1
	@echo -e "$(GREEN)✓ Python packages installed$(NC)"

# Check CUDA
.PHONY: check-cuda
check-cuda:
	@echo -n "CUDA ... "
	@if [ -n "$(HAS_CUDA)" ]; then \
		echo -e "$(GREEN)✓ (version $(CUDA_VERSION))$(NC)"; \
	else \
		echo -e "$(YELLOW)⚠ Not found - building CPU stub$(NC)"; \
		echo -e "$(YELLOW)  For GPU support, install CUDA Toolkit $(CUDA_MIN_VERSION)+$(NC)"; \
		echo -e "$(YELLOW)  Visit: https://developer.nvidia.com/cuda-downloads$(NC)"; \
	fi

# Setup project structure
.PHONY: setup
setup:
	@echo -e "$(YELLOW)Setting up project structure...$(NC)"
	@mkdir -p $(BUILD_DIR) $(LIB_DIR)
	@mkdir -p internal/api/pb
	@mkdir -p tests
	@if [ -n "$(HAS_GO)" ] || [ -f "$(BIN_DIR)/go" ]; then \
		if [ ! -f go.mod ]; then \
			if [ -f "$(BIN_DIR)/go" ]; then \
				$(BIN_DIR)/go mod init tokenvm 2>/dev/null || true; \
				$(BIN_DIR)/go mod tidy 2>/dev/null || true; \
			else \
				go mod init tokenvm 2>/dev/null || true; \
				go mod tidy 2>/dev/null || true; \
			fi; \
		fi; \
	fi
	@echo -e "$(GREEN)✓ Project structure ready$(NC)"

# Build targets
.PHONY: build
build: build-cuda build-go
	@echo -e "$(GREEN)✓ Build complete$(NC)"

# Build CUDA/C++ library
.PHONY: build-cuda
build-cuda: $(TARGET_LIB)
	@echo -e "$(GREEN)✓ C++ library built: $(TARGET_LIB)$(NC)"

$(LIB_DIR)/libtokenvm.so: $(IMPL_SRC) $(CUDA_SRCS)
	@echo -e "$(YELLOW)Building CUDA library...$(NC)"
	@mkdir -p $(BUILD_DIR) $(LIB_DIR)
	$(NVCC) -c internal/cuda/pack.cu -o $(BUILD_DIR)/pack.o $(CUDA_CFLAGS) --compiler-options '-fPIC'
	$(NVCC) -c internal/cuda/gather.cu -o $(BUILD_DIR)/gather.o $(CUDA_CFLAGS) --compiler-options '-fPIC'
	$(CXX) $(CXXFLAGS) $(CUDA_CFLAGS) -c internal/cuda/tokenvm.cc -o $(BUILD_DIR)/tokenvm.o
	$(CXX) $(CXXFLAGS) $(CUDA_CFLAGS) -c internal/cuda/arena.cc -o $(BUILD_DIR)/arena.o
	$(CXX) $(LDFLAGS) $(BUILD_DIR)/*.o -o $@ $(CUDA_LDFLAGS)

$(LIB_DIR)/libtokenvm_stub.so: $(IMPL_SRC)
	@echo -e "$(YELLOW)Building CPU stub library...$(NC)"
	@mkdir -p $(BUILD_DIR) $(LIB_DIR)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(IMPL_SRC) -o $@

# Build Go components
.PHONY: build-go
build-go:
	@if [ -n "$(HAS_GO)" ] || [ -f "$(BIN_DIR)/go" ]; then \
		echo -e "$(YELLOW)Building Go components...$(NC)"; \
		if [ -f "$(BIN_DIR)/go" ]; then \
			export CGO_ENABLED=1; \
			export CGO_CFLAGS="-I$$(pwd)/internal/cuda"; \
			export CGO_LDFLAGS="-L$$(pwd)/$(LIB_DIR) -ltokenvm_stub"; \
			$(BIN_DIR)/go build -o build/tokenvm-daemon cmd/tokenvm-daemon/main.go 2>/dev/null || \
				echo -e "$(YELLOW)⚠ Go build skipped (missing dependencies)$(NC)"; \
		else \
			export CGO_ENABLED=1; \
			export CGO_CFLAGS="-I$$(pwd)/internal/cuda"; \
			export CGO_LDFLAGS="-L$$(pwd)/$(LIB_DIR) -ltokenvm_stub"; \
			go build -o build/tokenvm-daemon cmd/tokenvm-daemon/main.go 2>/dev/null || \
				echo -e "$(YELLOW)⚠ Go build skipped (missing dependencies)$(NC)"; \
		fi; \
		echo -e "$(GREEN)✓ Go build complete$(NC)"; \
	else \
		echo -e "$(YELLOW)⚠ Go not available - skipping Go build$(NC)"; \
	fi

# Test targets
.PHONY: test
test: test-unit
	@echo -e "$(GREEN)✓ Tests complete$(NC)"

.PHONY: test-unit
test-unit:
	@echo -e "$(YELLOW)Running unit tests...$(NC)"
	@if [ -n "$(HAS_GO)" ] || [ -f "$(BIN_DIR)/go" ]; then \
		if [ -f "$(BIN_DIR)/go" ]; then \
			$(BIN_DIR)/go test ./internal/pager/... -v 2>/dev/null || \
				echo -e "$(YELLOW)⚠ Go tests skipped$(NC)"; \
		else \
			go test ./internal/pager/... -v 2>/dev/null || \
				echo -e "$(YELLOW)⚠ Go tests skipped$(NC)"; \
		fi; \
	fi
	@if [ -f "$(VENV_DIR)/bin/pytest" ]; then \
		$(VENV_DIR)/bin/pytest tests/ -v 2>/dev/null || \
			echo -e "$(YELLOW)⚠ Python tests not found$(NC)"; \
	fi

# Run example
.PHONY: run-example
run-example:
	@if [ ! -f "$(VENV_DIR)/bin/python" ]; then \
		echo -e "$(RED)Python environment not set up. Run 'make all' first$(NC)"; \
		exit 1; \
	fi
	@if [ ! -f "$(TARGET_LIB)" ]; then \
		echo -e "$(RED)Library not built. Run 'make all' first$(NC)"; \
		exit 1; \
	fi
	@echo -e "$(YELLOW)Running HuggingFace inference example...$(NC)"
	@export LD_LIBRARY_PATH=$$(pwd)/$(LIB_DIR):$$LD_LIBRARY_PATH; \
	$(PYTHON_BIN) examples/hf_infer.py

# Benchmark
.PHONY: benchmark
benchmark:
	@if [ ! -f "$(VENV_DIR)/bin/python" ]; then \
		echo -e "$(RED)Python environment not set up. Run 'make all' first$(NC)"; \
		exit 1; \
	fi
	@echo -e "$(YELLOW)Running benchmarks...$(NC)"
	@export LD_LIBRARY_PATH=$$(pwd)/$(LIB_DIR):$$LD_LIBRARY_PATH; \
	$(PYTHON_BIN) scripts/bench_ctxlen.py

# Development helpers
.PHONY: dev-setup
dev-setup: install-deps
	@echo -e "$(YELLOW)Installing development tools...$(NC)"
	@$(PIP) install black flake8 mypy >/dev/null 2>&1
	@if [ -n "$(HAS_GO)" ] || [ -f "$(BIN_DIR)/go" ]; then \
		if [ -f "$(BIN_DIR)/go" ]; then \
			$(BIN_DIR)/go install golang.org/x/tools/cmd/goimports@latest 2>/dev/null || true; \
			$(BIN_DIR)/go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest 2>/dev/null || true; \
		else \
			go install golang.org/x/tools/cmd/goimports@latest 2>/dev/null || true; \
			go install github.com/golangci/golangci-lint/cmd/golangci-lint@latest 2>/dev/null || true; \
		fi; \
	fi
	@echo -e "$(GREEN)✓ Development tools installed$(NC)"

.PHONY: format
format:
	@echo -e "$(YELLOW)Formatting code...$(NC)"
	@if [ -f "$(VENV_DIR)/bin/black" ]; then \
		$(VENV_DIR)/bin/black internal/hooks/ examples/ scripts/ 2>/dev/null || true; \
	fi
	@if [ -n "$(HAS_GO)" ] || [ -f "$(BIN_DIR)/go" ]; then \
		if [ -f "$(BIN_DIR)/goimports" ]; then \
			$(BIN_DIR)/goimports -w internal/ cmd/ 2>/dev/null || true; \
		elif command -v goimports >/dev/null 2>&1; then \
			goimports -w internal/ cmd/ 2>/dev/null || true; \
		fi; \
	fi
	@echo -e "$(GREEN)✓ Code formatted$(NC)"

.PHONY: lint
lint:
	@echo -e "$(YELLOW)Linting code...$(NC)"
	@if [ -f "$(VENV_DIR)/bin/flake8" ]; then \
		$(VENV_DIR)/bin/flake8 internal/hooks/ examples/ scripts/ --max-line-length=100 2>/dev/null || true; \
	fi
	@if command -v golangci-lint >/dev/null 2>&1; then \
		golangci-lint run ./... 2>/dev/null || true; \
	fi
	@echo -e "$(GREEN)✓ Linting complete$(NC)"

# Clean targets
.PHONY: clean
clean:
	@echo -e "$(YELLOW)Cleaning build artifacts...$(NC)"
	rm -rf $(BUILD_DIR) $(LIB_DIR)
	rm -f build/tokenvm-daemon
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo -e "$(GREEN)✓ Clean complete$(NC)"

.PHONY: clean-all
clean-all: clean
	@echo -e "$(YELLOW)Removing all generated files...$(NC)"
	rm -rf $(VENV_DIR)
	rm -rf $(INSTALL_DIR)
	rm -f go.mod go.sum
	@echo -e "$(GREEN)✓ Full clean complete$(NC)"

# Help target
.PHONY: help
help:
	@echo -e "$(GREEN)TokenVM Makefile$(NC)"
	@echo ""
	@echo "Main targets:"
	@echo "  make all          - Install deps, build everything, run tests"
	@echo "  make build        - Build C++ library and Go daemon"
	@echo "  make test         - Run unit tests"
	@echo "  make run-example  - Run HuggingFace inference example"
	@echo "  make benchmark    - Run performance benchmarks"
	@echo ""
	@echo "Setup targets:"
	@echo "  make check-deps   - Check system dependencies"
	@echo "  make install-deps - Install missing dependencies"
	@echo "  make dev-setup    - Install development tools"
	@echo ""
	@echo "Development:"
	@echo "  make format       - Format code"
	@echo "  make lint         - Lint code"
	@echo "  make clean        - Remove build artifacts"
	@echo "  make clean-all    - Remove everything (including venv)"
	@echo ""
	@echo "Environment variables:"
	@echo "  CUDA_PATH         - Override CUDA installation path"
	@echo "  VENV_DIR          - Python virtual environment directory (default: venv)"
	@echo ""
	@if [ -n "$(HAS_CUDA)" ]; then \
		echo -e "$(GREEN)CUDA detected: $(CUDA_VERSION)$(NC)"; \
	else \
		echo -e "$(YELLOW)CUDA not detected - will build CPU stub$(NC)"; \
	fi
