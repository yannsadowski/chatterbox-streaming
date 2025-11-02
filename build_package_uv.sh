#!/bin/bash
# Script de build avec UV (Astral) pour Chatterbox Streaming

set -e  # Exit on error

echo "🚀 Chatterbox Streaming - UV Package Build Script"
echo "=================================================="

# Couleurs pour output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Fonction pour afficher les messages
info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Vérifier que nous sommes dans le bon répertoire
if [ ! -f "pyproject.toml" ]; then
    error "pyproject.toml not found. Please run this script from the project root."
    exit 1
fi

# Vérifier que uv est installé
if ! command -v uv &> /dev/null; then
    error "uv is not installed. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

info "UV version: $(uv --version)"

info "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info src/*.egg-info
success "Cleaned build directories"

info "Creating/using virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv
    success "Virtual environment created"
else
    success "Using existing virtual environment"
fi

info "Installing build dependencies with uv..."
uv pip install build twine
success "Build tools ready"

info "Building package with UV..."
uv run python -m build
success "Package built successfully"

info "Checking package with twine..."
uv run twine check dist/*
success "Package check passed"

echo ""
echo "📦 Build Summary:"
echo "================="
ls -lh dist/

echo ""
echo "🎉 Package built successfully with UV!"
echo ""
echo "Next steps:"
echo "  • To install locally with uv:  uv pip install dist/*.whl"
echo "  • To install in dev mode:      uv pip install -e ."
echo "  • To add as dependency:        uv add chatterbox-streaming"
echo "  • To publish to PyPI:          uv run twine upload dist/*"
