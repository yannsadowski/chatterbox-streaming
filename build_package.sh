#!/bin/bash
# Script de build et installation du package Chatterbox Streaming

set -e  # Exit on error

echo "🚀 Chatterbox Streaming - Package Build Script"
echo "==============================================="

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

info "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info src/*.egg-info
success "Cleaned build directories"

info "Installing/upgrading build tools..."
python -m pip install --upgrade pip build twine
success "Build tools ready"

info "Building package..."
python -m build
success "Package built successfully"

info "Checking package with twine..."
python -m twine check dist/*
success "Package check passed"

echo ""
echo "📦 Build Summary:"
echo "================="
ls -lh dist/

echo ""
echo "🎉 Package built successfully!"
echo ""
echo "Next steps:"
echo "  • To install locally:       pip install dist/*.whl"
echo "  • To install in dev mode:   pip install -e ."
echo "  • To publish to PyPI:       twine upload dist/*"
echo "  • To publish to TestPyPI:   twine upload --repository testpypi dist/*"
