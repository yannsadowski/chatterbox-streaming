#!/bin/bash
set -e

echo "🔨 Simple Build Script"
echo "======================"

# Clean
echo "Cleaning..."
rm -rf dist/ build/ *.egg-info src/*.egg-info

# Create temp venv
echo "Creating temp venv..."
python3 -m venv /tmp/build-venv-$$

# Activate and install build
echo "Installing build tools..."
source /tmp/build-venv-$$/bin/activate
pip install -q build twine

# Build
echo "Building package..."
python -m build

# Check
echo "Checking package..."
twine check dist/*

# Cleanup
deactivate
rm -rf /tmp/build-venv-$$

# Show results
echo ""
echo "✅ Build complete!"
ls -lh dist/
