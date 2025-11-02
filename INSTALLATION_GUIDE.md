# Installation Guide - Chatterbox Streaming

Ce guide vous explique comment installer et builder le package Chatterbox Streaming.

## 📋 Table des matières

1. [Installation en tant qu'utilisateur](#installation-utilisateur)
2. [Installation pour développement](#installation-développement)
3. [Build du package](#build-du-package)
4. [Publication sur PyPI](#publication-pypi)

---

## 🔧 Installation utilisateur

### Option 1: Installation avec pip (stable)

```bash
pip install chatterbox-streaming
```

### Option 2: Installation avec uv (recommandé, plus rapide)

```bash
# Installer uv si nécessaire
curl -LsSf https://astral.sh/uv/install.sh | sh

# Installer le package
uv pip install chatterbox-streaming
```

### Option 3: Installation depuis GitHub

```bash
pip install git+https://github.com/davidbrowne17/chatterbox-streaming.git
```

---

## 🛠️ Installation développement

### Méthode 1: Avec pip

```bash
# Cloner le repo
git clone https://github.com/davidbrowne17/chatterbox-streaming.git
cd chatterbox-streaming

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows

# Installer en mode éditable
pip install -e .

# Installer les dépendances de dev (optionnel)
pip install -e ".[dev]"
```

### Méthode 2: Avec uv (recommandé)

```bash
# Cloner le repo
git clone https://github.com/davidbrowne17/chatterbox-streaming.git
cd chatterbox-streaming

# uv créera automatiquement l'environnement virtuel
uv pip install -e .

# Installer les dépendances de dev
uv pip install -e ".[dev]"
```

---

## 📦 Build du package

### Option 1: Build avec pip

```bash
# Utiliser le script fourni
./build_package.sh
```

Le script exécute:
1. Nettoyage des builds précédents
2. Installation des outils de build
3. Build du package (wheel + sdist)
4. Vérification avec twine

### Option 2: Build avec uv (recommandé)

```bash
# Utiliser le script UV
./build_package_uv.sh
```

### Build manuel

```bash
# Nettoyer
rm -rf dist/ build/ *.egg-info

# Installer les outils
pip install build twine
# ou
uv pip install build twine

# Builder
python -m build
# ou
uv run python -m build

# Vérifier
python -m twine check dist/*
```

Les fichiers buildés seront dans `dist/`:
- `chatterbox_streaming-X.Y.Z-py3-none-any.whl` (wheel)
- `chatterbox-streaming-X.Y.Z.tar.gz` (source distribution)

---

## 🚀 Publication sur PyPI

### Test sur TestPyPI (recommandé d'abord)

```bash
# Créer un compte sur https://test.pypi.org
# Créer un token API

# Upload vers TestPyPI
python -m twine upload --repository testpypi dist/*
# ou
uv run twine upload --repository testpypi dist/*

# Tester l'installation
pip install --index-url https://test.pypi.org/simple/ chatterbox-streaming
```

### Publication sur PyPI (production)

```bash
# Créer un compte sur https://pypi.org
# Créer un token API

# Upload vers PyPI
python -m twine upload dist/*
# ou
uv run twine upload dist/*
```

### Configuration de credentials

Créer `~/.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-...votre-token...

[testpypi]
username = __token__
password = pypi-...votre-token-test...
```

---

## ✅ Vérification de l'installation

Après installation, vérifiez:

```python
# Test import
import chatterbox
print(chatterbox.__version__)

# Test des classes principales
from chatterbox import ChatterboxTTS, SUPPORTED_LANGUAGES
print(f"Langues supportées: {len(SUPPORTED_LANGUAGES)}")
```

---

## 🐛 Dépannage

### Problème: Module not found

```bash
# Vérifier l'installation
pip list | grep chatterbox
# ou
uv pip list | grep chatterbox

# Réinstaller
pip uninstall chatterbox-streaming
pip install chatterbox-streaming
```

### Problème: Dépendances manquantes

```bash
# Réinstaller toutes les dépendances
pip install --force-reinstall chatterbox-streaming
```

### Problème: CUDA/GPU

```bash
# Vérifier PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Si False, réinstaller PyTorch avec CUDA
pip uninstall torch torchaudio
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
```

---

## 📚 Ressources

- [Documentation complète](PACKAGE_README.md)
- [Exemples d'utilisation](examples/multilingual_streaming_example.py)
- [Issues GitHub](https://github.com/davidbrowne17/chatterbox-streaming/issues)
