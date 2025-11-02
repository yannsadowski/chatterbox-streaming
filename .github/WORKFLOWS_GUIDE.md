# GitHub Actions Workflows Guide

Ce projet utilise GitHub Actions pour automatiser le build, les tests et la publication du package.

## 📋 Workflows Disponibles

### 1. 🧪 Tests (`tests.yml`)

**Déclenché par:**
- Push sur `master`, `main`, ou `develop`
- Pull requests vers ces branches

**Ce qu'il fait:**
- Teste le code sur Python 3.10, 3.11, 3.12
- Vérifie la syntaxe du code
- Teste les imports de base
- Vérifie le formatage avec `black`
- Lint avec `ruff`

**Utilisation:**
```bash
# Simplement push ou créer une PR
git push origin master
```

---

### 2. 🧪 Publication TestPyPI (`test-pypi-publish.yml`)

**Déclenché par:**
- Tags se terminant par `-test` (ex: `v0.2.0-test`)

**Ce qu'il fait:**
- Build le package avec UV
- Vérifie avec twine
- Publie sur TestPyPI

**Utilisation:**
```bash
# 1. Créer un tag de test
git tag v0.2.0-test
git push origin v0.2.0-test

# 2. Le workflow se déclenche automatiquement
# 3. Le package est publié sur test.pypi.org

# 4. Tester l'installation
pip install --index-url https://test.pypi.org/simple/ chatterbox-streaming
```

---

### 3. 🚀 Publication PyPI (`python-publish.yml`)

**Déclenché par:**
- Création d'une release GitHub

**Ce qu'il fait:**
- Build le package avec UV
- Vérifie avec twine
- Publie sur PyPI officiel

**Utilisation:**
```bash
# 1. Mettre à jour la version dans pyproject.toml
# version = "0.2.0"

# 2. Commit et push
git add pyproject.toml
git commit -m "Bump version to 0.2.0"
git push origin master

# 3. Créer un tag
git tag v0.2.0
git push origin v0.2.0

# 4. Créer une release sur GitHub
# Aller sur: https://github.com/[user]/chatterbox-streaming/releases/new
# - Tag: v0.2.0
# - Title: "Release v0.2.0"
# - Description: Changelog
# - Publish release

# 5. Le workflow se déclenche automatiquement
# 6. Le package est publié sur pypi.org
```

---

## 🔧 Configuration Requise

### Pour TestPyPI et PyPI

Vous devez configurer les **Trusted Publishers** sur PyPI et TestPyPI:

#### PyPI (Production)

1. Aller sur https://pypi.org/manage/account/publishing/
2. Ajouter un nouveau publisher:
   - **PyPI Project Name:** `chatterbox-streaming`
   - **Owner:** `[votre-username-github]`
   - **Repository name:** `chatterbox-streaming`
   - **Workflow name:** `python-publish.yml`
   - **Environment name:** `pypi`

#### TestPyPI

1. Aller sur https://test.pypi.org/manage/account/publishing/
2. Ajouter un nouveau publisher:
   - **PyPI Project Name:** `chatterbox-streaming`
   - **Owner:** `[votre-username-github]`
   - **Repository name:** `chatterbox-streaming`
   - **Workflow name:** `test-pypi-publish.yml`
   - **Environment name:** `testpypi`

### Environnements GitHub

Créer les environnements dans GitHub:

1. Aller dans **Settings** → **Environments**
2. Créer deux environnements:
   - `pypi` (pour production)
   - `testpypi` (pour test)

---

## 🔄 Workflow de Release Recommandé

### Étape 1: Test sur TestPyPI

```bash
# 1. Développement et tests locaux
git checkout -b feature/new-feature
# ... développement ...
git commit -m "Add new feature"
git push origin feature/new-feature

# 2. Créer une PR et merger

# 3. Tester sur TestPyPI
git checkout master
git pull origin master
git tag v0.2.0-test
git push origin v0.2.0-test

# 4. Vérifier l'installation depuis TestPyPI
pip install --index-url https://test.pypi.org/simple/ chatterbox-streaming
```

### Étape 2: Release Production

```bash
# 1. Mettre à jour la version
# Éditer pyproject.toml: version = "0.2.0"
git add pyproject.toml
git commit -m "Bump version to 0.2.0"
git push origin master

# 2. Créer le tag de release
git tag v0.2.0
git push origin v0.2.0

# 3. Créer la release sur GitHub
# Interface web: https://github.com/[user]/chatterbox-streaming/releases/new

# 4. Le package est automatiquement publié sur PyPI!
```

---

## 📊 Monitoring des Workflows

### Voir les workflows en cours

Aller sur: `https://github.com/[user]/chatterbox-streaming/actions`

### Voir les logs

Cliquer sur un workflow → Cliquer sur un job → Voir les logs

### En cas d'échec

1. Vérifier les logs dans Actions
2. Corriger le problème
3. Re-trigger le workflow:
   - Pour tests: push un nouveau commit
   - Pour TestPyPI: supprimer et recréer le tag
   - Pour PyPI: créer une nouvelle release

---

## 🛠️ Développement Local

Pour tester le build localement avant de pusher:

```bash
# Avec le script UV
./build_package_uv.sh

# Vérifier le package
python -m twine check dist/*

# Installer localement
uv pip install dist/*.whl

# Tester
python -c "import chatterbox; print(chatterbox.__version__)"
```

---

## 📝 Checklist Avant Release

- [ ] Tous les tests passent localement
- [ ] Version mise à jour dans `pyproject.toml`
- [ ] CHANGELOG mis à jour (si applicable)
- [ ] Documentation à jour
- [ ] Build local réussi (`./build_package_uv.sh`)
- [ ] Test sur TestPyPI réussi
- [ ] Tag créé avec bonne version
- [ ] Release notes préparées

---

## 🔒 Sécurité

- ✅ Utilise **Trusted Publishing** (OIDC) - pas de tokens à gérer
- ✅ Permissions minimales dans les workflows
- ✅ Environnements protégés pour PyPI
- ✅ Vérification avec `twine check` avant publication

---

## 💡 Tips

1. **Toujours tester sur TestPyPI d'abord**
2. **Utiliser les tags `-test` pour les tests**
3. **Créer des releases GitHub détaillées**
4. **Surveiller les Actions pour détecter les problèmes rapidement**
5. **Utiliser UV pour des builds plus rapides**

---

## 🆘 Dépannage

### Workflow échoue lors du build

```bash
# Vérifier localement
./build_package_uv.sh
```

### Workflow échoue lors de la publication

1. Vérifier que Trusted Publishing est configuré
2. Vérifier que l'environnement existe
3. Vérifier les permissions du workflow

### Le package n'apparaît pas sur PyPI

1. Vérifier que la release est "published" (pas draft)
2. Vérifier les logs du workflow
3. Attendre quelques minutes (propagation)

---

Pour plus d'informations, consultez:
- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
