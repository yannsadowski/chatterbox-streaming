# 🚀 GitHub Actions CI/CD - Configuration Complète

## ✅ Ce qui a été configuré

### 📁 Workflows Créés

1. **`.github/workflows/python-publish.yml`** (AMÉLIORÉ)
   - Publication automatique sur PyPI lors de la création d'une release
   - Utilise UV pour des builds optimisés
   - Vérification avec twine avant publication
   - Trusted Publishing (OIDC) - sécurisé

2. **`.github/workflows/test-pypi-publish.yml`** (NOUVEAU)
   - Publication automatique sur TestPyPI
   - Déclenché par tags `-test` (ex: `v0.2.0-test`)
   - Parfait pour tester avant release production

3. **`.github/workflows/tests.yml`** (NOUVEAU)
   - Tests automatiques sur chaque push/PR
   - Matrice: Python 3.10, 3.11, 3.12
   - Vérification syntaxe + imports
   - Linting avec black et ruff

### 📚 Documentation Créée

- **`.github/WORKFLOWS_GUIDE.md`**
  - Guide complet d'utilisation des workflows
  - Instructions pour configurer Trusted Publishing
  - Workflow de release recommandé
  - Checklist et dépannage

---

## 🔧 Configuration Requise (À FAIRE)

### 1. Configurer Trusted Publishing

#### Sur PyPI (production)
1. Aller sur https://pypi.org/manage/account/publishing/
2. Ajouter un publisher:
   ```
   PyPI Project Name: chatterbox-streaming
   Owner: [votre-username]
   Repository: chatterbox-streaming
   Workflow: python-publish.yml
   Environment: pypi
   ```

#### Sur TestPyPI
1. Aller sur https://test.pypi.org/manage/account/publishing/
2. Ajouter un publisher:
   ```
   PyPI Project Name: chatterbox-streaming
   Owner: [votre-username]
   Repository: chatterbox-streaming
   Workflow: test-pypi-publish.yml
   Environment: testpypi
   ```

### 2. Créer les Environnements GitHub

Dans **Settings** → **Environments**, créer:
- `pypi` (pour production)
- `testpypi` (pour tests)

---

## 🔄 Workflow d'Utilisation

### Développement Quotidien

```bash
# 1. Créer une branche
git checkout -b feature/ma-feature

# 2. Développer
# ... code ...

# 3. Commit et push
git add .
git commit -m "Add new feature"
git push origin feature/ma-feature

# 4. Créer une PR
# → Les tests se lancent automatiquement
```

### Test sur TestPyPI

```bash
# 1. Merger la PR dans master
git checkout master
git pull origin master

# 2. Créer un tag de test
git tag v0.2.0-test
git push origin v0.2.0-test

# 3. Le workflow publie automatiquement sur TestPyPI

# 4. Tester l'installation
pip install --index-url https://test.pypi.org/simple/ chatterbox-streaming
```

### Release Production

```bash
# 1. Mettre à jour la version
# Éditer pyproject.toml: version = "0.2.0"
git add pyproject.toml
git commit -m "Bump version to 0.2.0"
git push origin master

# 2. Créer le tag
git tag v0.2.0
git push origin v0.2.0

# 3. Créer une release sur GitHub
# https://github.com/[user]/chatterbox-streaming/releases/new
# → Le workflow publie automatiquement sur PyPI!
```

---

## 🎯 Avantages de cette Configuration

✅ **Automatisation complète** - Aucune publication manuelle
✅ **Tests systématiques** - Chaque PR est testée
✅ **Environnement de test** - TestPyPI pour validation
✅ **Sécurité** - Trusted Publishing (pas de tokens)
✅ **Optimisation** - Utilise UV pour des builds rapides
✅ **Multi-versions** - Tests sur Python 3.10, 3.11, 3.12
✅ **Qualité du code** - Linting automatique

---

## 📋 Checklist de Déploiement

Avant de faire votre première release:

- [ ] Configurer Trusted Publishing sur PyPI
- [ ] Configurer Trusted Publishing sur TestPyPI
- [ ] Créer les environnements `pypi` et `testpypi` sur GitHub
- [ ] Tester localement: `./build_package_uv.sh`
- [ ] Tester sur TestPyPI avec un tag `-test`
- [ ] Vérifier que tous les tests passent
- [ ] Mettre à jour la version dans pyproject.toml
- [ ] Créer une release GitHub

---

## 🛠️ Fichiers Modifiés/Créés

```
.github/
├── workflows/
│   ├── python-publish.yml      (AMÉLIORÉ - PyPI release)
│   ├── test-pypi-publish.yml   (NOUVEAU - TestPyPI)
│   └── tests.yml               (NOUVEAU - Tests auto)
└── WORKFLOWS_GUIDE.md          (NOUVEAU - Documentation)
```

---

## 🎓 Ressources

- Guide complet: `.github/WORKFLOWS_GUIDE.md`
- GitHub Actions: https://docs.github.com/en/actions
- Trusted Publishing: https://docs.pypi.org/trusted-publishers/
- UV Documentation: https://docs.astral.sh/uv/

---

## 💡 Conseils

1. **Testez toujours sur TestPyPI d'abord**
2. **Utilisez des tags `-test` pour les versions de test**
3. **Surveillez la page Actions** pour voir les workflows
4. **Documentez vos releases** avec des notes détaillées
5. **Gardez pyproject.toml à jour** avec la bonne version

---

**🎉 Votre projet est maintenant prêt pour une CI/CD professionnelle!**
