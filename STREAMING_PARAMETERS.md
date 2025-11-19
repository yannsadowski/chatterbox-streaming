# Guide des paramètres de streaming avancés

## Vue d'ensemble

Le système de streaming de Chatterbox inclut maintenant des paramètres configurables pour contrôler :
- **La détection de répétition de tokens** : Arrête la génération si trop de tokens identiques consécutifs
- **La détection d'hallucination audio** : Arrête si l'audio continue après la fin du texte
- **Les paramètres d'échantillonnage** : `repetition_penalty`, `min_p`, `top_p`

## Problèmes résolus

### ❌ Problème 1 : Génération arrêtée trop tôt
**Symptôme** : Le modèle détecte des "répétitions" et arrête la génération prématurément

**Cause** : Le seuil de détection de répétition de tokens était trop agressif (3 tokens identiques consécutifs)

**Solution** :
```python
model.generate_stream(
    text,
    token_repetition_threshold=8,  # Plus tolérant (défaut: 5)
    # OU complètement désactiver :
    token_repetition_threshold=0,  # Désactivé
)
```

### ❌ Problème 2 : Audio d'hallucination après la fin du texte
**Symptôme** : L'audio continue de se générer alors que le texte est terminé

**Cause** : Les seuils de détection d'hallucination étaient trop courts

**Solution** :
```python
model.generate_stream(
    text,
    long_tail_threshold=3,  # Plus agressif pour couper rapidement (défaut: 5)
    alignment_repetition_threshold=3,
    excessive_tail_threshold=5,  # Arrêt forcé plus rapide
)
```

## Paramètres disponibles dans `generate_stream()`

### Paramètres d'échantillonnage

#### `repetition_penalty` (float, défaut: 1.2)
Pénalité pour la répétition de tokens lors de l'échantillonnage.
- **1.0** : Aucune pénalité
- **1.2-1.5** : Pénalité modérée (recommandé)
- **2.0+** : Pénalité forte (peut produire du texte moins naturel)

```python
# Exemple : forte pénalité contre la répétition
model.generate_stream(text, repetition_penalty=1.8)
```

#### `min_p` (float, défaut: 0.0)
Seuil de probabilité minimum pour l'échantillonnage.
- **0.0** : Désactivé (tous les tokens sont considérés)
- **0.05-0.1** : Filtre les tokens très improbables
- Plus la valeur est élevée, plus l'échantillonnage est conservateur

```python
# Exemple : filtrer les tokens peu probables
model.generate_stream(text, min_p=0.05)
```

#### `top_p` (float, défaut: 0.95)
Échantillonnage nucleus (top-p).
- **1.0** : Désactivé (tous les tokens sont considérés)
- **0.9-0.95** : Recommandé pour la plupart des cas
- **0.8 ou moins** : Plus déterministe, moins de variation

```python
# Exemple : échantillonnage plus serré
model.generate_stream(text, top_p=0.85)
```

### Paramètres de détection d'hallucination

#### `token_repetition_threshold` (int, défaut: 5)
Nombre de tokens identiques consécutifs avant arrêt forcé.
- **0** : Désactivé (permet toute répétition)
- **3-5** : Agressif (détecte rapidement)
- **8-10** : Plus tolérant

```python
# Désactiver complètement la détection de répétition de tokens
model.generate_stream(text, token_repetition_threshold=0)

# Plus tolérant (permet plus de répétitions)
model.generate_stream(text, token_repetition_threshold=8)
```

#### `long_tail_threshold` (int, défaut: 5)
Nombre de frames d'activation du token final avant arrêt.
- **0** : Désactivé
- **3** : Agressif (coupe rapidement)
- **5** : Équilibré (défaut)
- **8+** : Tolérant (permet des fins plus longues)

```python
# Couper rapidement pour éviter les hallucinations
model.generate_stream(text, long_tail_threshold=3)
```

#### `alignment_repetition_threshold` (int, défaut: 5)
Seuil de réactivation des tokens précédents après complétion.
- **0** : Désactivé
- **3** : Agressif
- **5** : Équilibré (défaut)
- **8+** : Tolérant

#### `excessive_tail_threshold` (int, défaut: 10)
Arrêt forcé après N frames au-delà de la complétion du texte.
- **0** : Désactivé (pas d'arrêt forcé)
- **5** : Agressif
- **10** : Équilibré (défaut)
- **15+** : Tolérant

## Cas d'usage et recettes

### 🎯 Cas 1 : La génération s'arrête trop tôt

**Symptômes** :
- L'audio est coupé avant la fin de la phrase
- Le modèle détecte des "répétitions" qui n'en sont pas

**Solution** :
```python
for audio_chunk, metrics in model.generate_stream(
    text,
    token_repetition_threshold=10,  # Plus tolérant
    long_tail_threshold=8,  # Permet des fins plus longues
    alignment_repetition_threshold=8,
    excessive_tail_threshold=15,  # Arrêt forcé plus tardif
):
    ...
```

### 🎯 Cas 2 : Audio continue après la fin du texte (hallucination)

**Symptômes** :
- L'audio continue de se générer alors que le texte est fini
- Sons ou répétitions en fin d'audio

**Solution** :
```python
for audio_chunk, metrics in model.generate_stream(
    text,
    long_tail_threshold=3,  # Coupe rapidement
    alignment_repetition_threshold=3,
    excessive_tail_threshold=5,  # Arrêt forcé rapide
    repetition_penalty=1.8,  # Forte pénalité contre la répétition
):
    ...
```

### 🎯 Cas 3 : Désactiver TOUTE détection (expérimental)

**⚠️ Attention** : Peut produire des hallucinations

```python
for audio_chunk, metrics in model.generate_stream(
    text,
    token_repetition_threshold=0,  # Désactivé
    long_tail_threshold=0,  # Désactivé
    alignment_repetition_threshold=0,  # Désactivé
    excessive_tail_threshold=0,  # Désactivé
):
    ...
```

### 🎯 Cas 4 : Paramètres équilibrés (recommandé)

```python
for audio_chunk, metrics in model.generate_stream(
    text,
    # Échantillonnage
    repetition_penalty=1.2,
    min_p=0.0,
    top_p=0.95,
    # Détection d'hallucination
    token_repetition_threshold=5,
    long_tail_threshold=5,
    alignment_repetition_threshold=5,
    excessive_tail_threshold=10,
):
    ...
```

## Exemples de code

Voir le fichier `example_advanced_streaming.py` pour des exemples complets montrant :
1. Paramètres par défaut (équilibrés)
2. Toute détection désactivée (génération maximale)
3. Paramètres très stricts (prévenir les coupures prématurées)
4. Détection très aggressive (arrêt rapide)
5. Désactivation sélective (seulement la détection de tokens)

## Exécuter les exemples

```bash
# Exemple de base
uv run python example_tts_stream.py

# Exemple avancé avec tous les paramètres
uv run python example_advanced_streaming.py
```

## Logs et débogage

Le système log automatiquement les détections :
```
🚨 Detected 5x repetition of token 1234
forcing EOS token, long_tail=True, alignment_repetition=False, token_repetition=False, excessive_tail=False
```

Pour voir ces logs, activez le logging :
```python
import logging
logging.basicConfig(level=logging.WARNING)
```

## Migration depuis l'ancienne version

**Avant** (hardcodé) :
```python
# Pas de contrôle sur la détection
for audio_chunk, metrics in model.generate_stream(text):
    ...
```

**Maintenant** (configurable) :
```python
# Contrôle total
for audio_chunk, metrics in model.generate_stream(
    text,
    token_repetition_threshold=8,  # Personnalisable
    repetition_penalty=1.3,  # Personnalisable
):
    ...
```

## Contribution

Ces améliorations ont été ajoutées pour résoudre les problèmes de :
1. Détection trop agressive de répétition (arrêt prématuré)
2. Hallucinations audio après la fin du texte
3. Manque de flexibilité dans les paramètres d'échantillonnage

Pour plus d'informations, voir les fichiers modifiés :
- `src/chatterbox/tts.py` : Ajout des paramètres dans `generate_stream()` et `inference_stream()`
- `src/chatterbox/models/t3/inference/alignment_stream_analyzer.py` : Seuils configurables
