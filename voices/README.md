# Voice Reference Files

Placez vos fichiers audio de référence pour le clonage de voix dans ce dossier.

## Spécifications requises

- **Format** : WAV (PCM 16-bit)
- **Sample rate** : 24kHz
- **Durée** : 5-10 secondes minimum (10+ recommandé)
- **Qualité** : Enregistrement propre, sans bruit de fond
- **Contenu** : Phrase(s) naturelles en français
- **Taille** : ~500 KB par fichier (pour 10 secondes)

## Fichiers attendus

Selon la configuration dans `config/voices.json` :

- `default.wav` - Voix neutre par défaut
- `guillaume.wav` - Voix masculine adulte
- `laura.wav` - Voix féminine adulte
- `jean.wav` - Voix masculine grave
- `sophie.wav` - Voix féminine douce

## Conversion audio

Si vos fichiers ne sont pas au bon format, utilisez ffmpeg :

```bash
# Convertir en WAV 24kHz mono
ffmpeg -i input.mp3 -ar 24000 -ac 1 -sample_fmt s16 output.wav

# Extraire 10 secondes d'un fichier
ffmpeg -i input.wav -ss 00:00:00 -t 00:00:10 -ar 24000 -ac 1 output.wav
```

## Ajout d'une nouvelle voix

1. Préparer le fichier WAV selon les spécifications ci-dessus
2. Copier le fichier dans ce dossier : `voices/nouvelle_voix.wav`
3. Ajouter l'entrée dans `config/voices.json` :

```json
{
  "voices": {
    "nouvelle_voix": {
      "file": "nouvelle_voix.wav",
      "description": "Description de la voix",
      "gender": "male|female|neutral",
      "language": "fr"
    }
  }
}
```

4. Rebuild l'image Docker et redéployer

## Notes

- Les fichiers audio ne sont PAS inclus dans le repository Git par défaut (voir `.gitignore`)
- Assurez-vous d'avoir les droits sur les enregistrements vocaux utilisés
- Pour de meilleurs résultats, utilisez des enregistrements de haute qualité avec une prononciation claire
