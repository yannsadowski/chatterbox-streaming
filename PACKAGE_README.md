# Chatterbox Streaming - Unified Multilingual TTS

Chatterbox Streaming is an open source TTS model with **real-time streaming** and **multilingual support** (23 languages). Licensed under MIT, it has been benchmarked against leading closed-source systems like ElevenLabs.

## ✨ Key Features

- 🌍 **23 Languages Supported** - Arabic, Danish, German, Greek, English, Spanish, Finnish, French, Hebrew, Hindi, Italian, Japanese, Korean, Malay, Dutch, Norwegian, Polish, Portuguese, Russian, Swedish, Swahili, Turkish, Chinese
- 🎯 **Real-Time Streaming** - RTF of 0.499 on 4090 GPU, first chunk latency ~0.472s
- 🎭 **Emotion Exaggeration Control** - Unique intensity control for expressive speech
- 🔊 **Zero-Shot Voice Cloning** - Clone any voice from a short audio sample
- 💧 **Watermarked Outputs** - Built-in audio watermarking
- ⚡ **Optimized Performance** - 0.5B Llama backbone with alignment-informed inference

## 📦 Installation

### Using pip (recommended)

```bash
pip install chatterbox-streaming
```

### Using uv (faster)

```bash
uv pip install chatterbox-streaming
```

### From source

```bash
git clone https://github.com/davidbrowne17/chatterbox-streaming.git
cd chatterbox-streaming
pip install -e .
```

## 🚀 Quick Start

### English TTS (Non-Streaming)

```python
import torchaudio
from chatterbox import ChatterboxTTS

# Load English model
tts = ChatterboxTTS.from_pretrained(device="cuda", multilingual=False)

# Generate speech
text = "Hello! This is a text-to-speech example."
wav = tts.generate(text)
torchaudio.save("output.wav", wav, tts.sr)
```

### English TTS with Streaming

```python
import torchaudio
from chatterbox import ChatterboxTTS

# Load model
tts = ChatterboxTTS.from_pretrained(device="cuda", multilingual=False)

# Stream generation
text = "This text will be generated in real-time chunks."
audio_chunks = []

for audio_chunk, metrics in tts.generate_stream(
    text=text,
    chunk_size=25,
    print_metrics=True
):
    audio_chunks.append(audio_chunk)
    # Process each chunk as it arrives (play, save, stream, etc.)
    print(f"Chunk {metrics.chunk_count} received")

# Combine all chunks
full_audio = torch.cat(audio_chunks, dim=-1)
torchaudio.save("streaming_output.wav", full_audio, tts.sr)
```

### Multilingual TTS with Streaming

```python
import torch
import torchaudio
from chatterbox import ChatterboxTTS

# Load multilingual model
tts = ChatterboxTTS.from_pretrained(device="cuda", multilingual=True)

# French example with streaming
text_fr = "Bonjour! Ceci est un exemple de synthèse vocale en français."
chunks = []

for chunk, metrics in tts.generate_stream(
    text=text_fr,
    language_id="fr",  # Specify language
    chunk_size=25
):
    chunks.append(chunk)

audio = torch.cat(chunks, dim=-1)
torchaudio.save("french_output.wav", audio, tts.sr)

# Japanese example
text_ja = "こんにちは！これは日本語の音声合成の例です。"
wav = tts.generate(text=text_ja, language_id="ja")
torchaudio.save("japanese_output.wav", wav, tts.sr)
```

### Voice Cloning

```python
from chatterbox import ChatterboxTTS

tts = ChatterboxTTS.from_pretrained(device="cuda", multilingual=True)

# Prepare voice from reference audio
tts.prepare_conditionals("path/to/reference.wav", exaggeration=0.5)

# Generate with the cloned voice
wav = tts.generate(
    text="This will sound like the reference voice!",
    language_id="en"
)
```

## 🎛️ Advanced Parameters

### Generation Parameters

```python
wav = tts.generate(
    text="Your text here",
    language_id="en",           # Language code (only for multilingual)
    audio_prompt_path=None,     # Path to reference audio for voice cloning
    exaggeration=0.5,           # Emotion intensity (0.0-1.0)
    cfg_weight=0.5,             # Classifier-free guidance weight
    temperature=0.8,            # Sampling temperature
    repetition_penalty=1.2,     # Penalty for repetition
    min_p=0.05,                 # Minimum probability threshold
    top_p=1.0,                  # Top-p sampling
)
```

### Streaming Parameters

```python
for chunk, metrics in tts.generate_stream(
    text="Your text here",
    language_id="en",           # Language code (only for multilingual)
    chunk_size=25,              # Tokens per chunk
    context_window=50,          # Context for continuity
    fade_duration=0.02,         # Fade-in duration (seconds)
    print_metrics=True,         # Print RTF and latency
    **generation_params         # All generation params also work
):
    # Process chunk...
    pass
```

## 📊 Streaming Metrics

The streaming API provides real-time metrics:

```python
for chunk, metrics in tts.generate_stream(text="..."):
    print(f"Latency to first chunk: {metrics.latency_to_first_chunk}s")
    print(f"Real-time factor: {metrics.rtf}")
    print(f"Chunk count: {metrics.chunk_count}")
    print(f"Total audio duration: {metrics.total_audio_duration}s")
```

## 🌍 Supported Languages

Display all supported languages:

```python
from chatterbox import ChatterboxTTS, SUPPORTED_LANGUAGES

# Get language dictionary
languages = ChatterboxTTS.get_supported_languages()
# or
languages = SUPPORTED_LANGUAGES

for code, name in languages.items():
    print(f"{code}: {name}")
```

Output:
```
ar: Arabic
da: Danish
de: German
el: Greek
en: English
es: Spanish
fi: Finnish
fr: French
he: Hebrew
hi: Hindi
it: Italian
ja: Japanese
ko: Korean
ms: Malay
nl: Dutch
no: Norwegian
pl: Polish
pt: Portuguese
ru: Russian
sv: Swedish
sw: Swahili
tr: Turkish
zh: Chinese
```

## 💡 Tips & Best Practices

### General Use
- Match the reference clip language to the target language tag
- Default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most cases
- For fast-speaking voices, lower `cfg_weight` to ~0.3

### Expressive Speech
- Use lower `cfg_weight` (~0.3) + higher `exaggeration` (~0.7+)
- Higher exaggeration speeds up speech; lower cfg_weight compensates

### Streaming Performance
- Larger `chunk_size` = better quality, higher latency
- Smaller `chunk_size` = lower latency, potential quality trade-off
- Adjust `context_window` for smoother transitions between chunks

## 🔧 Development

Install with development dependencies:

```bash
pip install -e ".[dev]"
```

Run tests:

```bash
pytest
```

## 📝 Migration from Old API

If you were using the separate `ChatterboxMultilingualTTS` class:

**Old way:**
```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

tts = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
```

**New way (recommended):**
```python
from chatterbox import ChatterboxTTS

tts = ChatterboxTTS.from_pretrained(device="cuda", multilingual=True)
```

The old `ChatterboxMultilingualTTS` is still available as an alias for backwards compatibility.

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

## 🙏 Credits

Made with ♥️ by [Resemble AI](https://resemble.ai)

Streaming implementation by [@davidbrowne17](https://github.com/davidbrowne17)

## 🔗 Links

- [Original Chatterbox](https://github.com/resemble-ai/chatterbox)
- [Hugging Face Space](https://huggingface.co/spaces/ResembleAI/Chatterbox)
- [Demo Samples](https://resemble-ai.github.io/chatterbox_demopage/)
- [Discord Community](https://discord.gg/rJq9cRJBJ6)
