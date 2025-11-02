# Chatterbox TTS Streaming
Chatterbox is an open source TTS model. Licensed under MIT, Chatterbox has been benchmarked against leading closed-source systems like ElevenLabs, and is consistently preferred in side-by-side evaluations.
Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life. It's also the first open source TTS model to support **emotion exaggeration control**, a powerful feature that makes your voices stand out. This fork adds a streaming implementation that achieves a realtime factor of 0.499 (target < 1) on a 4090 gpu and a latency to first chunk of around 0.472s

<img width="1200" height="600" alt="Chatterbox-Multilingual" src="https://www.resemble.ai/wp-content/uploads/2025/09/Chatterbox-Multilingual-1.png" />


[![Alt Text](https://img.shields.io/badge/listen-demo_samples-blue)](https://resemble-ai.github.io/chatterbox_demopage/)
[![Alt Text](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/ResembleAI/Chatterbox)
[![Alt Text](https://static-public.podonos.com/badges/insight-on-pdns-sm-dark.svg)](https://podonos.com/resembleai/chatterbox)
[![Discord](https://img.shields.io/discord/1377773249798344776?label=join%20discord&logo=discord&style=flat)](https://discord.gg/rJq9cRJBJ6)

_Made with ♥️ by <a href="https://resemble.ai" target="_blank"><img width="100" alt="resemble-logo-horizontal" src="https://github.com/user-attachments/assets/35cf756b-3506-4943-9c72-c05ddfa4e525" /></a>

We're excited to introduce **Chatterbox Multilingual**, [Resemble AI's](https://resemble.ai) first production-grade open source TTS model supporting **23 languages** out of the box. Licensed under MIT, Chatterbox has been benchmarked against leading closed-source systems like ElevenLabs, and is consistently preferred in side-by-side evaluations.

Whether you're working on memes, videos, games, or AI agents, Chatterbox brings your content to life across languages. It's also the first open source TTS model to support **emotion exaggeration control** with robust **multilingual zero-shot voice cloning**. Try the english only version now on our [English Hugging Face Gradio app.](https://huggingface.co/spaces/ResembleAI/Chatterbox). Or try the multilingual version on our [Multilingual Hugging Face Gradio app.](https://huggingface.co/spaces/ResembleAI/Chatterbox-Multilingual-TTS).

If you like the model but need to scale or tune it for higher accuracy, check out our competitively priced TTS service (<a href="https://resemble.ai">link</a>). It delivers reliable performance with ultra-low latency of sub 200ms—ideal for production use in agents, applications, or interactive media.

# Key Details
- Multilingual, zero-shot TTS supporting 23 languages
- SoTA zeroshot English TTS
- 0.5B Llama backbone
- Unique exaggeration/intensity control
- Ultra-stable with alignment-informed inference
- Trained on 0.5M hours of cleaned data
- Watermarked outputs
- Easy voice conversion script
- **Real-time streaming generation**
- [Outperforms ElevenLabs]

# Supported Languages 
Arabic (ar) • Danish (da) • German (de) • Greek (el) • English (en) • Spanish (es) • Finnish (fi) • French (fr) • Hebrew (he) • Hindi (hi) • Italian (it) • Japanese (ja) • Korean (ko) • Malay (ms) • Dutch (nl) • Norwegian (no) • Polish (pl) • Portuguese (pt) • Russian (ru) • Swedish (sv) • Swahili (sw) • Turkish (tr) • Chinese (zh)
# Tips
- **General Use (TTS and Voice Agents):**
  - Ensure that the reference clip matches the specified language tag. Otherwise, language transfer outputs may inherit the accent of the reference clip’s language. To mitigate this, set `cfg_weight` to `0`.
  - The default settings (`exaggeration=0.5`, `cfg_weight=0.5`) work well for most prompts across all languages.
  - If the reference speaker has a fast speaking style, lowering `cfg_weight` to around `0.3` can improve pacing.

- **Expressive or Dramatic Speech:**
- Try lower `cfg_weight` values (e.g. `~0.3`) and increase `exaggeration` to around `0.7` or higher.
- Higher `exaggeration` tends to speed up speech; reducing `cfg_weight` helps compensate with slower, more deliberate pacing.

# Installation
```
python3.10 -m venv .venv
source .venv/bin/activate
pip install chatterbox-streaming
```

## Build for development
```
git clone https://github.com/davidbrowne17/chatterbox-streaming.git
pip install -e .
```
```shell
pip install chatterbox-tts
```

Alternatively, you can install from source:
```shell
# conda create -yn chatterbox python=3.11
# conda activate chatterbox

git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox
pip install -e .
```
We developed and tested Chatterbox on Python 3.11 on Debian 11 OS; the versions of the dependencies are pinned in `pyproject.toml` to ensure consistency. You can modify the code or dependencies in this installation mode.

# Usage

## Basic TTS Generation
```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

# English example
model = ChatterboxTTS.from_pretrained(device="cuda")
text = "Ezreal and Jinx teamed up with Ahri, Yasuo, and Teemo to take down the enemy's Nexus in an epic late-game pentakill."
wav = model.generate(text)
ta.save("test-english.wav", wav, model.sr)

# Multilingual examples
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device=device)

french_text = "Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues."
wav_french = multilingual_model.generate(spanish_text, language_id="fr")
ta.save("test-french.wav", wav_french, model.sr)

chinese_text = "你好，今天天气真不错，希望你有一个愉快的周末。"
wav_chinese = multilingual_model.generate(chinese_text, language_id="zh")
ta.save("test-chinese.wav", wav_chinese, model.sr)

# If you want to synthesize with a different voice, specify the audio prompt
AUDIO_PROMPT_PATH = "YOUR_FILE.wav"
wav = model.generate(text, audio_prompt_path=AUDIO_PROMPT_PATH)
ta.save("test-2.wav", wav, model.sr)
```

## Streaming TTS Generation
For real-time applications where you want to start playing audio as soon as it's available:

```python
import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")
text = "Welcome to the world of streaming text-to-speech! This audio will be generated and played in real-time chunks."

# Basic streaming
audio_chunks = []
for audio_chunk, metrics in model.generate_stream(text):
    audio_chunks.append(audio_chunk)
    # You can play audio_chunk immediately here for real-time playback
    print(f"Generated chunk {metrics.chunk_count}, RTF: {metrics.rtf:.3f}" if metrics.rtf else f"Chunk {metrics.chunk_count}")

# Combine all chunks into final audio
final_audio = torch.cat(audio_chunks, dim=-1)
ta.save("streaming_output.wav", final_audio, model.sr)
```

## Streaming with Voice Cloning
```python
import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="cuda")
text = "This streaming synthesis will use a custom voice from the reference audio file."
AUDIO_PROMPT_PATH = "reference_voice.wav"

audio_chunks = []
for audio_chunk, metrics in model.generate_stream(
    text, 
    audio_prompt_path=AUDIO_PROMPT_PATH,
    exaggeration=0.7,
    cfg_weight=0.3,
    chunk_size=25  # Smaller chunks for lower latency
):
    audio_chunks.append(audio_chunk)
    
    # Real-time metrics available
    if metrics.latency_to_first_chunk:
        print(f"First chunk latency: {metrics.latency_to_first_chunk:.3f}s")

# Save the complete streaming output
final_audio = torch.cat(audio_chunks, dim=-1)
ta.save("streaming_voice_clone.wav", final_audio, model.sr)
```

## Streaming Parameters
- `audio_prompt_path`: Reference audio path for voice cloning
- `chunk_size`: Number of speech tokens per chunk (default: 50). Smaller values = lower latency but more overhead
- `print_metrics`: Enable automatic printing of latency and RTF metrics (default: True)
- `exaggeration`: Emotion intensity control (0.0-1.0+)
- `cfg_weight`: Classifier-free guidance weight (0.0-1.0)
- `temperature`: Sampling randomness (0.1-1.0)

See `example_tts_stream.py` for more examples.

## Lora Fine-tuning
To fine-tune Chatterbox all you need are some wav audio files with the speaker voice you want to train, just the raw wavs. Place them in a folder called audio_data and run lora.py. You can configure the exact training params such as batch size, number of epochs and learning rate by modifying the values at the top of lora.py. You will need a CUDA gpu with at least 18gb of vram depending on your dataset size and training params. You can monitor the training metrics via the dynamic png created called training_metrics. This contains various graphs to help you track the training progress. If you want to try a checkpoint you can use the loadandmergecheckpoint.py (make sure to set the same R and Alpha values as you used in the training)

## GRPO Fine-tuning
Just like the lora fine-tuning for Chatterbox all you need are some wav audio files with the speaker voice you want to train, just the raw wavs. Place them in a folder called audio_data and run grpo.py. You can configure the exact training params such as batch size, number of epochs and learning rate by modifying the values at the top of grpo.py. You will need a CUDA gpu with at least 12gb of vram depending on your dataset size and training params. You can monitor the training metrics via the dynamic png created called grpo_training_metrics. This contains various graphs to help you track the training progress.

## Example metrics
Here are the example metrics for streaming latency on a 4090 using Linux
- Latency to first chunk: 0.472s
- Received chunk 1, shape: torch.Size([1, 24000]), duration: 1.000s
- Audio playback started!
- Received chunk 2, shape: torch.Size([1, 24000]), duration: 1.000s
- Received chunk 3, shape: torch.Size([1, 24000]), duration: 1.000s
- Received chunk 4, shape: torch.Size([1, 24000]), duration: 1.000s
- Received chunk 5, shape: torch.Size([1, 24000]), duration: 1.000s
- Received chunk 6, shape: torch.Size([1, 20160]), duration: 0.840s
- Total generation time: 2.915s
- Total audio duration: 5.840s
- RTF (Real-Time Factor): 0.499 (target < 1)
- Total chunks yielded: 6
See `example_tts.py` and `example_vc.py` for more examples.

# Acknowledgements
- [Cosyvoice](https://github.com/FunAudioLLM/CosyVoice)
- [Real-Time-Voice-Cloning](https://github.com/CorentinJ/Real-Time-Voice-Cloning)
- [HiFT-GAN](https://github.com/yl4579/HiFTNet)
- [Llama 3](https://github.com/meta-llama/llama3)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)

# Built-in PerTh Watermarking for Responsible AI
Every audio file generated by Chatterbox includes [Resemble AI's Perth (Perceptual Threshold) Watermarker](https://github.com/resemble-ai/perth) - imperceptible neural watermarks that survive MP3 compression, audio editing, and common manipulations while maintaining nearly 100% detection accuracy.


## Watermark extraction

You can look for the watermark using the following script.

```python
import perth
import librosa

AUDIO_PATH = "YOUR_FILE.wav"

# Load the watermarked audio
watermarked_audio, sr = librosa.load(AUDIO_PATH, sr=None)

# Initialize watermarker (same as used for embedding)
watermarker = perth.PerthImplicitWatermarker()

# Extract watermark
watermark = watermarker.get_watermark(watermarked_audio, sample_rate=sr)
print(f"Extracted watermark: {watermark}")
# Output: 0.0 (no watermark) or 1.0 (watermarked)
```


# Official Discord

👋 Join us on [Discord](https://discord.gg/rJq9cRJBJ6) and let's build something awesome together!

# Citation
If you find this model useful, please consider citing.
```
@misc{chatterboxtts2025,
  author       = {{Resemble AI}},
  title        = {{Chatterbox-TTS}},
  year         = {2025},
  howpublished = {\url{https://github.com/resemble-ai/chatterbox}},
  note         = {GitHub repository}
}
```
# Disclaimer
Don't use this model to do bad things. Prompts are sourced from freely available data on the internet.

## Streaming Implementation Author
David Browne

## Support me
Support this project on Ko-fi: https://ko-fi.com/davidbrowne17
