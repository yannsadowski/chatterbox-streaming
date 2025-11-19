#!/usr/bin/env python3
"""Non-streaming TTS generation with CLI arguments"""
import argparse
import torchaudio as ta
import torch
from pathlib import Path
from chatterbox.tts import ChatterboxTTS

# Try to import audio playback library
try:
    import sounddevice as sd
    import numpy as np
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("sounddevice not available. Install with: uv add sounddevice")


def play_audio(audio, sample_rate):
    """Play audio using sounddevice"""
    if not AUDIO_AVAILABLE:
        return

    try:
        audio_np = audio.squeeze().numpy()
        print("Playing audio...")
        sd.play(audio_np, sample_rate)
        sd.wait()
        print("Playback finished!")
    except Exception as e:
        print(f"Error playing audio: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate non-streaming TTS audio")
    parser.add_argument(
        "--text",
        type=str,
        # default="Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues.",
        default="Bonjour, comment ça va? Ceci est un loonng test...  Très long test.  ",
        help="Text to synthesize"
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="voices/guillaume.wav",
        help="Path to voice prompt audio file"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="fr",
        help="Language ID (e.g., 'fr', 'en', 'es')"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output-non-streaming.wav",
        help="Output WAV file path"
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.2,
        help="Exaggeration parameter (default: 0.4)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature parameter (default: 0.6)"
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.5,
        help="CFG weight (default: 0.8)"
    )
    parser.add_argument(
        "--no-playback",
        action="store_true",
        help="Disable audio playback after generation"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda, mps, cpu). Auto-detect if not specified."
    )
    parser.add_argument(
        "--multilingual",
        action="store_true",
        default=True,
        help="Use multilingual model (default: True)"
    )

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    print(f"Loading model (multilingual={args.multilingual})...")
    model = ChatterboxTTS.from_pretrained(device=device, multilingual=args.multilingual)

    print(f"\nText: {args.text}")
    print(f"Voice: {args.voice}")
    print(f"Language: {args.language}")
    print(f"Output: {args.output}")
    print(f"Exaggeration: {args.exaggeration}")
    print(f"Temperature: {args.temperature}")
    print(f"CFG weight: {args.cfg_weight}")

    print("\nGenerating audio (non-streaming)...")

    try:
        import time
        start_time = time.time()

        # Generate audio in one go (non-streaming)
        audio = model.generate(
            text=args.text,
            language_id=args.language,
            audio_prompt_path=args.voice,
            exaggeration=args.exaggeration,
            temperature=args.temperature,
            cfg_weight=args.cfg_weight,
        )

        generation_time = time.time() - start_time
        audio_duration = audio.shape[-1] / model.sr

        print(f"\n✅ Generation complete!")
        print(f"Generation time: {generation_time:.3f}s")
        print(f"Audio duration: {audio_duration:.3f}s")
        print(f"RTF (Real-Time Factor): {generation_time / audio_duration:.3f}")
        print(f"Audio shape: {audio.shape}")

    except KeyboardInterrupt:
        print("\nGeneration interrupted by user")
        return
    except Exception as e:
        print(f"Error during generation: {e}")
        raise

    # Save audio
    ta.save(args.output, audio, model.sr)
    print(f"\n💾 Saved audio to {args.output}")

    # Play audio if requested
    if AUDIO_AVAILABLE and not args.no_playback:
        print()
        play_audio(audio, model.sr)
    elif args.no_playback:
        print("\n🔇 Playback disabled")
    else:
        print("\n🔇 Playback unavailable (sounddevice not installed)")

    print("\n" + "="*60)
    print("NON-STREAMING GENERATION COMPLETE!")


if __name__ == "__main__":
    main()
