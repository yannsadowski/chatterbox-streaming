#!/usr/bin/env python3
"""Streaming TTS generation with CLI arguments"""
import argparse
import queue
import torchaudio as ta
import torch
import threading
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


def play_audio_chunk(audio_chunk, sample_rate):
    """Play audio chunk using sounddevice with proper sequencing"""
    if not AUDIO_AVAILABLE:
        return

    try:
        audio_np = audio_chunk.squeeze().numpy()
        sd.play(audio_np, sample_rate)
        sd.wait()
    except Exception as e:
        print(f"Error playing audio: {e}")


def audio_player_worker(audio_queue, sample_rate):
    """Worker thread that plays audio chunks from queue"""
    while True:
        try:
            audio_chunk = audio_queue.get(timeout=1.0)
            if audio_chunk is None:  # Sentinel to stop
                break
            play_audio_chunk(audio_chunk, sample_rate)
            audio_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Audio player error: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate streaming TTS audio")
    parser.add_argument(
        "--text",
        type=str,
        default="Bonjour, comment ça va? Ceci est le modèle de synthèse vocale multilingue Chatterbox, il prend en charge 23 langues.",
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
        default="output-streaming.wav",
        help="Output WAV file path"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=25,
        help="Tokens per chunk (default: 25)"
    )
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.4,
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
        default=0.8,
        help="CFG weight (default: 0.8)"
    )
    parser.add_argument(
        "--no-playback",
        action="store_true",
        help="Disable real-time audio playback"
    )
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Disable metrics printing"
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
    print(f"Chunk size: {args.chunk_size}")
    print(f"Exaggeration: {args.exaggeration}")
    print(f"Temperature: {args.temperature}")
    print(f"CFG weight: {args.cfg_weight}")

    # Setup audio playback queue and thread
    audio_queue = None
    if AUDIO_AVAILABLE and not args.no_playback:
        audio_queue = queue.Queue()
        audio_thread = threading.Thread(target=audio_player_worker, args=(audio_queue, model.sr))
        audio_thread.daemon = True
        audio_thread.start()
        print("\nReal-time audio playback enabled!")
    elif args.no_playback:
        print("\nReal-time playback disabled")
    else:
        print("\nReal-time playback unavailable (sounddevice not installed)")

    print("\nGenerating audio (streaming)...")
    streamed_chunks = []
    chunk_count = 0

    try:
        for audio_chunk, metrics in model.generate_stream(
            text=args.text,
            language_id=args.language,
            audio_prompt_path=args.voice,
            chunk_size=args.chunk_size,
            exaggeration=args.exaggeration,
            temperature=args.temperature,
            cfg_weight=args.cfg_weight,
            print_metrics=not args.no_metrics
        ):
            chunk_count += 1
            streamed_chunks.append(audio_chunk)

            # Queue audio for immediate playback
            if audio_queue:
                audio_queue.put(audio_chunk.clone())

            chunk_duration = audio_chunk.shape[-1] / model.sr
            print(f"Received chunk {chunk_count}, shape: {audio_chunk.shape}, duration: {chunk_duration:.3f}s")

            if chunk_count == 1 and audio_queue:
                print("Audio playback started!")

    except KeyboardInterrupt:
        print("\nPlayback interrupted by user")
    except Exception as e:
        print(f"Error during streaming generation: {e}")
        raise

    # Stop audio thread
    if audio_queue:
        audio_queue.join()  # Wait for all audio to finish playing
        audio_queue.put(None)  # Sentinel to stop thread

    # Save concatenated audio
    if streamed_chunks:
        full_streamed_audio = torch.cat(streamed_chunks, dim=-1)
        ta.save(args.output, full_streamed_audio, model.sr)
        print(f"\nSaved streaming audio to {args.output}")
        print(f"Total streaming chunks: {len(streamed_chunks)}")
        print(f"Final audio shape: {full_streamed_audio.shape}")
        print(f"Final audio duration: {full_streamed_audio.shape[-1] / model.sr:.3f}s")
    else:
        print("No audio chunks were generated!")

    print("\n" + "="*60)
    print("STREAMING GENERATION COMPLETE!")


if __name__ == "__main__":
    main()
