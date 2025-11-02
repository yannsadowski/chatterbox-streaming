"""
Example demonstrating the merged ChatterboxTTS with both multilingual and streaming support.

This example shows:
1. How to use the English-only model with streaming
2. How to use the multilingual model with streaming
3. How to use different languages
"""

import torch
from chatterbox.tts import ChatterboxTTS

def example_english_streaming():
    """Example: English-only model with streaming"""
    print("=== English-Only Model with Streaming ===")

    # Load English model
    tts = ChatterboxTTS.from_pretrained(device="cuda", multilingual=False)

    # Prepare voice conditioning from reference audio
    tts.prepare_conditionals("path/to/reference.wav", exaggeration=0.5)

    # Generate with streaming
    text = "Hello, this is a streaming example of text-to-speech synthesis."

    print(f"Generating: {text}")
    audio_chunks = []
    for audio_chunk, metrics in tts.generate_stream(
        text=text,
        temperature=0.8,
        chunk_size=25,
        print_metrics=True
    ):
        audio_chunks.append(audio_chunk)
        # You can play or process each chunk as it arrives
        print(f"Chunk {metrics.chunk_count} received")

    # Combine all chunks
    full_audio = torch.cat(audio_chunks, dim=-1)
    print(f"Total chunks: {len(audio_chunks)}")


def example_multilingual_streaming():
    """Example: Multilingual model with streaming"""
    print("\n=== Multilingual Model with Streaming ===")

    # Load multilingual model
    tts = ChatterboxTTS.from_pretrained(device="cuda", multilingual=True)

    # Prepare voice conditioning
    tts.prepare_conditionals("path/to/reference.wav", exaggeration=0.5)

    # Generate in different languages with streaming
    examples = [
        ("en", "Hello, this is English."),
        ("fr", "Bonjour, ceci est en français."),
        ("es", "Hola, esto es español."),
        ("de", "Hallo, das ist Deutsch."),
        ("ja", "こんにちは、これは日本語です。"),
        ("zh", "你好，这是中文。"),
    ]

    for lang_id, text in examples:
        print(f"\nGenerating in {lang_id}: {text}")
        audio_chunks = []

        for audio_chunk, metrics in tts.generate_stream(
            text=text,
            language_id=lang_id,
            temperature=0.8,
            chunk_size=25,
            print_metrics=False  # Disable metrics for this example
        ):
            audio_chunks.append(audio_chunk)

        full_audio = torch.cat(audio_chunks, dim=-1)
        print(f"Generated {len(audio_chunks)} chunks for {lang_id}")

        # Save the audio
        # torchaudio.save(f"output_{lang_id}.wav", full_audio, tts.sr)


def example_multilingual_non_streaming():
    """Example: Multilingual model without streaming"""
    print("\n=== Multilingual Model (Non-streaming) ===")

    # Load multilingual model
    tts = ChatterboxTTS.from_pretrained(device="cuda", multilingual=True)

    # Prepare voice conditioning
    tts.prepare_conditionals("path/to/reference.wav", exaggeration=0.5)

    # Generate in Spanish
    audio = tts.generate(
        text="Hola, este es un ejemplo sin streaming.",
        language_id="es",
        temperature=0.8,
        cfg_weight=0.5
    )

    print(f"Generated audio shape: {audio.shape}")
    # torchaudio.save("output_spanish.wav", audio, tts.sr)


def example_supported_languages():
    """Example: Display all supported languages"""
    print("\n=== Supported Languages ===")

    languages = ChatterboxTTS.get_supported_languages()

    print(f"Total supported languages: {len(languages)}")
    print("\nLanguage codes and names:")
    for code, name in sorted(languages.items()):
        print(f"  {code}: {name}")


if __name__ == "__main__":
    # Note: You'll need to update the paths to reference audio files

    # Display supported languages
    example_supported_languages()

    # Uncomment the examples you want to run:

    # example_english_streaming()
    # example_multilingual_streaming()
    # example_multilingual_non_streaming()

    print("\n✓ All examples completed!")
