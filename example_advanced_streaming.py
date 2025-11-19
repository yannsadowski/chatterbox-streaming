"""
Advanced Streaming Example - Control Repetition and Hallucination Detection

This example demonstrates how to use the new configurable parameters to control:
1. Token repetition detection
2. Audio hallucination after text completion
3. Repetition penalty and sampling parameters

Author: Enhanced by user request
"""

import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS

# Initialize model
model = ChatterboxTTS.from_pretrained(device="cuda")
text = "Welcome to the advanced streaming example! This demonstrates configurable hallucination detection and repetition control."

print("\n" + "="*80)
print("EXAMPLE 1: Default settings (balanced)")
print("="*80)
audio_chunks = []
for audio_chunk, metrics in model.generate_stream(
    text,
    chunk_size=25,
    print_metrics=True,
    # Default parameters shown explicitly
    repetition_penalty=1.2,  # Moderate penalty
    min_p=0.0,  # Disabled
    top_p=0.95,  # Nucleus sampling
    token_repetition_threshold=5,  # Stop after 5 identical tokens
    long_tail_threshold=5,  # Stop after 5 frames of final token
    alignment_repetition_threshold=5,
    excessive_tail_threshold=10,
):
    audio_chunks.append(audio_chunk)

final_audio = torch.cat(audio_chunks, dim=-1)
ta.save("output_default.wav", final_audio, model.sr)
print(f"✅ Saved: output_default.wav")


print("\n" + "="*80)
print("EXAMPLE 2: Disable ALL hallucination detection (maximum generation)")
print("="*80)
print("⚠️  This allows the model to generate freely but may hallucinate")
audio_chunks = []
for audio_chunk, metrics in model.generate_stream(
    text,
    chunk_size=25,
    print_metrics=True,
    repetition_penalty=1.1,  # Light penalty
    token_repetition_threshold=0,  # DISABLED - allow any repetition
    long_tail_threshold=0,  # DISABLED - no tail detection
    alignment_repetition_threshold=0,  # DISABLED
    excessive_tail_threshold=0,  # DISABLED - no hard stop
):
    audio_chunks.append(audio_chunk)

final_audio = torch.cat(audio_chunks, dim=-1)
ta.save("output_no_detection.wav", final_audio, model.sr)
print(f"✅ Saved: output_no_detection.wav")


print("\n" + "="*80)
print("EXAMPLE 3: Very strict settings (prevent early cutoffs)")
print("="*80)
print("🔒 This is more lenient with repetitions but still stops hallucinations")
audio_chunks = []
for audio_chunk, metrics in model.generate_stream(
    text,
    chunk_size=25,
    print_metrics=True,
    repetition_penalty=1.5,  # Strong penalty discourages repetition
    min_p=0.05,  # Filter low probability tokens
    top_p=0.9,  # Tighter nucleus sampling
    token_repetition_threshold=8,  # Allow up to 8 identical tokens (was 5)
    long_tail_threshold=8,  # Allow longer tails (was 5)
    alignment_repetition_threshold=8,  # More lenient (was 5)
    excessive_tail_threshold=15,  # Longer before hard stop (was 10)
):
    audio_chunks.append(audio_chunk)

final_audio = torch.cat(audio_chunks, dim=-1)
ta.save("output_lenient.wav", final_audio, model.sr)
print(f"✅ Saved: output_lenient.wav")


print("\n" + "="*80)
print("EXAMPLE 4: Very aggressive detection (stop quickly)")
print("="*80)
print("✂️  This cuts off aggressively to prevent any hallucination")
audio_chunks = []
for audio_chunk, metrics in model.generate_stream(
    text,
    chunk_size=25,
    print_metrics=True,
    repetition_penalty=2.0,  # Very strong penalty
    token_repetition_threshold=3,  # Stop after just 3 identical tokens
    long_tail_threshold=3,  # Stop quickly after completion
    alignment_repetition_threshold=3,
    excessive_tail_threshold=5,  # Hard stop quickly
):
    audio_chunks.append(audio_chunk)

final_audio = torch.cat(audio_chunks, dim=-1)
ta.save("output_aggressive.wav", final_audio, model.sr)
print(f"✅ Saved: output_aggressive.wav")


print("\n" + "="*80)
print("EXAMPLE 5: Only disable token repetition detection")
print("="*80)
print("🎯 This keeps alignment-based detection but allows token repetition")
audio_chunks = []
for audio_chunk, metrics in model.generate_stream(
    text,
    chunk_size=25,
    print_metrics=True,
    repetition_penalty=1.3,
    token_repetition_threshold=0,  # DISABLED - only this one
    long_tail_threshold=5,  # Keep enabled
    alignment_repetition_threshold=5,  # Keep enabled
    excessive_tail_threshold=10,  # Keep enabled
):
    audio_chunks.append(audio_chunk)

final_audio = torch.cat(audio_chunks, dim=-1)
ta.save("output_no_token_detection.wav", final_audio, model.sr)
print(f"✅ Saved: output_no_token_detection.wav")


print("\n" + "="*80)
print("🎉 All examples completed!")
print("="*80)
print("\nParameter Guide:")
print("  repetition_penalty: 1.0 (none) to 2.0+ (strong) - discourages token repetition")
print("  min_p: 0.0 (disabled) to 0.1+ - filters low probability tokens")
print("  top_p: 0.0 to 1.0 - nucleus sampling (1.0 = disabled)")
print("  token_repetition_threshold: N identical tokens before stopping (0=disabled)")
print("  long_tail_threshold: N frames of final token activation (0=disabled)")
print("  alignment_repetition_threshold: Reactivation threshold (0=disabled)")
print("  excessive_tail_threshold: Hard stop after N frames (0=disabled)")
print("\nRecommendations:")
print("  • If generation stops too early: increase thresholds or set to 0")
print("  • If audio continues after text ends: decrease thresholds")
print("  • If repetitive: increase repetition_penalty, decrease token_repetition_threshold")
