# Copyright (c) 2025 Resemble AI
# Author: John Meade, Jeremy Hsu
# MIT License
import logging
import torch
from dataclasses import dataclass
from types import MethodType


logger = logging.getLogger(__name__)


LLAMA_ALIGNED_HEADS = [(12, 15), (13, 11), (9, 2)]


@dataclass
class AlignmentAnalysisResult:
    # was this frame detected as being part of a noisy beginning chunk with potential hallucinations?
    false_start: bool
    # was this frame detected as being part of a long tail with potential hallucinations?
    long_tail: bool
    # was this frame detected as repeating existing text content?
    repetition: bool
    # was the alignment position of this frame too far from the previous frame?
    discontinuity: bool
    # has inference reached the end of the text tokens? eg, this remains false if inference stops early
    complete: bool
    # approximate position in the text token sequence. Can be used for generating online timestamps.
    position: int


class AlignmentStreamAnalyzer:
    def __init__(
        self,
        tfmr,
        queue,
        text_tokens_slice,
        alignment_layer_idx=9,
        eos_idx=0,
        # New configurable parameters for hallucination detection
        token_repetition_threshold=5,  # Number of identical consecutive tokens before stopping (0 to disable)
        long_tail_threshold=5,  # Frames of final token activation before stopping (was 3)
        alignment_repetition_threshold=5,  # Activations in previous tokens after completion (was 3)
        excessive_tail_threshold=10,  # Max frames after completion before hard stop
        pause_tokens=None,  # List of token IDs that are allowed to repeat more (e.g., silence tokens)
        pause_token_multiplier=4.0,  # Multiply threshold for pause tokens (default: 4x)
    ):
        """
        Some transformer TTS models implicitly solve text-speech alignment in one or more of their self-attention
        activation maps. This module exploits this to perform online integrity checks which streaming.
        A hook is injected into the specified attention layer, and heuristics are used to determine alignment
        position, repetition, etc.

        NOTE: currently requires no queues.

        Args:
            token_repetition_threshold: Stop if this many identical tokens in a row (0=disabled, default=5)
            long_tail_threshold: Stop if final token activation exceeds this many frames (default=5)
            alignment_repetition_threshold: Stop if previous token activations exceed this after completion (default=5)
            excessive_tail_threshold: Hard stop if generation continues this many frames after completion (default=10)
            pause_tokens: List of token IDs for silence/pause (e.g., [4218]). These can repeat more.
            pause_token_multiplier: Multiply threshold for pause tokens (default: 4.0 = 4x the threshold)
        """
        # self.queue = queue
        self.text_tokens_slice = (i, j) = text_tokens_slice
        self.eos_idx = eos_idx
        self.alignment = torch.zeros(0, j-i)
        # self.alignment_bin = torch.zeros(0, j-i)
        self.curr_frame_pos = 0
        self.text_position = 0

        self.started = False
        self.started_at = None

        self.complete = False
        self.completed_at = None

        # Track generated tokens for repetition detection
        self.generated_tokens = []

        # Configurable thresholds
        self.token_repetition_threshold = token_repetition_threshold
        self.long_tail_threshold = long_tail_threshold
        self.alignment_repetition_threshold = alignment_repetition_threshold
        self.excessive_tail_threshold = excessive_tail_threshold

        # Pause token special handling
        self.pause_tokens = set(pause_tokens) if pause_tokens else {4218, 4137}  # Default pause tokens
        self.pause_token_multiplier = pause_token_multiplier

        # Using `output_attentions=True` is incompatible with optimized attention kernels, so
        # using it for all layers slows things down too much. We can apply it to just one layer
        # by intercepting the kwargs and adding a forward hook (credit: jrm)
        self.last_aligned_attns = []
        for i, (layer_idx, head_idx) in enumerate(LLAMA_ALIGNED_HEADS):
            self.last_aligned_attns += [None]
            self._add_attention_spy(tfmr, i, layer_idx, head_idx)

    def _add_attention_spy(self, tfmr, buffer_idx, layer_idx, head_idx):
        """
        Adds a forward hook to a specific attention layer to collect outputs.
        """
        def attention_forward_hook(module, input, output):
            """
            See `LlamaAttention.forward`; the output is a 3-tuple: `attn_output, attn_weights, past_key_value`.
            NOTE:
            - When `output_attentions=True`, `LlamaSdpaAttention.forward` calls `LlamaAttention.forward`.
            - `attn_output` has shape [B, H, T0, T0] for the 0th entry, and [B, H, 1, T0+i] for the rest i-th.
            """
            if isinstance(output, tuple) and len(output) > 1 and output[1] is not None:
                step_attention = output[1].cpu()  # (B, n_heads, T0, Ti)
                self.last_aligned_attns[buffer_idx] = step_attention[0, head_idx]  # (T0, Ti)

        target_layer = tfmr.layers[layer_idx].self_attn
        # Register hook and store the handle
        target_layer.register_forward_hook(attention_forward_hook)
        if hasattr(tfmr, 'config') and hasattr(tfmr.config, 'output_attentions'):
            self.original_output_attentions = tfmr.config.output_attentions
            tfmr.config.output_attentions = True

    def step(self, logits, next_token=None):
        """
        Emits an AlignmentAnalysisResult into the output queue, and potentially modifies the logits to force an EOS.
        """
        # extract approximate alignment matrix chunk (1 frame at a time after the first chunk)
        aligned_attn = torch.stack(self.last_aligned_attns).mean(dim=0) # (N, N)
        i, j = self.text_tokens_slice
        if self.curr_frame_pos == 0:
            # first chunk has conditioning info, text tokens, and BOS token
            A_chunk = aligned_attn[j:, i:j].clone().cpu() # (T, S)
        else:
            # subsequent chunks have 1 frame due to KV-caching
            A_chunk = aligned_attn[:, i:j].clone().cpu() # (1, S)

        # TODO: monotonic masking; could have issue b/c spaces are often skipped.
        A_chunk[:, self.curr_frame_pos + 1:] = 0


        self.alignment = torch.cat((self.alignment, A_chunk), dim=0)

        A = self.alignment
        T, S = A.shape

        # update position
        cur_text_posn = A_chunk[-1].argmax()
        discontinuity = not(-4 < cur_text_posn - self.text_position < 7) # NOTE: very lenient!
        if not discontinuity:
            self.text_position = cur_text_posn

        # Hallucinations at the start of speech show up as activations at the bottom of the attention maps!
        # To mitigate this, we just wait until there are no activations far off-diagonal in the last 2 tokens,
        # and there are some strong activations in the first few tokens.
        false_start = (not self.started) and (A[-2:, -2:].max() > 0.1 or A[:, :4].max() < 0.5)
        self.started = not false_start
        if self.started and self.started_at is None:
            self.started_at = T

        # Is generation likely complete?
        self.complete = self.complete or self.text_position >= S - 3
        if self.complete and self.completed_at is None:
            self.completed_at = T

        # NOTE: EOS rarely assigned activations, and second-last token is often punctuation, so use last 3 tokens.
        # NOTE: due to the false-start behaviour, we need to make sure we skip activations for the first few tokens.
        last_text_token_duration = A[15:, -3:].sum()

        # Activations for the final token that last too long are likely hallucinations.
        long_tail = (
            self.long_tail_threshold > 0 and
            self.complete and
            (A[self.completed_at:, -3:].sum(dim=0).max() >= self.long_tail_threshold)
        )

        # If there are activations in previous tokens after generation has completed, assume this is a repetition error.
        alignment_repetition = (
            self.alignment_repetition_threshold > 0 and
            self.complete and
            (A[self.completed_at:, :-5].max(dim=1).values.sum() > self.alignment_repetition_threshold)
        )

        # Hard stop if generation continues too long after completion
        excessive_tail = (
            self.excessive_tail_threshold > 0 and
            self.complete and
            self.completed_at is not None and
            (T - self.completed_at > self.excessive_tail_threshold)
        )
        
        # Track generated tokens for repetition detection
        if next_token is not None:
            # Convert tensor to scalar if needed
            if isinstance(next_token, torch.Tensor):
                token_id = next_token.item() if next_token.numel() == 1 else next_token.view(-1)[0].item()
            else:
                token_id = next_token
            self.generated_tokens.append(token_id)

            # Keep only last tokens needed for repetition check (consider pause token multiplier)
            max_pause_threshold = int(self.token_repetition_threshold * self.pause_token_multiplier) if self.pause_token_multiplier > 0 else self.token_repetition_threshold
            max_window = max(8, max_pause_threshold, self.token_repetition_threshold)
            if len(self.generated_tokens) > max_window:
                self.generated_tokens = self.generated_tokens[-max_window:]

        # Check for excessive token repetition (configurable threshold)
        token_repetition = False
        if self.token_repetition_threshold > 0 and len(self.generated_tokens) >= self.token_repetition_threshold:
            # Check if all last N tokens are the same
            if len(set(self.generated_tokens[-self.token_repetition_threshold:])) == 1:
                repeated_token = self.generated_tokens[-1]

                # Special handling for pause tokens - allow them to repeat more
                if repeated_token in self.pause_tokens:
                    # Use multiplied threshold for pause tokens
                    pause_threshold = int(self.token_repetition_threshold * self.pause_token_multiplier)
                    if len(self.generated_tokens) >= pause_threshold:
                        if len(set(self.generated_tokens[-pause_threshold:])) == 1:
                            token_repetition = True
                            logger.warning(f"🚨 Detected {pause_threshold}x repetition of PAUSE token {repeated_token} (threshold: {pause_threshold})")
                        else:
                            logger.info(f"ℹ️  Pause token {repeated_token} repeated {self.token_repetition_threshold}x (allowed up to {pause_threshold}x)")
                    # else: not enough repetitions yet for pause token
                else:
                    # Regular token - use normal threshold
                    token_repetition = True
                    logger.warning(f"🚨 Detected {self.token_repetition_threshold}x repetition of token {repeated_token}")

        # Suppress EoS to prevent early termination
        if cur_text_posn < S - 3 and S > 5:  # Only suppress if text is longer than 5 tokens
            logits[..., self.eos_idx] = -2**15

        # If a bad ending is detected, force emit EOS by modifying logits
        # NOTE: this means logits may be inconsistent with latents!
        if long_tail or alignment_repetition or token_repetition or excessive_tail:
            logger.warning(f"forcing EOS token, {long_tail=}, {alignment_repetition=}, {token_repetition=}, {excessive_tail=}")
            # (±2**15 is safe for all dtypes >= 16bit)
            logits = -(2**15) * torch.ones_like(logits)
            logits[..., self.eos_idx] = 2**15

        self.curr_frame_pos += 1
        return logits
