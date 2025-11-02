try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version  # For Python <3.8

__version__ = version("chatterbox-streaming")


from .tts import ChatterboxTTS, SUPPORTED_LANGUAGES, StreamingMetrics
from .vc import ChatterboxVC

# Legacy compatibility - ChatterboxMultilingualTTS is now merged into ChatterboxTTS
# Users should use ChatterboxTTS(multilingual=True) instead
ChatterboxMultilingualTTS = ChatterboxTTS

__all__ = [
    "ChatterboxTTS",
    "ChatterboxVC",
    "ChatterboxMultilingualTTS",  # Deprecated, kept for backwards compatibility
    "SUPPORTED_LANGUAGES",
    "StreamingMetrics",
]
