"""
TTS Factory - Creates the appropriate TTS manager based on configuration
"""

from src.utils.config import Config


def create_tts_manager():
    """
    Factory function to create the appropriate TTS manager based on Config.TTS_BACKEND.

    Returns:
        TTSManager or SesameCSMTTS instance based on configuration

    Raises:
        ValueError: If TTS_BACKEND is not recognized
    """
    backend = Config.TTS_BACKEND.lower()

    if backend == "edge":
        from src.core.tts import TTSManager
        print(f"[TTS Factory] Using Edge TTS with voice: {Config.TTS_VOICE}")
        return TTSManager()

    elif backend == "sesame":
        from src.core.tts_sesame import SesameCSMTTS
        print(f"[TTS Factory] Using Sesame CSM-1b TTS with speaker ID: {Config.SESAME_SPEAKER_ID}")
        return SesameCSMTTS()

    else:
        raise ValueError(
            f"Unknown TTS backend: '{backend}'. "
            f"Valid options: 'edge', 'sesame'. "
            f"Set TTS_BACKEND in .env or config.py"
        )


def get_available_backends():
    """
    Returns a list of available TTS backends with their descriptions.

    Returns:
        dict: Dictionary mapping backend names to descriptions
    """
    return {
        "edge": "Microsoft Edge TTS (cloud-based, low resource usage, network latency)",
        "sesame": "Sesame CSM-1b (local GPU, high quality, low latency, requires CUDA)",
    }
