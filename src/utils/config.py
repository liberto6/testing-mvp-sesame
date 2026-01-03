import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Audio settings
    SAMPLE_RATE = 16000
    CHANNELS = 1
    CHUNK_SIZE = 512  # Small chunk for low latency VAD
    FORMAT = "int16"
    
    # VAD settings
    VAD_THRESHOLD = 0.5
    MIN_SPEECH_DURATION_MS = 250
    MIN_SILENCE_DURATION_MS = 800  # Increased to 800ms for more natural turn-taking and thinking pauses
    
    # TTS Settings
    # Options: 'edge' (Microsoft Edge TTS), 'sesame' (Sesame CSM-1b GPU-accelerated)
    TTS_BACKEND = os.getenv("TTS_BACKEND", "edge")

    # Edge TTS settings (when TTS_BACKEND='edge')
    TTS_VOICE = "en-US-GuyNeural" # 'en-US-AriaNeural', 'en-GB-RyanNeural'

    # Sesame CSM-1b settings (when TTS_BACKEND='sesame')
    SESAME_SPEAKER_ID = int(os.getenv("SESAME_SPEAKER_ID", "0"))  # Speaker ID (0-based)

    # Models
    WHISPER_MODEL_SIZE = "base.en"  # 'base.en' is slightly better/faster than tiny multilingual for English
    COMPUTE_TYPE = "int8"
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
