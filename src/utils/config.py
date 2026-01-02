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
    TTS_PROVIDER = "inworld" # 'edge-tts' or 'inworld'
    TTS_VOICE = "en-US-GuyNeural" # Used for EdgeTTS
    
    # Inworld TTS Settings
    INWORLD_API_KEY = os.getenv("INWORLD_API_KEY")
    INWORLD_VOICE_ID = "Ashley" # Default voice ID
    INWORLD_MODEL_ID = "inworld-tts-1"

    # Models
    WHISPER_MODEL_SIZE = "base.en"  # 'base.en' is slightly better/faster than tiny multilingual for English
    COMPUTE_TYPE = "int8"
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
