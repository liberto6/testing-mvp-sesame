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
    MIN_SILENCE_DURATION_MS = 350
    
    # Models
    WHISPER_MODEL_SIZE = "base.en"  # 'base.en' is slightly better/faster than tiny multilingual for English
    COMPUTE_TYPE = "int8"
    
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")

    # Paths
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
