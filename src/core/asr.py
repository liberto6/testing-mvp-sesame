from faster_whisper import WhisperModel
import numpy as np
from src.utils.config import Config
import os

class ASRManager:
    def __init__(self):
        # Determine device
        device = "cpu" # Default to CPU for safety
        # In a real scenario we'd check for cuda
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading Whisper Model ({Config.WHISPER_MODEL_SIZE}) on {device}...")
        self.model = WhisperModel(
            Config.WHISPER_MODEL_SIZE, 
            device=device, 
            compute_type=Config.COMPUTE_TYPE
        )

    def transcribe(self, audio_data):
        """
        Transcribe the given audio data (numpy array).
        """
        # faster-whisper expects float32
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0

        # Transcribe with auto-detection or English preference
        # We allow auto-detection so we can hear if the student asks in Spanish
        segments, info = self.model.transcribe(audio_data, beam_size=5)
        
        text = ""
        for segment in segments:
            text += segment.text + " "
            
        return text.strip()
