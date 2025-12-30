import torch
import numpy as np
from src.utils.config import Config

class VADManager:
    def __init__(self):
        # Load Silero VAD model
        self.model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = utils
        
        self.model.eval()
        self.reset()

    def reset(self):
        self.model.reset_states()

    def is_speech(self, audio_chunk):
        """
        Check if the current chunk contains speech.
        audio_chunk: numpy array of int16
        """
        # Convert int16 to float32 and normalize
        audio_float32 = audio_chunk.astype(np.float32) / 32768.0
        audio_tensor = torch.from_numpy(audio_float32)
        
        # Add batch dimension if needed, Silero expects (1, N) or just (N)
        # But usually for streaming it handles it.
        # Note: Silero VAD expects 512, 1024, or 1536 samples for 16khz
        
        speech_prob = self.model(audio_tensor, Config.SAMPLE_RATE).item()
        return speech_prob > Config.VAD_THRESHOLD, speech_prob
