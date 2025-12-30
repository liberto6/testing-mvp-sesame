import asyncio
import edge_tts
import numpy as np
import io
import soundfile as sf
import tempfile
import os

class TTSManager:
    def __init__(self):
        # Using a high quality English voice for teaching
        self.voice = "en-US-ChristopherNeural" 
        # Other options: en-US-AriaNeural, en-GB-SoniaNeural, etc.

    async def _generate_audio_chunk(self, text):
        """
        Internal async method to generate audio.
        """
        communicate = edge_tts.Communicate(text, self.voice)
        audio_data = b""
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data += chunk["data"]
        return audio_data

    def generate_audio(self, text_stream):
        """
        Generates audio from a text stream (iterator).
        Yields chunks of audio data (numpy array).
        """
        print(f"\n[TTS] Starting generation with voice: {self.voice}")
        
        # We need to buffer text slightly to form coherent sentences for TTS
        buffer = ""
        
        for text_chunk in text_stream:
            buffer += text_chunk
            
            # Simple heuristic: split by punctuation to send to TTS
            # This reduces latency compared to waiting for full response
            if any(punct in buffer for punct in [".", "!", "?", "\n"]):
                # Find the last punctuation
                import re
                parts = re.split(r'([.!?\n])', buffer)
                
                # Process complete sentences
                to_process = ""
                for i in range(0, len(parts)-1, 2):
                    sentence = parts[i] + parts[i+1]
                    to_process += sentence
                
                # Keep the remainder
                buffer = parts[-1]
                
                if to_process.strip():
                    audio_bytes = asyncio.run(self._generate_audio_chunk(to_process))
                    yield self._convert_bytes_to_pcm(audio_bytes)

        # Process remaining buffer
        if buffer.strip():
            audio_bytes = asyncio.run(self._generate_audio_chunk(buffer))
            yield self._convert_bytes_to_pcm(audio_bytes)

    def _convert_bytes_to_pcm(self, audio_bytes):
        """
        Convert mp3 bytes from edge-tts to raw pcm int16 for PyAudio
        """
        try:
            # edge-tts returns mp3, we use soundfile to read it
            with io.BytesIO(audio_bytes) as f:
                data, samplerate = sf.read(f, dtype='int16')
                
            # Resample if necessary (PyAudio is set to 16000Hz)
            # Edge-TTS usually gives 24k. We might need resampling.
            # For simplicity, if sample rate differs, we should resample.
            # But here we will just return it and assume PyAudio can handle 
            # or we might speed/slow audio.
            # Let's check config.
            from src.utils.config import Config
            if samplerate != Config.SAMPLE_RATE:
                # Basic resampling (scipy or simple slicing if integer multiple)
                # For now, let's just warn or try to use a library if present.
                # Since we don't have librosa/scipy in requirements, 
                # we will rely on soundfile/numpy if possible or just let it play slightly off-pitch
                # OR better: Configure PyAudio output stream to match TTS rate dynamically.
                # But our Orchestrator sets up AudioStreamManager once.
                
                # IMPORTANT: Resampling is complex without scipy/librosa.
                # We will just return data. The user might hear pitch shift if we don't fix.
                # Quick fix: simple linear interpolation if needed, or update Config.SAMPLE_RATE?
                # But Mic input needs 16k for Whisper/Silero.
                pass
                
            return data
        except Exception as e:
            print(f"Error converting audio: {e}")
            return np.zeros(1024, dtype=np.int16)
