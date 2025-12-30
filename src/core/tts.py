import asyncio
import edge_tts
import numpy as np
import io
import soundfile as sf
import ffmpeg

class TTSManager:
    def __init__(self):
        # Using a high quality English voice for teaching
        # en-US-GuyNeural is a standard, clear American male voice
        self.voice = "en-US-GuyNeural" 
        # Other options: en-US-AriaNeural (Female), en-GB-RyanNeural (British Male)

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
        Convert mp3 bytes from edge-tts to raw pcm int16 16000Hz using ffmpeg
        """
        try:
            # Use ffmpeg to convert directly to s16le 16000Hz
            out, _ = (
                ffmpeg
                .input('pipe:0')
                .output('pipe:1', format='s16le', acodec='pcm_s16le', ac=1, ar='16000')
                .run(input=audio_bytes, capture_stdout=True, capture_stderr=True)
            )
            return np.frombuffer(out, dtype=np.int16)
        except ffmpeg.Error as e:
            print('ffmpeg error:', e.stderr.decode('utf8'))
            return np.zeros(1024, dtype=np.int16)
        except Exception as e:
            print(f"Error converting audio: {e}")
            return np.zeros(1024, dtype=np.int16)
