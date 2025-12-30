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
            should_process = False
            
            # Strong punctuation (always split)
            if any(punct in buffer for punct in [".", "!", "?", "\n"]):
                should_process = True
            # Weak punctuation (split only if buffer is long enough)
            elif len(buffer) > 50 and any(punct in buffer for punct in [",", ";", ":"]):
                should_process = True
                
            if should_process:
                # Find the last punctuation
                import re
                # Split keeping the delimiters
                parts = re.split(r'([.!?\n,;:])', buffer)
                
                # Process complete sentences/phrases
                to_process = ""
                
                # Iterate up to the second to last element (last element is what comes after last punctuation)
                # If the last char of buffer was punctuation, parts[-1] is empty.
                
                # We need to be careful with the reconstruction.
                # Example: "Hello, world." -> ['Hello', ',', ' world', '.', '']
                
                # We want to process everything up to the last found punctuation that triggered the split
                
                current_chunk = ""
                last_processed_index = -1
                
                for i in range(0, len(parts)-1, 2):
                    phrase = parts[i] + parts[i+1]
                    current_chunk += phrase
                    
                    # If this chunk ends with strong punctuation OR is long enough with weak punctuation
                    is_strong = any(p in parts[i+1] for p in [".", "!", "?", "\n"])
                    is_weak = any(p in parts[i+1] for p in [",", ";", ":"])
                    
                    if is_strong or (is_weak and len(current_chunk) > 30):
                        to_process += current_chunk
                        current_chunk = ""
                
                # Whatever remains in current_chunk is not ready to be processed yet (unless we force it, but let's keep it in buffer)
                # Actually, simpler logic: process everything that was split.
                
                # Re-do: simpler approach.
                # Just take everything except the last part if it doesn't end in punctuation
                
                to_process = ""
                new_buffer = ""
                
                # Reconstruct
                reconstructed = ""
                for i in range(0, len(parts)-1, 2):
                    reconstructed += parts[i] + parts[i+1]
                reconstructed += parts[-1]
                
                # If we are here, we know we have some punctuation.
                # Let's just process up to the last punctuation found.
                
                # Find the LAST punctuation index in the original buffer
                last_punct_idx = -1
                for punct in [".", "!", "?", "\n", ",", ";", ":"]:
                    idx = buffer.rfind(punct)
                    if idx > last_punct_idx:
                        last_punct_idx = idx
                
                if last_punct_idx != -1:
                    to_process = buffer[:last_punct_idx+1]
                    buffer = buffer[last_punct_idx+1:]
                    
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
