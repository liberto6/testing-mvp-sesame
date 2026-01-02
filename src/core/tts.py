import asyncio
import numpy as np
import io
import soundfile as sf
import ffmpeg
import aiohttp
import base64
import socket
from src.utils.config import Config

class TTSManager:
    def __init__(self):
        self.inworld_voice_id = Config.INWORLD_VOICE_ID
        print(f"[TTS] Initialized using Inworld TTS")

    def set_voice(self, voice_id: str):
        """Update the voice ID dynamically"""
        self.inworld_voice_id = voice_id
        print(f"[TTS] Voice changed to: {voice_id}")

    async def _generate_inworld(self, text):
        """
        Generate audio using Inworld API.
        """
        if not Config.INWORLD_API_KEY:
            print("[TTS] Error: INWORLD_API_KEY not set.")
            return b""

        url = "https://api.inworld.ai/tts/v1/voice"
        headers = {
            "Authorization": f"Basic {Config.INWORLD_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "text": text,
            "voiceId": self.inworld_voice_id,
            "modelId": Config.INWORLD_MODEL_ID,
            "outputFormat": "mp3" 
        }

        # Robust connection settings
        # Force IPv4 to avoid DNS resolution issues in some pod environments
        timeout = aiohttp.ClientTimeout(total=10) # 10 seconds timeout

        for attempt in range(3):
            try:
                # Create a new connector for each attempt to avoid "Session is closed" errors
                conn = aiohttp.TCPConnector(family=socket.AF_INET, ssl=False)
                async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
                    async with session.post(url, json=payload, headers=headers) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            print(f"[TTS] Inworld API Error (Attempt {attempt+1}): {response.status} - {error_text}")
                            if attempt < 2:
                                await asyncio.sleep(0.5) # Short backoff
                                continue
                            return b""
                        
                        data = await response.json()
                        if "audioContent" in data:
                            return base64.b64decode(data["audioContent"])
                        else:
                            print(f"[TTS] Warning: Inworld response missing 'audioContent'. Keys: {list(data.keys())}")
                            return b""
            except Exception as e:
                print(f"[TTS] Error calling Inworld API (Attempt {attempt+1}): {e}")
                if attempt < 2:
                    await asyncio.sleep(0.5)
                    # Re-create connector if session closed unexpectedly
                    conn = aiohttp.TCPConnector(family=socket.AF_INET, ssl=False)
                else:
                    return b""
        return b""

    async def _generate_audio_chunk(self, text):
        """
        Internal async method to generate audio.
        """
        # Clean text from multiple tags or middle-sentence tags
        import re
        
        # 1. Find all supported tags
        tags = re.findall(r'\[(neutral|happy|sad|angry|fearful|disgusted|surprised)\]', text)
        
        # 2. Strip all tags from the text to prevent "reading" them
        clean_text = re.sub(r'\[.*?\]', '', text).strip()
        clean_text = re.sub(r'\(.*?\)', '', clean_text).strip() # Also remove parentheses just in case
        
        # 3. If a valid tag was found, prepend ONLY the first one to the start
        if tags:
            # Reconstruct text with single tag at start
            final_text = f"[{tags[0]}] {clean_text}"
        else:
            final_text = clean_text

        # Security check: ensure there is speakable content
        if not any(c.isalnum() for c in final_text):
            return b""

        return await self._generate_inworld(final_text)

    async def generate_audio(self, text_stream):
        """
        Generates audio from a text stream (iterator).
        Yields chunks of audio data (numpy array).
        """
        print(f"\n[TTS] Starting generation with voice: {self.inworld_voice_id}")
        
        # We need to buffer text slightly to form coherent sentences for TTS
        buffer = ""
        
        # text_stream is an async generator (from LLM)
        async for text_chunk in text_stream:
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
                        audio_bytes = await self._generate_audio_chunk(to_process)
                        if audio_bytes:
                            yield await self._convert_bytes_to_pcm(audio_bytes)

        # Process remaining buffer
        if buffer.strip():
            audio_bytes = await self._generate_audio_chunk(buffer)
            if audio_bytes:
                yield await self._convert_bytes_to_pcm(audio_bytes)

    async def _convert_bytes_to_pcm(self, audio_bytes):
        """
        Convert mp3 bytes from edge-tts to raw pcm int16 16000Hz using ffmpeg.
        Runs in executor to avoid blocking event loop.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._convert_bytes_to_pcm_sync, audio_bytes)

    def _convert_bytes_to_pcm_sync(self, audio_bytes):
        if not audio_bytes:
            return np.zeros(0, dtype=np.int16)
            
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
