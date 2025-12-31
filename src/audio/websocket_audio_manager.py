import queue
import asyncio
import numpy as np
from src.utils.config import Config

class WebSocketAudioManager:
    def __init__(self, websocket):
        self.websocket = websocket
        self.input_queue = asyncio.Queue()
        self.loop = asyncio.get_event_loop()
        self.is_running = True
        self.buffer = np.array([], dtype=np.int16)

    def start(self):
        self.is_running = True

    def stop(self):
        self.is_running = False

    async def read_frame(self):
        """
        Reads a frame from the input queue (filled by the websocket receiver).
        Matches the interface of AudioStreamManager.
        """
        try:
            # Async get
            return await self.input_queue.get()
        except asyncio.CancelledError:
            return None
        except Exception:
            return None

    async def play_audio(self, audio_data, interrupt_check_callback=None):
        """
        Sends audio data back to the client via WebSocket.
        """
        if not self.is_running:
            return

        # Check interruption before sending
        if interrupt_check_callback and interrupt_check_callback():
            return

        # Convert to bytes if numpy array
        if isinstance(audio_data, np.ndarray):
            audio_bytes = audio_data.tobytes()
        else:
            audio_bytes = audio_data

        # Send to WebSocket
        try:
            await self.websocket.send_bytes(audio_bytes)
        except Exception as e:
            print(f"Error sending audio to websocket: {e}")

    async def clear_audio_buffer(self):
        """
        Sends a signal to the client to clear its audio buffer (stop playback).
        """
        if not self.is_running:
            return
        try:
            await self.websocket.send_json({"type": "CLEAR_BUFFER"})
        except Exception as e:
            print(f"Error sending clear buffer signal: {e}")

    def add_input_audio(self, audio_bytes):
        """
        Called by the FastAPI route when new bytes arrive from client.
        Handles buffering and splitting into Config.CHUNK_SIZE chunks.
        """
        try:
            # Convert bytes to numpy int16
            new_data = np.frombuffer(audio_bytes, dtype=np.int16)
            
            # Append to internal buffer
            self.buffer = np.concatenate((self.buffer, new_data))
            
            # Process chunks of Config.CHUNK_SIZE
            chunk_size = Config.CHUNK_SIZE
            
            while len(self.buffer) >= chunk_size:
                chunk = self.buffer[:chunk_size]
                self.buffer = self.buffer[chunk_size:]
                self.input_queue.put_nowait(chunk)
                
        except Exception as e:
            print(f"Error processing input audio: {e}")
