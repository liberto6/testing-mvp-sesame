import queue
import asyncio
import numpy as np
from src.utils.config import Config

class WebSocketAudioManager:
    def __init__(self, websocket):
        self.websocket = websocket
        self.input_queue = queue.Queue()
        self.loop = asyncio.get_event_loop()
        self.is_running = True

    def start(self):
        self.is_running = True

    def stop(self):
        self.is_running = False

    def read_frame(self):
        """
        Reads a frame from the input queue (filled by the websocket receiver).
        Matches the interface of AudioStreamManager.
        """
        try:
            # Non-blocking get with timeout to allow checking is_running
            return self.input_queue.get(timeout=0.1)
        except queue.Empty:
            return None

    def play_audio(self, audio_data, interrupt_check_callback=None):
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

        # Send to WebSocket (thread-safe)
        try:
            future = asyncio.run_coroutine_threadsafe(
                self.websocket.send_bytes(audio_bytes), 
                self.loop
            )
            # Optional: wait for it to be sent? usually fire and forget is faster for streaming
            # future.result() 
        except Exception as e:
            print(f"Error sending audio to websocket: {e}")

    def add_input_audio(self, audio_bytes):
        """
        Called by the FastAPI route when new bytes arrive from client.
        """
        # Convert bytes to numpy int16 (assuming client sends raw PCM 16-bit)
        try:
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
            self.input_queue.put(audio_data)
        except Exception as e:
            print(f"Error processing input audio: {e}")
