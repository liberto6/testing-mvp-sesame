import pyaudio
import queue
import threading
import numpy as np
from src.utils.config import Config

class AudioStreamManager:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.is_running = False
        self.is_playing = False
        self.input_stream = None
        self.output_stream = None
        
    def start_input_stream(self):
        def callback(in_data, frame_count, time_info, status):
            if self.is_running:
                # Convert to float32 for VAD compatibility if needed, 
                # but usually we keep it raw and convert later.
                # Silero expects float32, but we read int16.
                audio_data = np.frombuffer(in_data, dtype=np.int16)
                self.input_queue.put(audio_data)
            return (in_data, pyaudio.paContinue)

        self.input_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=Config.CHANNELS,
            rate=Config.SAMPLE_RATE,
            input=True,
            frames_per_buffer=Config.CHUNK_SIZE,
            stream_callback=callback
        )
        self.input_stream.start_stream()

    def start_output_stream(self):
        self.output_stream = self.p.open(
            format=pyaudio.paInt16,
            channels=Config.CHANNELS,
            rate=Config.SAMPLE_RATE,
            output=True
        )

    def play_audio(self, audio_data, interrupt_check_callback=None):
        """
        Play audio data. 
        interrupt_check_callback: function that returns True if playback should stop.
        """
        self.is_playing = True
        # Assuming audio_data is a numpy array or bytes
        if isinstance(audio_data, np.ndarray):
            audio_bytes = audio_data.tobytes()
        else:
            audio_bytes = audio_data

        # Write in chunks to allow interruption
        chunk_size = 1024
        for i in range(0, len(audio_bytes), chunk_size):
            if not self.is_playing:
                break
            
            if interrupt_check_callback and interrupt_check_callback():
                self.is_playing = False
                break

            self.output_stream.write(audio_bytes[i:i+chunk_size])
        
        self.is_playing = False

    def stop_playback(self):
        """Interrupts current playback."""
        self.is_playing = False

    def start(self):
        self.is_running = True
        self.start_input_stream()
        self.start_output_stream()

    def stop(self):
        self.is_running = False
        self.is_playing = False
        if self.input_stream:
            self.input_stream.stop_stream()
            self.input_stream.close()
        if self.output_stream:
            self.output_stream.stop_stream()
            self.output_stream.close()
        self.p.terminate()

    def read_frame(self):
        try:
            return self.input_queue.get(timeout=0.1)
        except queue.Empty:
            return None
