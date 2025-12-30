import queue
import time
import numpy as np
from src.utils.config import Config

class Orchestrator:
    def __init__(self, audio_manager, vad_manager, asr_manager, llm_manager, tts_manager):
        self.audio = audio_manager
        self.vad = vad_manager
        self.asr = asr_manager
        self.llm = llm_manager
        self.tts = tts_manager
        
        self.speech_buffer = []
        self.silence_frames = 0
        self.is_speech_active = False
        
        # State
        self.should_interrupt = False

    def process_audio_frame(self, frame):
        """
        Process a single audio frame: VAD check.
        Returns: is_speech (bool)
        """
        is_speech, prob = self.vad.is_speech(frame)
        return is_speech

    def interruption_check(self):
        """
        Callback for audio player to check if we should stop.
        This needs to drain the input queue and check VAD.
        """
        # We need to peek or read from the audio queue without blocking
        try:
            # We process up to N frames to keep latency low
            # If we find speech, we return True
            while not self.audio.input_queue.empty():
                frame = self.audio.input_queue.get_nowait()
                is_speech = self.process_audio_frame(frame)
                
                if is_speech:
                    print("\n[!] Interruption detected!")
                    self.should_interrupt = True
                    # Start buffering this new speech
                    self.speech_buffer = [frame]
                    self.is_speech_active = True
                    return True
                    
        except queue.Empty:
            pass
            
        return False

    def run(self):
        print("System is ready. Listening...")
        self.audio.start()
        
        try:
            while True:
                # 1. Read Audio
                frame = self.audio.read_frame()
                if frame is None:
                    continue

                # 2. VAD Check
                is_speech = self.process_audio_frame(frame)

                if is_speech:
                    if not self.is_speech_active:
                        print("\n[User] Started speaking...")
                        self.is_speech_active = True
                        self.speech_buffer = []
                    
                    self.speech_buffer.append(frame)
                    self.silence_frames = 0
                else:
                    if self.is_speech_active:
                        self.speech_buffer.append(frame)
                        self.silence_frames += 1
                        
                        # Check for end of speech (Silence duration)
                        # Chunk size 512 @ 16k is 32ms. 
                        # 500ms / 32ms ~= 15 frames
                        silence_threshold = (Config.MIN_SILENCE_DURATION_MS * Config.SAMPLE_RATE) / (Config.CHUNK_SIZE * 1000)
                        
                        if self.silence_frames > silence_threshold:
                            print(f"\n[User] Finished speaking. ({len(self.speech_buffer)} frames)")
                            self.is_speech_active = False
                            self.handle_turn()
                            self.speech_buffer = []
                            self.silence_frames = 0

        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            self.audio.stop()

    def handle_turn(self):
        """
        Process the collected speech buffer: ASR -> LLM -> TTS -> Play
        """
        # Concatenate frames
        if not self.speech_buffer:
            return

        turn_start = time.perf_counter()
        audio_data = np.concatenate(self.speech_buffer)
        
        # 1. ASR
        asr_start = time.perf_counter()
        print(f"\n[Pipeline] 🎤 Starting ASR...")
        text = self.asr.transcribe(audio_data)
        asr_duration = time.perf_counter() - asr_start
        print(f"[Pipeline] ✅ ASR Finished in {asr_duration:.3f}s. User said: '{text}'")
        
        if not text.strip():
            return

        # 2. LLM
        print(f"[Pipeline] 🧠 Starting LLM generation...")
        llm_start = time.perf_counter()
        response_stream = self.llm.generate_response(text)
        
        # 3. TTS & Playback
        # We assume response_stream yields text chunks.
        # We wrap it to generate audio chunks.
        print(f"[Pipeline] 🗣️  Starting TTS generation...")
        audio_stream = self.tts.generate_audio(response_stream)
        
        self.should_interrupt = False
        
        first_chunk = True
        
        for audio_chunk in audio_stream:
            if first_chunk:
                time_to_first_audio = time.perf_counter() - turn_start
                ttft = time.perf_counter() - llm_start # Approximate time to first audio from LLM start
                print(f"\n[Pipeline] ⚡ LATENCY REPORT:")
                print(f"  - Total Latency (End-of-Speech -> Audio): {time_to_first_audio:.3f}s")
                print(f"  - ASR Duration: {asr_duration:.3f}s")
                print(f"  - Processing (LLM+TTS) Latency: {ttft:.3f}s")
                print("-" * 40)
                first_chunk = False
            
            if self.should_interrupt:
                print("[Pipeline] 🛑 Playback interrupted by user.")
                break
                
            self.audio.play_audio(audio_chunk, interrupt_check_callback=self.interruption_check)
            
            # Double check after playback of chunk
            if self.should_interrupt:
                break
        
        total_turn_duration = time.perf_counter() - turn_start
        print(f"[Pipeline] 🏁 Turn finished. Total duration: {total_turn_duration:.3f}s")
