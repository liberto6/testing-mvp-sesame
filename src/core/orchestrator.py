import queue
import time
import numpy as np
import asyncio
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
        self.state = "LISTENING" # LISTENING, PROCESSING, SPEAKING
        self.should_interrupt = False
        self.stop_event = asyncio.Event()

    def stop(self):
        self.stop_event.set()

    def process_audio_frame(self, frame):
        """
        Process a single audio frame: VAD check.
        Returns: is_speech (bool)
        """
        is_speech, prob = self.vad.is_speech(frame)
        return is_speech

    async def interruption_check(self):
        """
        Async check for interruption.
        Reads from audio queue without blocking.
        """
        try:
            while not self.audio.input_queue.empty():
                frame = self.audio.input_queue.get_nowait()
                is_speech = self.process_audio_frame(frame)
                
                if is_speech:
                    print("\n[!] Interruption detected!")
                    self.should_interrupt = True
                    # Clear frontend buffer immediately
                    self.audio.clear_audio_buffer()
                    
                    # Start buffering this new speech
                    self.speech_buffer = [frame]
                    self.is_speech_active = True
                    self.state = "LISTENING"
                    return True
        except queue.Empty:
            pass
        return False

    async def run(self):
        print("System is ready. Listening...")
        self.audio.start()
        
        try:
            while not self.stop_event.is_set():
                # Process events loop
                if self.state == "LISTENING":
                    await self._handle_listening()
                elif self.state == "PROCESSING":
                    await self._handle_processing()
                
                # Small sleep to yield control if needed
                await asyncio.sleep(0.001)

        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            self.audio.stop()

    async def _handle_listening(self):
        # Async read from queue (waits until data is available)
        try:
            frame = await self.audio.read_frame()
            if frame is None:
                # This happens if queue is closed or error
                await asyncio.sleep(0.01)
                return

            # VAD Check
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
                    
                    # Check for end of speech
                    silence_threshold = (Config.MIN_SILENCE_DURATION_MS * Config.SAMPLE_RATE) / (Config.CHUNK_SIZE * 1000)
                    
                    if self.silence_frames > silence_threshold:
                        print(f"\n[User] Finished speaking. ({len(self.speech_buffer)} frames)")
                        self.is_speech_active = False
                        self.state = "PROCESSING"
                        self.silence_frames = 0
        except Exception as e:
            print(f"Error in listening loop: {e}")
            await asyncio.sleep(1.0) # Prevent tight loop on error

    async def _handle_processing(self):
        """
        Process the collected speech buffer: ASR -> LLM -> TTS -> Play
        """
        if not self.speech_buffer:
            self.state = "LISTENING"
            return

        turn_start = time.perf_counter()
        audio_data = np.concatenate(self.speech_buffer)
        
        # 1. ASR (Sync for now, but fast)
        asr_start = time.perf_counter()
        print(f"\n[Pipeline] 🎤 Starting ASR...")
        # Ideally run in executor to avoid blocking event loop
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self.asr.transcribe, audio_data)
        
        asr_duration = time.perf_counter() - asr_start
        print(f"[Pipeline] ✅ ASR Finished in {asr_duration:.3f}s. User said: '{text}'")
        
        if not text.strip():
            self.state = "LISTENING"
            self.speech_buffer = []
            return

        # 2. LLM
        print(f"[Pipeline] 🧠 Starting LLM generation...")
        llm_start = time.perf_counter()
        
        # LLM generation is a generator, we iterate it.
        # Since it calls API, it might block if not async. 
        # For now assuming sync generator, we can iterate carefully or wrap.
        # But we need to pass it to TTS.
        
        response_stream = self.llm.generate_response(text)
        
        # 3. TTS & Playback
        print(f"[Pipeline] 🗣️  Starting TTS generation...")
        
        self.state = "SPEAKING"
        self.should_interrupt = False
        
        # We need to interleave generation and playback while checking for interruption
        # Since TTS generation is now async, we use 'async for'
        
        first_chunk = True
        
        async for audio_chunk in self.tts.generate_audio(response_stream):
            # Check for interruption before playing
            if await self.interruption_check():
                print("[Pipeline] 🛑 Playback interrupted by user.")
                break

            if first_chunk:
                time_to_first_audio = time.perf_counter() - turn_start
                ttft = time.perf_counter() - llm_start
                print(f"\n[Pipeline] ⚡ LATENCY REPORT:")
                print(f"  - Total Latency (End-of-Speech -> Audio): {time_to_first_audio:.3f}s")
                print(f"  - ASR Duration: {asr_duration:.3f}s")
                print(f"  - Processing (LLM+TTS) Latency: {ttft:.3f}s")
                print("-" * 40)
                first_chunk = False
            
            # Play audio (blocking for the chunk duration, but we can check interrupt between chunks)
            # For better async, we would push to an audio queue and have a separate player task.
            # Here we do a simple blocking write but check inputs.
            self.audio.play_audio(audio_chunk, interrupt_check_callback=lambda: self.should_interrupt)
            
            # Check immediately after play
            if await self.interruption_check():
                print("[Pipeline] 🛑 Playback interrupted by user.")
                break
        
        if not self.should_interrupt:
            total_turn_duration = time.perf_counter() - turn_start
            print(f"[Pipeline] 🏁 Turn finished. Total duration: {total_turn_duration:.3f}s")
        
        # Reset to listening if not already interrupted (which sets it to listening)
        if self.state == "SPEAKING":
            self.state = "LISTENING"
            self.speech_buffer = []
