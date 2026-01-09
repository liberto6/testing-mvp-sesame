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
        self.estimated_playback_end = 0
        
        # Interaction timer
        self.last_interaction_time = time.time()
        self.silence_timeout = 15.0 # Seconds

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
                    self.last_interaction_time = time.time() # Reset silence timer on interruption
                    # Clear frontend buffer immediately
                    await self.audio.clear_audio_buffer()
                    
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
        self.last_interaction_time = time.time()
        
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
                self.last_interaction_time = time.time() # Keep timer reset while user is speaking
                if not self.is_speech_active:
                    print("\n[User] Started speaking...")
                    self.is_speech_active = True
                    self.speech_buffer = []
                    
                    # FORCE STOP PLAYBACK ON CLIENT
                    # The user might interrupt while audio is still playing in the browser buffer
                    # even if the server finished sending it.
                    
                    # Check if we are interrupting active playback (latent interruption)
                    if time.time() < self.estimated_playback_end:
                         print("\n[!] Interruption detected!")
                         print("[Pipeline] 🛑 Playback interrupted by user.")
                         
                    await self.audio.clear_audio_buffer()
                
                self.speech_buffer.append(frame)
                self.silence_frames = 0
                print("Hasta aqui he llegado mostro")
            else:
                if not self.is_speech_active:
                     # Check silence timeout
                     if time.time() - self.last_interaction_time > self.silence_timeout:
                          print(f"\n[Orchestrator] 🕒 Silence timeout ({self.silence_timeout}s) reached.")
                          self.last_interaction_time = time.time() # Reset immediately
                          await self._handle_silence_nudge()
                          return

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
        print("PROCESANDO")
        if not self.speech_buffer:
            self.state = "LISTENING"
            return

        turn_start = time.perf_counter()
        audio_data = np.concatenate(self.speech_buffer)
        
        # 1. ASR (Sync for now, but fast)
        asr_start = time.perf_counter()
        print(f"\n[Pipeline] 🎤 Starting ASR...")
        
        # Use a dedicated thread pool for ASR to avoid blocking the main event loop
        # and prevent it from interfering with other async tasks
        loop = asyncio.get_running_loop()
        import concurrent.futures
        
        # Ideally this executor should be created in __init__ and reused
        # Creating it every time is overhead. Let's fix this in a proper refactor later
        # For now, we keep it simple but safe.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            text = await loop.run_in_executor(executor, self.asr.transcribe, audio_data)
        
        asr_duration = time.perf_counter() - asr_start
        print(f"[Pipeline] ✅ ASR Finished in {asr_duration:.3f}s. User said: '{text}'")
        
        # --- Turn-Taking Logic ---
        # 1. Check for incomplete phrases or hesitation
        # If the user is just hesitating ("um", "ah") or the sentence trails off ("..."), 
        # we should continue listening instead of responding.
        
        is_hesitation = text.strip().lower() in ["um", "uh", "ah", "hmm", "well", "so", "like"]
        is_trailing = text.strip().endswith("...")
        
        if is_hesitation or is_trailing:
            print(f"[Pipeline] ⏳ Detected hesitation/incomplete phrase. continuing listening...")
            self.state = "LISTENING"
            self.last_interaction_time = time.time() # Reset timer if we decided not to respond
            # We DO NOT clear self.speech_buffer here. 
            # The next frames will be appended, and we'll re-transcribe the whole context next time.
            return

        if not text.strip():
            self.state = "LISTENING"
            self.last_interaction_time = time.time() # Reset timer on empty input
            self.speech_buffer = [] # Clear buffer if it was just silence/noise that produced no text
            return

        await self._process_response(text, turn_start, asr_duration)

    async def _handle_silence_nudge(self):
        """
        Triggered when silence timeout is reached.
        """
        print("[Orchestrator] Generating silence nudge...")
        # Use a system prompt injected as user message for the LLM to react
        prompt = "[System: The user has been silent for a few seconds. Say something brief to encourage them to speak, like 'Are you there?' or 'I'm listening'. Use [neutral] or [happy] tags.]"
        
        # We treat this as a new turn
        await self._process_response(prompt, is_nudge=True)

    async def _process_response(self, text, turn_start=None, asr_duration=0.0, is_nudge=False):
        if turn_start is None:
            turn_start = time.perf_counter()

        # 2. LLM
        print(f"[Pipeline] 🧠 Starting LLM generation... (Nudge: {is_nudge})")
        llm_start = time.perf_counter()
        
        response_stream = self.llm.generate_response(text)
        
        # 3. TTS & Playback
        print(f"[Pipeline] 🗣️  Starting TTS generation...")
        
        self.state = "SPEAKING"
        self.should_interrupt = False
        
        stop_signal = asyncio.Event()
        
        async def generate_and_play():
            first_chunk = True
            try:
                async for audio_chunk in self.tts.generate_audio(response_stream):
                    if stop_signal.is_set():
                        break
                    
                    if first_chunk:
                        time_to_first_audio = time.perf_counter() - turn_start
                        ttft = time.perf_counter() - llm_start
                        print(f"\n[Pipeline] ⚡ LATENCY REPORT:")
                        print(f"  - Total Latency: {time_to_first_audio:.3f}s")
                        if not is_nudge:
                            print(f"  - ASR Duration: {asr_duration:.3f}s")
                        print(f"  - Processing (LLM+TTS) Latency: {ttft:.3f}s")
                        print("-" * 40)
                        first_chunk = False

                    if stop_signal.is_set():
                        break
                        
                    await self.audio.play_audio(audio_chunk, interrupt_check_callback=lambda: stop_signal.is_set())
                    
                    duration = len(audio_chunk) / 2 / 16000
                    now = time.time()
                    if self.estimated_playback_end < now:
                        self.estimated_playback_end = now + duration
                    else:
                        self.estimated_playback_end += duration
                        
            except Exception as e:
                print(f"Error in generation task: {e}")

        generation_task = asyncio.create_task(generate_and_play())
        
        while not generation_task.done():
            if await self.interruption_check():
                print("[Pipeline] 🛑 Playback interrupted by user.")
                stop_signal.set()
                generation_task.cancel()
                try:
                    await generation_task
                except asyncio.CancelledError:
                    pass
                return 
            
            await asyncio.sleep(0.02)
            
        if not self.should_interrupt:
            total_turn_duration = time.perf_counter() - turn_start
            print(f"[Pipeline] 🏁 Turn finished. Total duration: {total_turn_duration:.3f}s")
        
        if self.state == "SPEAKING":
            self.state = "LISTENING"
            self.speech_buffer = []
            self.last_interaction_time = time.time()
