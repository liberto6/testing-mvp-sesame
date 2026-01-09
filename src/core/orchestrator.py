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
        self.next_to_send_index = 0
        self.speech_buffer = []
        self.silence_frames = 0
        self.is_speech_active = False
        
        # State
        self.state = "LISTENING" # LISTENING, PROCESSING, SPEAKING
        self.should_interrupt = False
        self.stop_event = asyncio.Event()
        self.estimated_playback_end = 0
        self.is_processing = False  # Flag to prevent re-entry into _handle_processing()
        
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

                    # Notify frontend of state change
                    await self.audio.send_json({
                        "type": "STATE_CHANGE",
                        "state": "LISTENING"
                    })

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

                        # Notify frontend of state change
                        await self.audio.send_json({
                            "type": "STATE_CHANGE",
                            "state": "PROCESSING"
                        })

                        self.silence_frames = 0
        except Exception as e:
            print(f"Error in listening loop: {e}")
            await asyncio.sleep(1.0) # Prevent tight loop on error

    async def _handle_processing(self):
        """
        Process the collected speech buffer: ASR -> LLM -> TTS -> Play
        """
        # Prevent re-entry - critical for avoiding duplicate responses
        if self.is_processing:
            return

        if not self.speech_buffer:
            self.state = "LISTENING"
            return

        # Set flag and move buffer to local variable immediately
        # This prevents re-processing if this method is called again
        self.is_processing = True
        audio_data_frames = self.speech_buffer
        self.speech_buffer = []

        try:
            turn_start = time.perf_counter()
            audio_data = np.concatenate(audio_data_frames)

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

            # Send transcript to frontend
            if text.strip():
                await self.audio.send_json({
                    "type": "TRANSCRIPT",
                    "text": text.strip()
                })

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
                return

            await self._process_response(text, turn_start, asr_duration)

        finally:
            # Always reset flag, even if there's an exception
            self.is_processing = False

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

        # Notify frontend of state change
        await self.audio.send_json({
            "type": "STATE_CHANGE",
            "state": "SPEAKING"
        })

        self.should_interrupt = False
        self.next_to_send_index = 0
        
        async def generate_and_schedule():
            buffer = ""
            sentence_index = 0
            import re
            
            try:
                async for text_chunk in response_stream:
                    if self.should_interrupt:
                        break
                    
                    buffer += text_chunk
                    
                    # Split by strong punctuation
                    if any(punct in buffer for punct in [".", "!", "?", "\n"]):
                        parts = re.split(r'([.!?\n])', buffer)
                        
                        # parts: ['Hello', '!', ' How are you', '?', '']
                        if len(parts) > 1:
                            for i in range(0, len(parts)-1, 2):
                                phrase = parts[i] + parts[i+1]
                                if phrase.strip():
                                    asyncio.create_task(self.process_tts_fragment(phrase, sentence_index))
                                    sentence_index += 1
                            
                            buffer = parts[-1]
                
                # Process remaining buffer
                if buffer.strip() and not self.should_interrupt:
                    asyncio.create_task(self.process_tts_fragment(buffer, sentence_index))
                    sentence_index += 1
                
                # Wait for all sentences to be played
                while self.next_to_send_index < sentence_index:
                    if self.should_interrupt:
                        break
                    await asyncio.sleep(0.1)

            except Exception as e:
                print(f"Error in generation task: {e}")

        generation_task = asyncio.create_task(generate_and_schedule())
        
        while not generation_task.done():
            if await self.interruption_check():
                print("[Pipeline] 🛑 Playback interrupted by user.")
                self.should_interrupt = True
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

            # Notify frontend of state change
            await self.audio.send_json({
                "type": "STATE_CHANGE",
                "state": "LISTENING"
            })

            # Buffer already cleared at start of _handle_processing()
            self.last_interaction_time = time.time()

    async def handle_interrupt(self):   
        print("[Orchestrator]  ^=^{^q Handling interrupt from frontend...")

        # Signal interruption
        self.should_interrupt = True

        # Reset sequence indices for the next interaction
        self.next_to_send_index = 0
        # Si usas un contador para asignar índices a las frases, resetealo también:
        if hasattr(self, 'current_sentence_index'):
            self.current_sentence_index = 0

        # Clear frontend audio buffer immediately
        await self.audio.clear_audio_buffer()

        # Reset state
        self.state = "LISTENING"

        # Notify frontend of state change
        await self.audio.send_json({
            "type": "STATE_CHANGE",
            "state": "LISTENING"
        })

        self.speech_buffer = []
        self.is_speech_active = False
        self.silence_frames = 0
        self.last_interaction_time = time.time()

        print("[Orchestrator]  ^|^e Interrupt hand")

    async def process_tts_fragment(self, text, index):
        try:
            # Clean text for TTS
            # Remove asterisks and other non-spoken characters that might slip through
            text = text.replace("*", "").replace("`", "").strip()
            
            if not text:
                # If text is empty after cleaning, we still need to respect the order
                # Wait until it's our turn to "play" (i.e., do nothing) and then increment
                while self.next_to_send_index < index:
                    if self.should_interrupt:
                        return
                    await asyncio.sleep(0.02)
                
                if not self.should_interrupt:
                    self.next_to_send_index += 1
                return

            # 1. Generación de audio (esto ocurre en paralelo con otros fragmentos)
            # Asumo que self.tts.generate devuelve el payload de audio
            audio_payload = await self.tts.generate(text)
            
            # Convertir a PCM para playback (mantiene consistencia con el sistema actual)
            pcm_audio = await self.tts._convert_bytes_to_pcm(audio_payload)

            # 2. Control de flujo: Esperar hasta que sea el turno de este índice
            while self.next_to_send_index < index:
                if self.should_interrupt:
                    return
                await asyncio.sleep(0.02)  # Pequeña pausa para no saturar la CPU

            # 3. Doble check de interrupción antes de enviar
            if self.should_interrupt:
                return

            # 4. Envío al frontend
            await self.audio.play_audio(pcm_audio)
            
            # Calcular duración del audio para evitar solapamientos
            # pcm_audio es int16 a 16000Hz
            duration = len(pcm_audio) / 16000.0
            
            # Esperar a que termine de reproducirse (con chequeo de interrupción)
            elapsed = 0
            while elapsed < duration:
                if self.should_interrupt:
                    return
                await asyncio.sleep(0.05)
                elapsed += 0.05

            # 5. Incrementar el contador para permitir que el siguiente índice proceda
            if not self.should_interrupt:
                self.next_to_send_index += 1

        except Exception as e:
            print(f"[Orchestrator] Error processing TTS fragment {index}: {e}")
            # Incrementamos el índice incluso en error para no bloquear la cola
            self.next_to_send_index += 1


