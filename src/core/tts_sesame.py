"""
Sesame CSM-1b TTS Manager with GPU-accelerated streaming
Optimized for low-latency conversational speech generation
"""

import asyncio
import numpy as np
import torch
import torchaudio
from typing import AsyncGenerator, Optional
import time
from src.utils.config import Config


class SesameCSMTTS:
    """
    GPU-accelerated TTS using Sesame CSM-1b model with streaming support.
    Generates high-quality conversational speech with low latency.
    """

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None
        self.speaker_id = Config.SESAME_SPEAKER_ID
        self.warmup_done = False

        # Performance metrics
        self.total_generation_time = 0
        self.total_chunks = 0

        print(f"[Sesame TTS] Initializing on device: {self.device}")
        if self.device == "cpu":
            print("[Sesame TTS] WARNING: Running on CPU. Performance will be significantly slower.")
            print("[Sesame TTS] For optimal latency, use CUDA-enabled GPU.")

    def _load_model(self):
        """
        Lazy loading of the model (first call only).
        Uses fp16/bfloat16 for GPU optimization.
        """
        if self.model is not None:
            return

        print("[Sesame TTS] Loading CSM-1b model...")
        start_time = time.time()

        try:
            from transformers import CsmForConditionalGeneration, AutoProcessor

            model_id = "sesame/csm-1b"

            # Load processor
            self.processor = AutoProcessor.from_pretrained(model_id)

            # Load model with optimizations
            if self.device == "cuda":
                # Use bfloat16 for better performance on modern GPUs
                self.model = CsmForConditionalGeneration.from_pretrained(
                    model_id,
                    device_map=self.device,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                )

                # Enable CUDA optimizations
                torch.backends.cudnn.benchmark = True

                # Static cache for better performance (CUDA Graph compatible)
                if hasattr(self.model, 'generation_config'):
                    self.model.generation_config.cache_implementation = "static"
                    if hasattr(self.model, 'depth_decoder'):
                        self.model.depth_decoder.generation_config.cache_implementation = "static"

                    # Set reasonable max length for conversational speech
                    # Higher values = more memory but longer utterances supported
                    self.model.generation_config.max_length = 500
            else:
                # CPU fallback (slower)
                self.model = CsmForConditionalGeneration.from_pretrained(
                    model_id,
                    device_map=self.device,
                )

            self.model.eval()  # Set to inference mode

            load_time = time.time() - start_time
            print(f"[Sesame TTS] Model loaded successfully in {load_time:.2f}s")
            print(f"[Sesame TTS] Using speaker ID: {self.speaker_id}")

        except ImportError as e:
            print(f"[Sesame TTS] ERROR: Missing required libraries: {e}")
            print("[Sesame TTS] Install with: pip install transformers>=4.52.1")
            raise
        except Exception as e:
            print(f"[Sesame TTS] ERROR loading model: {e}")
            raise

    def _warmup(self):
        """
        Warm up the model with a dummy generation to avoid first-call latency.
        This pre-allocates GPU memory and optimizes CUDA kernels.
        """
        if self.warmup_done:
            return

        print("[Sesame TTS] Warming up model...")
        try:
            dummy_text = f"[{self.speaker_id}]Hello."
            inputs = self.processor(
                dummy_text,
                add_special_tokens=True,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                _ = self.model.generate(**inputs, output_audio=True)

            # Clear cache after warmup
            if self.device == "cuda":
                torch.cuda.empty_cache()

            self.warmup_done = True
            print("[Sesame TTS] Warmup complete")
        except Exception as e:
            print(f"[Sesame TTS] Warning: Warmup failed: {e}")
            # Continue anyway - not critical

    async def generate_audio(self, text_stream: AsyncGenerator[str, None]) -> AsyncGenerator[np.ndarray, None]:
        """
        Generate audio from streaming text (from LLM).

        Args:
            text_stream: Async generator yielding text chunks from LLM

        Yields:
            numpy arrays of int16 PCM audio at 16kHz mono

        Note:
            CSM-1b generates audio at 24kHz, which is automatically resampled to 16kHz
            to match the project's audio pipeline requirements.
        """
        # Lazy load model on first use
        self._load_model()

        # Warmup on first generation
        if not self.warmup_done:
            self._warmup()

        print(f"[Sesame TTS] Starting generation with speaker: {self.speaker_id}")

        # Buffer text to form complete sentences
        # CSM-1b works better with complete phrases
        buffer = ""

        async for text_chunk in text_stream:
            buffer += text_chunk

            # Determine if we should process the buffer
            should_process = False

            # Strong punctuation (always split)
            if any(punct in buffer for punct in [".", "!", "?", "\n"]):
                should_process = True
            # Weak punctuation (split only if buffer is long enough)
            elif len(buffer) > 50 and any(punct in buffer for punct in [",", ";", ":"]):
                should_process = True

            if should_process:
                # Find the last punctuation to split at
                last_punct_idx = -1
                for punct in [".", "!", "?", "\n", ",", ";", ":"]:
                    idx = buffer.rfind(punct)
                    if idx > last_punct_idx:
                        last_punct_idx = idx

                if last_punct_idx != -1:
                    to_process = buffer[:last_punct_idx + 1].strip()
                    buffer = buffer[last_punct_idx + 1:]

                    if to_process:
                        # Generate audio for this chunk
                        async for audio_chunk in self._generate_audio_chunk(to_process):
                            yield audio_chunk

        # Process any remaining buffer
        if buffer.strip():
            async for audio_chunk in self._generate_audio_chunk(buffer.strip()):
                yield audio_chunk

        # Log performance metrics
        if self.total_chunks > 0:
            avg_time = self.total_generation_time / self.total_chunks
            print(f"[Sesame TTS] Generation complete. Avg time per chunk: {avg_time:.3f}s")

    async def _generate_audio_chunk(self, text: str) -> AsyncGenerator[np.ndarray, None]:
        """
        Generate audio for a single text chunk.

        Args:
            text: Text to convert to speech

        Yields:
            numpy int16 PCM audio chunks at 16kHz
        """
        # Skip empty or non-speakable text
        if not text or not any(c.isalnum() for c in text):
            return

        # Format text with speaker ID
        formatted_text = f"[{self.speaker_id}]{text}"

        try:
            start_time = time.time()

            # Run generation in executor to avoid blocking event loop
            audio_tensor = await asyncio.get_running_loop().run_in_executor(
                None,
                self._generate_sync,
                formatted_text
            )

            generation_time = time.time() - start_time
            self.total_generation_time += generation_time
            self.total_chunks += 1

            if audio_tensor is None:
                return

            # Convert to 16kHz PCM (project standard)
            pcm_audio = await self._convert_to_pcm(audio_tensor)

            if pcm_audio is not None and len(pcm_audio) > 0:
                duration = len(pcm_audio) / Config.SAMPLE_RATE
                rtf = generation_time / duration if duration > 0 else 0
                print(f"[Sesame TTS] Generated {duration:.2f}s audio in {generation_time:.3f}s (RTF: {rtf:.2f}x)")
                yield pcm_audio

        except Exception as e:
            print(f"[Sesame TTS] Error generating audio for '{text[:50]}...': {e}")
            import traceback
            traceback.print_exc()

    def _generate_sync(self, text: str) -> Optional[torch.Tensor]:
        """
        Synchronous generation (called in executor).

        Args:
            text: Formatted text with speaker ID

        Returns:
            Audio tensor at 24kHz or None if generation fails
        """
        try:
            # Prepare inputs
            inputs = self.processor(
                text,
                add_special_tokens=True,
                return_tensors="pt"
            ).to(self.device)

            # Generate with no gradient tracking
            with torch.no_grad():
                # Clear CUDA cache before generation
                if self.device == "cuda":
                    torch.cuda.empty_cache()

                # Generate audio (returns dict with 'audio' key)
                output = self.model.generate(
                    **inputs,
                    output_audio=True,
                    do_sample=False,  # Deterministic generation for consistency
                    temperature=1.0,
                )

                # Extract audio tensor
                # Output format: {'audio': tensor of shape [batch, samples]}
                if isinstance(output, dict) and 'audio' in output:
                    audio_tensor = output['audio']
                elif hasattr(output, 'audio'):
                    audio_tensor = output.audio
                else:
                    # Fallback: output might be the audio directly
                    audio_tensor = output

                # Handle batch dimension
                if audio_tensor.dim() == 2:
                    audio_tensor = audio_tensor[0]  # Take first batch item
                elif audio_tensor.dim() == 3:
                    audio_tensor = audio_tensor[0, 0]  # Take first batch, first channel

                return audio_tensor.cpu()

        except Exception as e:
            print(f"[Sesame TTS] Sync generation error: {e}")
            import traceback
            traceback.print_exc()
            return None

    async def _convert_to_pcm(self, audio_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        Convert audio tensor from 24kHz to 16kHz PCM int16.

        Args:
            audio_tensor: Audio tensor at 24kHz from CSM model

        Returns:
            numpy int16 array at 16kHz, or None if conversion fails
        """
        try:
            # Run conversion in executor to avoid blocking
            return await asyncio.get_running_loop().run_in_executor(
                None,
                self._convert_to_pcm_sync,
                audio_tensor
            )
        except Exception as e:
            print(f"[Sesame TTS] Error in async conversion: {e}")
            return None

    def _convert_to_pcm_sync(self, audio_tensor: torch.Tensor) -> Optional[np.ndarray]:
        """
        Synchronous audio conversion.

        CSM-1b outputs at 24kHz, but the project uses 16kHz.
        This resamples and converts to int16 PCM format.
        """
        try:
            # Ensure correct shape [1, samples] for resampling
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # Resample from 24kHz to 16kHz
            resampler = torchaudio.transforms.Resample(
                orig_freq=24000,
                new_freq=Config.SAMPLE_RATE
            )
            resampled = resampler(audio_tensor)

            # Convert to numpy
            audio_np = resampled.squeeze().numpy()

            # Normalize to [-1, 1] if needed
            if audio_np.max() > 1.0 or audio_np.min() < -1.0:
                audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))

            # Convert to int16 PCM
            audio_int16 = (audio_np * 32767).astype(np.int16)

            return audio_int16

        except Exception as e:
            print(f"[Sesame TTS] Error converting audio: {e}")
            import traceback
            traceback.print_exc()
            return None

    def cleanup(self):
        """
        Clean up GPU resources.
        Call this when shutting down to free VRAM.
        """
        if self.model is not None:
            print("[Sesame TTS] Cleaning up model...")
            del self.model
            del self.processor
            self.model = None
            self.processor = None

            if self.device == "cuda":
                torch.cuda.empty_cache()

            print("[Sesame TTS] Cleanup complete")
