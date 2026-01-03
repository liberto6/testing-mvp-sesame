"""
TTS Latency Comparison Script

Compares latency between Edge TTS and Sesame CSM-1b TTS backends.
Provides detailed metrics for Time to First Byte (TTFB), total generation time,
and real-time factor (RTF).

Usage:
    python test_tts_latency.py
    python test_tts_latency.py --backend sesame
    python test_tts_latency.py --backend edge
    python test_tts_latency.py --compare
"""

import asyncio
import time
import argparse
import numpy as np
from typing import List, AsyncGenerator
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import Config


class MockTextStream:
    """Simulates LLM streaming output for testing."""

    def __init__(self, text: str, chunk_delay: float = 0.05):
        self.text = text
        self.chunk_delay = chunk_delay

    async def stream(self) -> AsyncGenerator[str, None]:
        """Yield text in small chunks to simulate LLM streaming."""
        words = self.text.split()
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(self.chunk_delay)


class LatencyMetrics:
    """Tracks and reports latency metrics."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.first_audio_time = None
        self.end_time = None
        self.total_audio_duration = 0
        self.chunk_count = 0

    def start(self):
        self.start_time = time.time()

    def first_audio(self):
        if self.first_audio_time is None:
            self.first_audio_time = time.time()

    def end(self):
        self.end_time = time.time()

    def add_audio_chunk(self, audio: np.ndarray, sample_rate: int = 16000):
        """Add an audio chunk to calculate total duration."""
        self.chunk_count += 1
        duration = len(audio) / sample_rate
        self.total_audio_duration += duration

    @property
    def ttfb(self) -> float:
        """Time to first byte (audio chunk)."""
        if self.first_audio_time and self.start_time:
            return self.first_audio_time - self.start_time
        return 0

    @property
    def total_time(self) -> float:
        """Total generation time."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0

    @property
    def rtf(self) -> float:
        """Real-time factor: generation_time / audio_duration."""
        if self.total_time > 0 and self.total_audio_duration > 0:
            return self.total_time / self.total_audio_duration
        return 0

    def report(self):
        """Print formatted metrics report."""
        print(f"\n{'=' * 60}")
        print(f"  {self.name} - Latency Metrics")
        print(f"{'=' * 60}")
        print(f"  Time to First Audio (TTFB):  {self.ttfb:.3f}s")
        print(f"  Total Generation Time:        {self.total_time:.3f}s")
        print(f"  Total Audio Duration:         {self.total_audio_duration:.3f}s")
        print(f"  Real-Time Factor (RTF):       {self.rtf:.2f}x")
        print(f"  Number of Chunks:             {self.chunk_count}")
        print(f"  Avg Time per Chunk:           {self.total_time / max(self.chunk_count, 1):.3f}s")
        print(f"{'=' * 60}\n")


async def test_tts_backend(backend: str, test_text: str) -> LatencyMetrics:
    """
    Test a specific TTS backend and measure latency.

    Args:
        backend: 'edge' or 'sesame'
        test_text: Text to synthesize

    Returns:
        LatencyMetrics object with results
    """
    print(f"\n🔊 Testing {backend.upper()} TTS Backend...")
    print(f"📝 Text: \"{test_text}\"\n")

    # Temporarily override config
    original_backend = Config.TTS_BACKEND
    Config.TTS_BACKEND = backend

    try:
        # Import and create TTS manager
        from src.core.tts_factory import create_tts_manager

        tts = create_tts_manager()

        # Create mock text stream
        text_stream = MockTextStream(test_text, chunk_delay=0.02)

        # Metrics tracker
        metrics = LatencyMetrics(f"{backend.upper()} TTS")

        # Start timer
        metrics.start()

        # Generate audio
        print("⏱️  Generating audio...\n")

        async for audio_chunk in tts.generate_audio(text_stream.stream()):
            # Record first audio
            metrics.first_audio()

            # Add chunk to metrics
            metrics.add_audio_chunk(audio_chunk, Config.SAMPLE_RATE)

            # Log chunk
            duration = len(audio_chunk) / Config.SAMPLE_RATE
            print(f"  ✅ Chunk {metrics.chunk_count}: {duration:.2f}s audio")

        # End timer
        metrics.end()

        # Cleanup if available
        if hasattr(tts, 'cleanup'):
            tts.cleanup()

        return metrics

    finally:
        # Restore original config
        Config.TTS_BACKEND = original_backend


async def compare_backends(test_text: str):
    """Run comparison between Edge TTS and Sesame TTS."""

    print("\n" + "=" * 60)
    print("  TTS BACKEND COMPARISON")
    print("=" * 60)
    print(f"  Test Text: \"{test_text}\"")
    print("=" * 60)

    results = {}

    # Test Edge TTS
    try:
        edge_metrics = await test_tts_backend("edge", test_text)
        edge_metrics.report()
        results['edge'] = edge_metrics
    except Exception as e:
        print(f"❌ Edge TTS test failed: {e}")
        import traceback
        traceback.print_exc()

    # Test Sesame TTS
    try:
        sesame_metrics = await test_tts_backend("sesame", test_text)
        sesame_metrics.report()
        results['sesame'] = sesame_metrics
    except Exception as e:
        print(f"❌ Sesame TTS test failed: {e}")
        import traceback
        traceback.print_exc()

    # Comparison summary
    if 'edge' in results and 'sesame' in results:
        print("\n" + "=" * 60)
        print("  COMPARISON SUMMARY")
        print("=" * 60)

        edge = results['edge']
        sesame = results['sesame']

        def compare_metric(name, edge_val, sesame_val, lower_is_better=True):
            diff = sesame_val - edge_val
            pct = (diff / edge_val * 100) if edge_val != 0 else 0

            if lower_is_better:
                winner = "🏆 Sesame" if sesame_val < edge_val else "🏆 Edge"
                improvement = -pct  # Negative diff = improvement
            else:
                winner = "🏆 Sesame" if sesame_val > edge_val else "🏆 Edge"
                improvement = pct

            print(f"\n  {name}:")
            print(f"    Edge:    {edge_val:.3f}")
            print(f"    Sesame:  {sesame_val:.3f}")
            print(f"    Diff:    {diff:+.3f} ({improvement:+.1f}%)")
            print(f"    Winner:  {winner}")

        compare_metric("Time to First Audio (TTFB)", edge.ttfb, sesame.ttfb)
        compare_metric("Total Generation Time", edge.total_time, sesame.total_time)
        compare_metric("Real-Time Factor (RTF)", edge.rtf, sesame.rtf)

        print("\n" + "=" * 60)

        # Overall recommendation
        if sesame.total_time < edge.total_time:
            speedup = edge.total_time / sesame.total_time
            print(f"\n  ✨ Sesame is {speedup:.2f}x FASTER than Edge TTS!")
        else:
            slowdown = sesame.total_time / edge.total_time
            print(f"\n  ⚠️  Sesame is {slowdown:.2f}x slower than Edge TTS.")
            print(f"      (This may indicate GPU issues or CPU fallback)")

        print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="TTS Latency Testing Tool")
    parser.add_argument(
        "--backend",
        choices=["edge", "sesame"],
        help="Test a specific backend (edge or sesame)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare both backends"
    )
    parser.add_argument(
        "--text",
        type=str,
        default="Hello! I'm testing the text to speech latency. How are you doing today?",
        help="Custom text to synthesize"
    )

    args = parser.parse_args()

    # Default to comparison if no specific backend chosen
    if not args.backend and not args.compare:
        args.compare = True

    try:
        if args.compare:
            asyncio.run(compare_backends(args.text))
        else:
            async def run_single():
                metrics = await test_tts_backend(args.backend, args.text)
                metrics.report()
            asyncio.run(run_single())

    except KeyboardInterrupt:
        print("\n\n⏸️  Test interrupted by user.")
    except Exception as e:
        print(f"\n\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
