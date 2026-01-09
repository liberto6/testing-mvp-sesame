import os
import sys
import asyncio

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.audio.stream_manager import AudioStreamManager
from src.core.vad import VADManager
from src.core.asr import ASRManager
from src.core.llm import LLMManager
from src.core.tts import TTSManager
from src.core.orchestrator import Orchestrator

async def main():
    print("Initializing Sesame-inspired Voice Assistant...")
    
    # Initialize modules
    try:
        audio = AudioStreamManager()
        vad = VADManager()
        asr = ASRManager()
        llm = LLMManager()
        tts = TTSManager()
        
        orchestrator = Orchestrator(audio, vad, asr, llm, tts)
        await orchestrator.run()

    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Cleanup done.")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
