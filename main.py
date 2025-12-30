import os
import sys

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.audio.stream_manager import AudioStreamManager
from src.core.vad import VADManager
from src.core.asr import ASRManager
from src.core.llm import LLMManager
from src.core.tts import TTSManager
from src.core.orchestrator import Orchestrator

def main():
    print("Initializing Sesame-inspired Voice Assistant...")
    
    # Initialize modules
    try:
        audio = AudioStreamManager()
        vad = VADManager()
        asr = ASRManager()
        llm = LLMManager()
        tts = TTSManager()
        
        orchestrator = Orchestrator(audio, vad, asr, llm, tts)
        
        orchestrator.run()
        
    except Exception as e:
        print(f"Error initializing system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
