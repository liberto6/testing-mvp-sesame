import asyncio
import threading
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from src.core.vad import VADManager
from src.core.asr import ASRManager
from src.core.llm import LLMManager
from src.core.tts import TTSManager
from src.core.orchestrator import Orchestrator
from src.audio.websocket_audio_manager import WebSocketAudioManager
from src.utils.config import Config

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="src/web/static"), name="static")

# Initialize models globally to avoid reloading them per connection
print("Loading models...")
vad = VADManager()
asr = ASRManager()
llm = LLMManager()
tts = TTSManager()
print("Models loaded!")

@app.get("/")
async def get():
    with open("src/web/static/index.html", "r") as f:
        return HTMLResponse(f.read())

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")
    
    # Create audio manager for this connection
    audio_manager = WebSocketAudioManager(websocket)
    
    # Create orchestrator instance for this connection
    # We pass the global models but a unique audio manager
    orchestrator = Orchestrator(audio_manager, vad, asr, llm, tts)
    
    # Run orchestrator in a separate thread because it has a blocking loop
    orchestrator_thread = threading.Thread(target=orchestrator.run)
    orchestrator_thread.daemon = True
    orchestrator_thread.start()
    
    try:
        while True:
            # Receive audio data from client
            data = await websocket.receive_bytes()
            audio_manager.add_input_audio(data)
            
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        audio_manager.stop()
        orchestrator.stop()
    except Exception as e:
        print(f"Error: {e}")
        audio_manager.stop()
        orchestrator.stop()

if __name__ == "__main__":
    import uvicorn
    # Listen on 0.0.0.0 to be accessible externally (e.g. from pod)
    uvicorn.run(app, host="0.0.0.0", port=8000)
