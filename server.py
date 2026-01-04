import asyncio
import threading
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Body
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.core.vad import VADManager
from src.core.asr import ASRManager
from src.core.llm import LLMManager
from src.core.tts import TTSManager
from src.core.orchestrator import Orchestrator
from src.audio.websocket_audio_manager import WebSocketAudioManager
from src.utils.config import Config

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models
    print("Loading models...")
    global vad, asr, llm, tts
    vad = VADManager()
    asr = ASRManager()
    llm = LLMManager()
    tts = TTSManager()
    print("Models loaded!")
    yield
    # Clean up
    print("Cleaning up resources...")
    # Add any cleanup logic here if needed

app = FastAPI(lifespan=lifespan)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default dev server
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def get():
    with open("src/web/static/index.html", "r") as f:
        return HTMLResponse(f.read())

class VoiceRequest(BaseModel):
    voice_id: str

@app.post("/api/set_voice")
async def set_voice(request: VoiceRequest):
    """
    Update the active TTS voice.
    """
    global tts
    if tts:
        tts.set_voice(request.voice_id)
        return JSONResponse(content={"status": "ok", "message": f"Voice set to {request.voice_id}"})
    return JSONResponse(content={"status": "error", "message": "TTS model not initialized"}, status_code=500)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")
    
    # Create audio manager for this connection
    audio_manager = WebSocketAudioManager(websocket)
    
    # Create orchestrator instance for this connection
    # We pass the global models but a unique audio manager
    orchestrator = Orchestrator(audio_manager, vad, asr, llm, tts)
    
    # Run orchestrator as an asyncio Task (non-blocking)
    orchestrator_task = asyncio.create_task(orchestrator.run())
    
    try:
        while True:
            # Receive audio data from client
            data = await websocket.receive_bytes()
            audio_manager.add_input_audio(data)
            
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        audio_manager.stop()
        orchestrator.stop()
        orchestrator_task.cancel()
    except Exception as e:
        print(f"Error: {e}")
        audio_manager.stop()
        orchestrator.stop()
        orchestrator_task.cancel()

if __name__ == "__main__":
    import uvicorn
    # Listen on 0.0.0.0 to be accessible externally (e.g. from pod)
    uvicorn.run(app, host="0.0.0.0", port=8000)
