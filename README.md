# Sesame-Inspired Voice Assistant

Este proyecto implementa un asistente de voz en tiempo real inspirado en los repositorios de SesameAILabs, priorizando baja latencia, interrupciones (barge-in) y procesamiento local.

## Estructura del Proyecto

El sistema es modular y está compuesto por:

- **Audio Manager (`src/audio`)**: Gestiona la entrada (micrófono) y salida (altavoces) usando `PyAudio`. Implementa buffers circulares para baja latencia.
- **VAD Manager (`src/core/vad.py`)**: Usa **Silero VAD** para detectar voz humana con alta precisión y filtrar ruido.
- **ASR Manager (`src/core/asr.py`)**: Usa **Faster Whisper** para transcripción rápida y eficiente en local.
- **Orchestrator (`src/core/orchestrator.py`)**: El cerebro del sistema. Gestiona el bucle de conversación, detecta interrupciones durante el habla del asistente y coordina los turnos.
- **LLM & TTS (`src/core/llm.py`, `src/core/tts.py`)**: Interfaces modulares para generación de texto y síntesis de voz.

## Requisitos

- Python 3.8+
- PyAudio (requiere PortAudio)
- Torch

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

```bash
python main.py
```

## Características Clave

1.  **Cancelación de Eco / Barge-in**: El sistema escucha activamente mientras habla. Si detecta voz del usuario (VAD), interrumpe inmediatamente la respuesta (TTS) y comienza a escuchar.
2.  **Baja Latencia**: Uso de `faster-whisper` y buffers optimizados.
3.  **Modularidad**: Fácil de sustituir el LLM por OpenAI/LocalLLM o el TTS por XTTS/Parler-TTS.
