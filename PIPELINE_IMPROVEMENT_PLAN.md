# Plan de Mejora del Pipeline Verba AI

Este documento detalla las mejoras arquitectónicas propuestas para reducir la latencia y mejorar la robustez del sistema de conversación en tiempo real.

## 1. Estado Actual (Problemas Identificados)

*   **Bloqueo Síncrono**: El `Orchestrator` opera en un bucle síncrono. La escucha se detiene mientras se procesa la respuesta.
*   **Latencia TTS**: Se espera a signos de puntuación fuertes (`.!?`) para generar audio, y se usa `ffmpeg` (proceso externo) para conversión, lo que añade overhead.
*   **Gestión de Interrupciones**: La detección de interrupciones ("Barge-in") depende de callbacks durante la reproducción, lo que puede no ser instantáneo.
*   **ASR Latency**: Se transcribe todo el buffer de audio al final del turno.

## 2. Mejoras Implementadas (Fase 1 - Inmediata)

*   **ASR**: Reducción de `beam_size` de 5 a 1 en `FasterWhisper` para inferencia más rápida.
*   **Config**: Reducción de `MIN_SILENCE_DURATION_MS` de 500ms a 350ms para turnos más rápidos.
*   **TTS**: Mejora en la segmentación de texto para procesar frases largas divididas por comas (`,;:`), reduciendo el Time-To-First-Audio (TTFA).

## 3. Propuesta de Arquitectura Asíncrona (Fase 2)

El objetivo es migrar `src/core/orchestrator.py` a un modelo completamente asíncrono usando `asyncio`.

### A. Bucle de Eventos Asíncrono
En lugar de `while True` bloqueante:
```python
async def run(self):
    input_task = asyncio.create_task(self.process_input())
    output_task = asyncio.create_task(self.process_output())
    await asyncio.gather(input_task, output_task)
```

### B. Gestión de Estados (State Machine)
Implementar una máquina de estados explícita:
*   `LISTENING`: Capturando audio y analizando VAD.
*   `PROCESSING`: Usuario terminó de hablar, generando respuesta.
*   `SPEAKING`: Reproduciendo audio.
*   `INTERRUPTED`: Estado transitorio para limpiar colas.

### C. Streaming Real
*   **ASR**: Si es posible, alimentar el audio a Whisper en chunks más grandes mientras el usuario habla (speculative decoding) o usar un modelo de streaming real.
*   **TTS**: Mantener una conexión WebSocket abierta con el servicio TTS o cargar el modelo local en memoria para evitar overhead de inicialización.

## 4. Mejoras de Infraestructura

*   **Audio Codec**: Eliminar la dependencia de `ffmpeg` CLI. Usar decodificación directa en Python (`miniaudio`, `av`, o `pydub` optimizado) para convertir el MP3 de EdgeTTS a PCM.
*   **WebSockets**: Si el cliente final es web, mover la lógica de audio al navegador y enviar audio comprimido (Opus) por WebSocket para reducir ancho de banda y latencia.

## 5. Próximos Pasos Recomendados

1.  **Refactorizar Orchestrator**: Convertir `run` y `handle_turn` a `async def`.
2.  **Optimizar Audio Playback**: Implementar un reproductor de audio asíncrono que no bloquee el hilo principal.
3.  **Benchmarking**: Medir latencia "end-to-end" con logs precisos.
