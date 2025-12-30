# Plan de Mejora del Pipeline (Verba)

Este documento detalla los pasos para optimizar la lógica y reducir la latencia del pipeline de conversación.

## Fase 1: Optimización de Configuración y Modelos (Completado)
- [x] Ajustar parámetros de VAD (silence duration).
- [x] Reducir `beam_size` de Whisper a 1.
- [x] Seleccionar modelo Whisper adecuado (`base.en`).

## Fase 2: Refactorización Asíncrona (Completado)
- [x] **Orchestrator Async**: Migrar el bucle principal de `threading` a `asyncio`.
- [x] **Streaming Real**: Asegurar que LLM y TTS procesen en streaming sin bloquear.
- [x] **Gestión de Estados**: Implementar máquina de estados (LISTENING, PROCESSING, SPEAKING) para evitar condiciones de carrera.
- [x] **Asyncio Queue**: Reemplazar colas síncronas por `asyncio.Queue`.
- [x] **Non-blocking LLM/TTS**: Uso de `AsyncGroq` y ejecución de ffmpeg en executor.

## Fase 3: Mejoras de UX y Estabilidad (En Progreso/Pendiente)
- [x] **Interrupción Instantánea**: Implementar mensaje `CLEAR_BUFFER` al frontend.
- [ ] **Reconexión WebSocket**: Manejo robusto de desconexiones en frontend.
- [ ] **Feedback Visual**: Mostrar estado (Escuchando/Pensando/Hablando) en UI.

## Notas Técnicas
- Se ha eliminado el uso de `threading` para evitar conflictos con el event loop de asyncio.
- Se utiliza `edge-tts` para síntesis rápida y natural.
- Se mantiene el formato de audio `int16` para transporte eficiente.
