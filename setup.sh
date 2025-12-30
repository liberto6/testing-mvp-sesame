#!/bin/bash

echo "Actualizando listas de paquetes..."
apt-get update

echo "Instalando dependencias del sistema (Nano, FFmpeg, PortAudio)..."
apt-get install -y nano ffmpeg portaudio19-dev python3-dev build-essential

echo "Instalando dependencias de Python..."
pip install -r requirements.txt

echo "¡Instalación completada!"
