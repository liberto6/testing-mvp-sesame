#!/bin/bash

echo "Actualizando listas de paquetes..."
apt-get update

echo "Instalando dependencias del sistema (Nano, FFmpeg, PortAudio)..."
apt-get install -y nano ffmpeg portaudio19-dev python3-dev build-essential

echo "Instalando dependencias de Python..."
pip install -r requirements.txt

echo "Configurando entorno CUDA para Faster-Whisper..."
# Add nvidia libs to LD_LIBRARY_PATH automatically
export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`:$LD_LIBRARY_PATH

echo "¡Instalación completada!"
echo "IMPORTANTE: Antes de ejecutar el servidor, ejecuta este comando manualmente:"
echo "export LD_LIBRARY_PATH=\`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + \":\" + os.path.dirname(nvidia.cudnn.lib.__file__))'\`:\$LD_LIBRARY_PATH"
