#!/bin/bash

echo "Actualizando listas de paquetes..."
apt-get update

echo "Instalando dependencias del sistema (Nano, FFmpeg, PortAudio)..."
apt-get install -y nano ffmpeg portaudio19-dev python3-dev build-essential

echo "Instalando dependencias de Python..."
pip install -r requirements.txt

echo "Generando script de entorno GPU..."
cat > start_server.sh << 'EOF'
#!/bin/bash
export LD_LIBRARY_PATH=`python3 -c 'import os; import nvidia.cublas.lib; import nvidia.cudnn.lib; print(os.path.dirname(nvidia.cublas.lib.__file__) + ":" + os.path.dirname(nvidia.cudnn.lib.__file__))'`:$LD_LIBRARY_PATH
echo "Entorno GPU configurado. LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
python3 server.py
EOF

chmod +x start_server.sh

echo "¡Instalación completada!"
echo "IMPORTANTE: Para iniciar el servidor con soporte GPU, usa este comando:"
echo "./start_server.sh"
