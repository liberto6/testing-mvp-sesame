#!/bin/bash

# ==========================================
# Verba Backend - RunPod Startup Script
# ==========================================

set -e  # Exit on error

echo "🚀 Starting Verba Backend Setup on RunPod..."
echo ""

# ==========================================
# 1. INSTALL SYSTEM DEPENDENCIES
# ==========================================
echo "📦 Installing system dependencies..."
apt-get update -qq
apt-get install -y -qq portaudio19-dev ffmpeg curl git > /dev/null 2>&1
echo "✅ System dependencies installed"

# ==========================================
# 2. INSTALL PYTHON DEPENDENCIES
# ==========================================
echo "🐍 Installing Python dependencies..."
if [ ! -d "/root/.cache/pip" ]; then
    echo "   (First time installation, this may take a few minutes...)"
fi

pip install -q -r requirements.txt
echo "✅ Python dependencies installed"

# ==========================================
# 3. CHECK .ENV FILE
# ==========================================
echo ""
echo "⚙️  Checking configuration..."

if [ ! -f ".env" ]; then
    echo "⚠️  WARNING: .env file not found!"
    echo "   Creating .env template..."
    cat > .env << 'EOF'
GROQ_API_KEY=your_groq_api_key_here
WHISPER_MODEL=base
TTS_VOICE=en-US-Neural2-C
HOST=0.0.0.0
PORT=8000
EOF
    echo ""
    echo "❌ IMPORTANT: You must set your GROQ_API_KEY in .env"
    echo "   Edit with: vi .env"
    echo "   Or run: echo 'GROQ_API_KEY=gsk_your_key' > .env"
    echo ""
    exit 1
fi

# Check if API key is set
if grep -q "your_groq_api_key_here" .env; then
    echo "❌ ERROR: GROQ_API_KEY not configured in .env"
    echo "   Please set your API key:"
    echo "   echo 'GROQ_API_KEY=gsk_your_key' > .env"
    echo ""
    exit 1
fi

echo "✅ Configuration file found"

# ==========================================
# 4. DISPLAY INFO
# ==========================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 Setup Complete! Ready to start backend"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Configuration:"
echo "   • Python: $(python --version)"
echo "   • Working directory: $(pwd)"
echo "   • Port: 8000"
echo ""
echo "🚀 To start the backend server, run:"
echo ""
echo "   python server.py"
echo ""
echo "💡 Or run this script with --start flag:"
echo ""
echo "   ./startup.sh --start"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ==========================================
# 5. AUTO-START IF REQUESTED
# ==========================================
if [ "$1" == "--start" ] || [ "$1" == "-s" ]; then
    echo "🚀 Starting backend server..."
    echo ""
    python server.py
fi
