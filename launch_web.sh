#!/bin/bash

# CANShield Web Interface Launcher
# Quick start script for the web dashboard

echo "🛡️  CANShield Web Interface Launcher"
echo "======================================"
echo ""

# Check if virtual environment exists
if [ -d "canshield_env" ]; then
    echo "✅ Virtual environment found"
    source canshield_env/bin/activate
else
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv canshield_env
    source canshield_env/bin/activate
    echo "✅ Virtual environment created"
fi

# Check if web requirements are installed
if python -c "import streamlit" 2>/dev/null; then
    echo "✅ Web dependencies found"
else
    echo "📦 Installing web dependencies..."
    pip install -r requirements_web.txt
    echo "✅ Dependencies installed"
fi

echo ""
echo "🚀 Launching CANShield Dashboard..."
echo ""
echo "📍 Opening browser at: http://localhost:8501"
echo ""
echo "💡 Tips:"
echo "   - Use sidebar to configure settings"
echo "   - Upload datasets or select existing ones"
echo "   - Click 'Load Model' then 'Run Detection'"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Launch Streamlit app
streamlit run app.py

echo ""
echo "👋 CANShield dashboard closed"

