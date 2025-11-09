#!/bin/bash
# run_streamlit.sh - Launch the CMAR Streamlit frontend

echo "🏥 Starting CMAR Streamlit Application..."
echo "================================================"
echo ""

# Activate virtual environment if it exists
if [ -d "graph" ]; then
    echo "📦 Activating virtual environment..."
    source graph/bin/activate
fi

# Check if streamlit is installed
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "🚀 Launching Streamlit app..."
echo "   -> Open your browser to: http://localhost:8501"
echo "   -> Press Ctrl+C to stop the server"
echo ""

# Run streamlit
streamlit run streamlit_app.py
