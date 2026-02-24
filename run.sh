#!/bin/bash
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "🚀 First run — creating sandbox and installing everything (1-2 min)..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "✅ Ready! Next time will be instant."
else
    source venv/bin/activate
fi

echo "🌟 Starting CamOCR web UI..."
echo "Open in browser → http://127.0.0.1:7860"
python app.py
