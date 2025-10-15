#!/bin/bash
# DeepCSAT Flask App - Quick Start Script (Unix/Linux/Mac)

echo "================================================"
echo "  🧠 DeepCSAT - Customer Satisfaction Prediction"
echo "================================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
echo "Checking dependencies..."
if ! pip list | grep -q "Flask"; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    echo "✓ Dependencies installed"
fi

# Check if model files exist
echo "Checking model files..."
MODEL_FILES=(
    "models/csat_model.keras"
    "models/scaler.pkl"
    "models/encoders.pkl"
    "models/feature_columns.json"
)

ALL_FILES_EXIST=true
for file in "${MODEL_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing: $file"
        ALL_FILES_EXIST=false
    fi
done

if [ "$ALL_FILES_EXIST" = true ]; then
    echo "✓ All model files present"
else
    echo "⚠️  Some model files are missing. The app may not work correctly."
fi

echo ""
echo "================================================"
echo "  🚀 Starting Flask Application..."
echo "================================================"
echo ""
echo "📡 Server will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

# Run the Flask app
python app.py
