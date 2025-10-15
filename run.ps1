# DeepCSAT Flask App - Quick Start Script
# Run this script to start the Flask application

Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  🧠 DeepCSAT - Customer Satisfaction Prediction" -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path "venv")) {
    Write-Host "❌ Virtual environment not found!" -ForegroundColor Red
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
}

# Activate virtual environment
Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& .\venv\Scripts\Activate.ps1

# Check if dependencies are installed
Write-Host "Checking dependencies..." -ForegroundColor Yellow
$pipList = pip list
if ($pipList -notmatch "Flask") {
    Write-Host "Installing dependencies..." -ForegroundColor Yellow
    pip install -r requirements.txt
    Write-Host "✓ Dependencies installed" -ForegroundColor Green
}

# Check if model files exist
Write-Host "Checking model files..." -ForegroundColor Yellow
$modelFiles = @(
    "models/csat_model.keras",
    "models/scaler.pkl",
    "models/encoders.pkl",
    "models/feature_columns.json"
)

$allFilesExist = $true
foreach ($file in $modelFiles) {
    if (-not (Test-Path $file)) {
        Write-Host "❌ Missing: $file" -ForegroundColor Red
        $allFilesExist = $false
    }
}

if ($allFilesExist) {
    Write-Host "✓ All model files present" -ForegroundColor Green
} else {
    Write-Host "⚠️  Some model files are missing. The app may not work correctly." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "================================================" -ForegroundColor Cyan
Write-Host "  🚀 Starting Flask Application..." -ForegroundColor Cyan
Write-Host "================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "📡 Server will be available at: http://localhost:5000" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Run the Flask app
python app.py
