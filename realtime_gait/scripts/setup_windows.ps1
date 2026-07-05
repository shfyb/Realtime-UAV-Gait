param(
    [string]$TorchIndexUrl = "https://download.pytorch.org/whl/cu118"
)

$ErrorActionPreference = "Stop"

Write-Host "== realtime_gait Windows setup ==" -ForegroundColor Cyan

if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

Write-Host "Activating virtual environment..."
. .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip toolchain..."
python -m pip install --upgrade pip setuptools wheel

Write-Host "Installing PyTorch from: $TorchIndexUrl"
pip install torch torchvision torchaudio --index-url $TorchIndexUrl

Write-Host "Installing realtime_gait dependencies..."
pip install -r .\realtime_gait\requirements-windows.txt

Write-Host "Installing paddlepaddle-gpu (adjust command if needed)..."
pip install paddlepaddle-gpu

Write-Host "Setup finished." -ForegroundColor Green
