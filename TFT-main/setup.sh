#!/bin/bash

# TFT Stock Prediction System Setup Script
# This script sets up the complete environment for the TFT system

set -e  # Exit on any error

echo "🚀 Setting up TFT Stock Prediction System..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
print_status "Checking Python version..."
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    print_error "Python 3.8+ is required. Found: $python_version"
    exit 1
fi
print_success "Python version OK: $python_version"

# Create virtual environment
print_status "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_warning "Virtual environment already exists"
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch first (with CUDA support if available)
print_status "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    print_status "NVIDIA GPU detected, installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    print_status "No NVIDIA GPU detected, installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
print_status "Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
print_status "Creating directory structure..."
directories=(
    "data"
    "data/cache"
    "models"
    "logs"
    "predictions"
    "reports"
    "output"
    "tests"
)

for dir in "${directories[@]}"; do
    if [ ! -d "$dir" ]; then
        mkdir -p "$dir"
        print_success "Created directory: $dir"
    else
        print_warning "Directory already exists: $dir"
    fi
done

# Set up configuration files
print_status "Setting up configuration files..."

# Create default config
cat > config/default_config.json << EOF
{
  "data": {
    "db_path": "data/stock_data.db",
    "cache_dir": "data/cache",
    "symbols": null,
    "start_date": "2020-01-01"
  },
  "model": {
    "max_encoder_length": 63,
    "max_prediction_length": 5,
    "batch_size": 64,
    "learning_rate": 0.001,
    "hidden_size": 64,
    "lstm_layers": 2,
    "attention_head_size": 4,
    "dropout": 0.2,
    "max_epochs": 100,
    "quantiles": [0.1, 0.5, 0.9]
  },
  "trading": {
    "liquidity_threshold": 500,
    "confidence_threshold": 0.1,
    "max_positions": 20,
    "max_position_size": 0.05,
    "sector_limit": 0.3,
    "turnover_limit": 0.5
  },
  "scheduler": {
    "training_schedule": {
      "frequency": "weekly",
      "day": "sunday",
      "time": "02:00"
    },
    "prediction_schedule": {
      "frequency": "daily",
      "time": "06:00"
    },
    "data_update_schedule": {
      "frequency": "daily",
      "time": "18:00"
    }
  },
  "api": {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 1
  }
}
EOF

mkdir -p config
print_success "Created default configuration file"

# Create environment file
cat > .env << EOF
# TFT Stock Prediction Environment Variables

# Data
TFT_DB_PATH=data/stock_data.db
TFT_CACHE_DIR=data/cache

# Model
TFT_MODEL_PATH=models/tft_model.pth
TFT_PREPROCESSOR_PATH=models/preprocessor.pkl

# API
TFT_API_HOST=0.0.0.0
TFT_API_PORT=8000

# Logging
TFT_LOG_LEVEL=INFO
TFT_LOG_DIR=logs

# GPU
TFT_GPU_ENABLED=true

# External APIs (add your keys here)
ALPHA_VANTAGE_API_KEY=your_api_key_here
FINNHUB_API_KEY=your_api_key_here
NEWS_API_KEY=your_api_key_here
EOF

print_success "Created environment configuration file"

# Create systemd service file (optional)
print_status "Creating systemd service file..."
cat > tft-api.service << EOF
[Unit]
Description=TFT Stock Prediction API
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment=PATH=$(pwd)/venv/bin
ExecStart=$(pwd)/venv/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

print_success "Created systemd service file (tft-api.service)"

# Test installation
print_status "Testing installation..."

# Test basic imports
python3 -c "
import sys
sys.path.append('.')

try:
    import pandas as pd
    import numpy as np
    import torch
    print('✅ Core dependencies imported successfully')
    
    # Test PyTorch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'✅ PyTorch device: {device}')
    
    # Test data preprocessing
    from data_preprocessing import StockDataPreprocessor
    print('✅ Data preprocessing module loaded')
    
    # Test model module
    from tft_model import EnhancedTFTModel
    print('✅ TFT model module loaded')
    
    # Test ranking system
    from stock_ranking import StockRankingSystem
    print('✅ Stock ranking module loaded')
    
    print('🎉 All modules loaded successfully!')
    
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    print_success "Installation test passed!"
else
    print_error "Installation test failed!"
    exit 1
fi

# Create quick start script
print_status "Creating quick start scripts..."

cat > run_training.sh << 'EOF'
#!/bin/bash
# Quick training script

echo "🎯 Starting TFT training with sample data..."

source venv/bin/activate

python train.py \
    --data-source api \
    --symbols AAPL GOOGL MSFT TSLA AMZN NVDA META NFLX \
    --start-date 2022-01-01 \
    --target-type returns \
    --max-epochs 20 \
    --batch-size 32 \
    --generate-predictions \
    --output-dir output/training_$(date +%Y%m%d_%H%M%S)

echo "✅ Training completed! Check the output directory for results."
EOF

cat > run_predictions.sh << 'EOF'
#!/bin/bash
# Quick prediction script

echo "🔮 Generating predictions..."

source venv/bin/activate

if [ ! -f "models/tft_model.pth" ]; then
    echo "❌ No trained model found. Please run training first with: ./run_training.sh"
    exit 1
fi

python predict.py \
    --model-path models/tft_model.pth \
    --data-source api \
    --symbols AAPL GOOGL MSFT TSLA AMZN NVDA META NFLX \
    --prediction-method quintile \
    --include-portfolio \
    --output-format both \
    --output-dir predictions/$(date +%Y%m%d_%H%M%S)

echo "✅ Predictions completed! Check the predictions directory for results."
EOF

cat > run_api.sh << 'EOF'
#!/bin/bash
# Quick API server script

echo "🚀 Starting TFT API server..."

source venv/bin/activate

if [ ! -f "models/tft_model.pth" ]; then
    echo "⚠️  No trained model found. API will start but training will be required."
fi

python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
EOF

chmod +x run_training.sh run_predictions.sh run_api.sh
print_success "Created quick start scripts"

# Create sample data collection script
cat > collect_sample_data.sh << 'EOF'
#!/bin/bash
# Collect sample data

echo "📊 Collecting sample stock data..."

source venv/bin/activate

python -c "
from data_pipeline import DataPipeline

pipeline = DataPipeline()
symbols = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN', 'NVDA', 'META', 'NFLX']

print('Collecting historical data...')
results = pipeline.run_historical_backfill(
    symbols=symbols,
    start_date='2022-01-01'
)

print(f'Data collection completed: {results}')

# Verify data
df = pipeline.collector.load_data_from_db(symbols=symbols)
print(f'Data loaded: {df.shape} records')
print(f'Date range: {df[\"timestamp\"].min()} to {df[\"timestamp\"].max()}')
print(f'Symbols: {df[\"symbol\"].unique()}')
"

echo "✅ Sample data collection completed!"
EOF

chmod +x collect_sample_data.sh
print_success "Created sample data collection script"

# Print setup summary
print_success "🎉 TFT Stock Prediction System setup completed!"

echo ""
echo "📋 Setup Summary:"
echo "  ✅ Virtual environment created and activated"
echo "  ✅ Python dependencies installed"
echo "  ✅ Directory structure created"
echo "  ✅ Configuration files generated"
echo "  ✅ Quick start scripts created"
echo "  ✅ Installation tested"

echo ""
echo "🚀 Quick Start:"
echo "  1. Collect sample data:    ./collect_sample_data.sh"
echo "  2. Train the model:        ./run_training.sh"
echo "  3. Generate predictions:   ./run_predictions.sh"
echo "  4. Start API server:       ./run_api.sh"

echo ""
echo "📚 Manual Usage:"
echo "  • Activate environment:    source venv/bin/activate"
echo "  • Train model:            python train.py --help"
echo "  • Generate predictions:   python predict.py --help"
echo "  • Run scheduler:          python scheduler.py --help"
echo "  • Start API:              uvicorn api:app --reload"

echo ""
echo "⚙️  Configuration:"
echo "  • Main config:            config/default_config.json"
echo "  • Environment vars:       .env"
echo "  • Systemd service:        tft-api.service"

echo ""
echo "📂 Important Directories:"
echo "  • Data:                   data/"
echo "  • Models:                 models/"
echo "  • Predictions:            predictions/"
echo "  • Logs:                   logs/"
echo "  • Reports:                reports/"

echo ""
print_warning "Next Steps:"
echo "  1. Edit .env file to add your API keys if using external data sources"
echo "  2. Review and modify config/default_config.json as needed"
echo "  3. Run ./collect_sample_data.sh to get started with sample data"

echo ""
print_success "Happy trading! 📈"
