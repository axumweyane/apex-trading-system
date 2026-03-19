# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

APEX is a Temporal Fusion Transformer (TFT) stock prediction and trading platform. The main codebase lives in `TFT-main/`. It supports two data backends: a legacy SQLite/file-based pipeline and a recommended PostgreSQL pipeline. There is also a microservices deployment layer with Kafka-based event streaming.

## Build & Run Commands

All commands run from `TFT-main/`.

### Initial Setup
```bash
./setup.sh                    # Full setup: venv, deps, dirs, config
pip install -r requirements.txt  # Manual dependency install
python postgres_schema.py     # Create PostgreSQL schema
cp .env.example .env          # Create env config (edit with real credentials)
```

### Training
```bash
# PostgreSQL-based (recommended)
python train_postgres.py --symbols AAPL GOOGL MSFT --start-date 2022-01-01 --target-type returns --max-epochs 50

# Legacy file-based
python train.py --data-source api --symbols AAPL GOOGL MSFT --start-date 2020-01-01 --target-type returns --max-epochs 50
```

### Predictions
```bash
python predict.py --model-path models/tft_model.pth --symbols AAPL GOOGL MSFT --prediction-method quintile --include-portfolio
```

### API Server
```bash
# PostgreSQL API
python -m uvicorn api_postgres:app --host 0.0.0.0 --port 8000 --reload

# Legacy API
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

### Scheduler
```bash
python scheduler.py --mode scheduler       # Automated scheduled runs
python scheduler.py --mode manual --task training  # Manual single task
```

### Docker (microservices)
```bash
docker-compose up              # All services
docker-compose up data-ingestion sentiment-engine  # Specific services
```

### Tests
```bash
pytest                                    # All tests
python test_end_to_end.py                # End-to-end pipeline test
python test_polygon_integration.py       # Polygon.io integration test
python test_service.py                   # Service-level test
./devtools/prompt_runner.sh --test-all   # Copilot prompt validation
```

### Linting
```bash
black .          # Format
flake8 .         # Lint
mypy .           # Type check
```

## Architecture

### Two Parallel Pipelines

The system has a **legacy pipeline** (SQLite/file-based) and a **PostgreSQL pipeline**. Each has its own set of modules:

| Layer | Legacy | PostgreSQL |
|-------|--------|------------|
| Data loading | `data_pipeline.py` (StockDataCollector → SQLite) | `postgres_data_loader.py` (PostgresDataLoader → psycopg2) |
| Preprocessing | `data_preprocessing.py` (StockDataPreprocessor) | `postgres_data_pipeline.py` (PostgresDataPipeline) |
| Model | `tft_model.py` (EnhancedTFTModel) | `tft_postgres_model.py` (TFTPostgresModel + AdvancedOptionsModel) |
| Training | `train.py` | `train_postgres.py` |
| API | `api.py` | `api_postgres.py` |

Both pipelines share: `stock_ranking.py` (signal generation + portfolio construction), `scheduler.py`, `config_manager.py`, `predict.py`.

### Core Flow
1. **Data ingestion** — Polygon.io OHLCV, Reddit sentiment, fundamentals → database
2. **Preprocessing** — Technical indicators (RSI, MACD, Bollinger Bands), temporal features, normalization
3. **TFT model** — PyTorch Forecasting `TemporalFusionTransformer` with quantile loss, attention, multi-horizon forecasting
4. **Signal generation** — `StockRankingSystem` ranks predictions → `PortfolioConstructor` builds long/short portfolios with risk constraints
5. **API serving** — FastAPI endpoints for `/predict`, `/train`, `/health`

### Microservices Layer (`microservices/`)

Five FastAPI services coordinated via Kafka topics and Redis caching:

| Service | Port | Role |
|---------|------|------|
| `data-ingestion` | 8001 | Polygon.io + Reddit data → Kafka |
| `sentiment-engine` | 8002 | NLP sentiment scoring (FinBERT/VADER) |
| `tft-predictor` | 8003 | GPU inference, MLflow model versioning |
| `trading-engine` | 8004 | Alpaca paper/live order execution |
| `orchestrator` | 8005 | Saga-pattern workflow coordination |

Infrastructure: TimescaleDB (PostgreSQL 15), Redis, Kafka, Prometheus, Grafana, MLflow. See `docker-compose.yml`.

### Key Kafka Topics
`market-data`, `sentiment-scores`, `tft-predictions`, `trading-signals`, `order-updates`, `portfolio-updates`, `system-events`

### Configuration

- Environment: `.env` (copy from `.env.template` for full config, `.env.example` for minimal)
- Model/trading params: `config_manager.py` with `TFTConfig` and `TradingConfig` dataclasses
- JSON config: `config/default_config.json` (created by `setup.sh`)

### Key External Dependencies
- **Polygon.io** — Primary market data API (OHLCV, news, fundamentals, options). Rate limited: 5 req/min on free tier
- **Alpaca** — Paper/live trading execution
- **Reddit (PRAW)** — Sentiment data from financial subreddits
- **pytorch-forecasting** — TFT model implementation
- **MLflow** — Experiment tracking and model registry

### Data Directories
`data/` (raw data + SQLite), `models/` (trained `.pth` files + preprocessors), `predictions/`, `logs/`, `reports/`, `output/`

## Notes

- The `docker-compose.yml` has duplicate service definitions for `tft-predictor`, `trading-engine`, and `orchestrator` — the second definitions use simplified env vars. This needs cleanup.
- The docker-compose references `tft_network` as an external network — create it with `docker network create tft_network` before running.
- PostgreSQL env vars differ between `.env.template` (`DB_HOST`, `DB_PORT`, etc.) and `train_postgres.py` (`POSTGRES_HOST`, `POSTGRES_PORT`, etc.). Check which convention the target module expects.
- The `tft_postgres_model.py` contains an `AdvancedOptionsModel` class with Black-Scholes pricing that is separate from the TFT model itself.
