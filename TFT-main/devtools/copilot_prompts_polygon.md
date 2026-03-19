# 🚀 GitHub Copilot Validation Prompts for Polygon.io Integration

## Overview

This document contains optimized prompts for GitHub Copilot to generate high-quality, production-ready code for Polygon.io API integration in the TFT Stock Prediction System.

## Core Implementation Prompts

### 1. Data Pipeline Optimization

**File**: `polygon_data_loader.py`

```python
# Implement batch OHLCV fetcher for 100+ symbols with Polygon.io
# Features: auto-throttling (5 req/min), retry on 429, and PostgreSQL caching
# Return: Dict of DataFrames with {symbol: df}
# Handle Polygon's aggregates endpoint with proper date formatting
# Include vwap and trade_count fields from Polygon response
```

**Expected Output**: Batch processing class with connection pooling and intelligent caching.

### 2. Advanced Feature Engineering

**File**: `data_preprocessing.py`

```python
# Calculate technical indicators from Polygon's vwap:
# - Volume-weighted RSI(14) using vwap instead of close
# - MACD(12,26,9) using vwap for signal line calculation
# - Bollinger %B(20,2) with vwap-based standard deviation
# Handle corporate actions via Polygon's 'adjusted' flag
# Ensure all indicators are properly normalized for TFT model input
```

**Expected Output**: Enhanced technical indicator functions with VWAP integration.

### 3. Real-Time Prediction Endpoint

**File**: `api_postgres.py`

```python
# Create FastAPI endpoint: /polygon/realtime-predict
# Input: List of Polygon-formatted symbols (e.g., 'O:SPY230818C00325000')
# Output: Predictions with Polygon's native symbol format
# Use Polygon's WebSocket client for live data streaming
# Include options chain data for derivatives predictions
# Handle equity, options, and forex symbol formats
```

**Expected Output**: WebSocket-enabled FastAPI endpoint with real-time data processing.

### 4. Intelligent Rate Limit Handling

**File**: `polygon_data_loader.py`

```python
# Implement decorator for Polygon API calls:
# - Exponential backoff on 429 errors (1s, 2s, 4s, 8s)
# - Automatic 60s cooldown after 5 consecutive 429s
# - Fallback to cached PostgreSQL data when API unavailable
# - Track API usage and warn at 80% of daily limit
# - Support both free (5 req/min) and paid (unlimited) tier limits
```

**Expected Output**: Robust rate limiting decorator with intelligent fallback mechanisms.

### 5. News Sentiment Integration

**File**: `enhanced_data_pipeline.py`

```python
# Process Polygon news into trading features:
# 1. Calculate sentiment polarity score per article using TextBlob
# 2. Compute daily sentiment momentum (3-day rolling change)
# 3. Merge with OHLCV using Polygon's news timestamp alignment
# 4. Weight sentiment by article publisher authority score
# 5. Generate sentiment volatility metrics for risk assessment
```

**Expected Output**: Comprehensive news sentiment processing pipeline with temporal alignment.

### 6. Fundamental Data Processing

**File**: `polygon_data_loader.py`

```python
# Fetch and normalize Polygon fundamental data:
# - Map 'marketCap' to size quantiles (micro, small, mid, large, mega)
# - Convert 'peRatio' to z-scores within sector groups
# - Handle missing values with sector averages and forward-fill
# - Create fundamental momentum features (QoQ growth rates)
# - Integrate with earnings calendar for event-driven features
```

**Expected Output**: Normalized fundamental data processor with sector-aware statistics.

### 7. WebSocket Real-Time Integration

**File**: `realtime_handler.py`

```python
# Implement Polygon WebSocket client that:
# 1. Subscribes to specified symbols via Polygon's streaming API
# 2. Updates PostgreSQL every 15 seconds with batched inserts
# 3. Triggers predictions on volume spikes (3x average volume)
# 4. Handles WebSocket reconnection and error recovery
# 5. Processes both trade and quote data streams
# 6. Maintains connection health monitoring
```

**Expected Output**: Production-ready WebSocket client with automatic reconnection and health monitoring.

### 8. Automated Batch Processing

**File**: `scheduler.py`

```python
# Create daily job to:
# 1. Fetch all S&P 500 symbols from Polygon reference API
# 2. Update OHLCV data in parallel threads (10 concurrent)
# 3. Validate corporate action adjustments using Polygon's splits/dividends API
# 4. Run data quality checks (missing data, outliers, volume anomalies)
# 5. Generate daily data completeness report
```

**Expected Output**: Robust batch processing system with parallel execution and validation.

### 9. Production Error Handling

**File**: `polygon_data_loader.py`

```python
# Implement fault tolerance for:
# - Polygon API downtime (switch to cached PostgreSQL data)
# - Symbol delistings (auto-purge from active symbol list)
# - Data gaps (linear interpolation with confidence intervals)
# - Network timeouts (retry with increasing delays)
# - Invalid symbol formats (log and skip gracefully)
# - Rate limit exceeded (intelligent queuing system)
```

**Expected Output**: Comprehensive error handling with graceful degradation.

### 10. Model Feature Optimization

**File**: `tft_postgres_model.py`

```python
# Add Polygon-specific features to TFT:
# - vwap_ratio: vwap relative to close price
# - news_sentiment_momentum: 3-day sentiment change
# - fundamental_zscore: sector-adjusted fundamental metrics
# - volume_profile: intraday volume distribution
# Quantize model weights for faster Polygon real-time predictions
# Implement feature importance tracking for Polygon-specific inputs
```

**Expected Output**: Enhanced TFT model with Polygon-optimized features and performance optimization.

## 🔥 Critical Options Trading Validation

**File**: `api_postgres.py`

```python
# Create endpoint that:
# 1. Accepts Polygon options symbols (e.g., 'O:SPY230818C00325000')
# 2. Fetches underlying equity data and option chain
# 3. Generates volatility-adjusted predictions using Black-Scholes
# 4. Returns predictions in Polygon's options response format
# 5. Includes Greeks calculation (delta, gamma, theta, vega)
# 6. Handles American vs European style options
```

**Expected Output**: Advanced options trading endpoint with full derivatives support.

## 🧠 Copilot Integration Standards

### Symbol Format Recognition
- **Equities**: `C:AAPL` (common stock)
- **Options**: `O:TSLA240118C00500000` (TSLA call option)
- **Forex**: `C:EURUSD` (currency pair)
- **Crypto**: `X:BTCUSD` (cryptocurrency)
- **Indices**: `I:SPX` (index data)

### API Response Handling
- Always check `status` field in Polygon responses
- Handle `results` array properly (may be empty)
- Use `adjusted` parameter for split/dividend adjustments
- Implement proper timestamp conversion (milliseconds to datetime)

### Rate Limiting Best Practices
- Free tier: 5 calls per minute
- Basic tier: 100 calls per minute
- Advanced tier: Unlimited with fair use
- Always implement exponential backoff
- Cache responses to minimize API calls

### Error Code Handling
- `200`: Success
- `400`: Bad request (invalid parameters)
- `401`: Unauthorized (invalid API key)
- `429`: Rate limit exceeded
- `500`: Internal server error

## Implementation Guidelines

### File Header Standards

Each file should include the relevant Copilot prompt as a header comment:

```python
"""
# COPILOT PROMPT: [Relevant prompt from above]
# EXPECTED OUTPUT: [Brief description of expected functionality]
# POLYGON INTEGRATION: [Specific Polygon.io features used]
"""
```

### Function-Level Guidance

Before complex functions, include specific prompts:

```python
def fetch_polygon_data(symbols: List[str]) -> Dict[str, pd.DataFrame]:
    """
    # COPILOT: Implement batch fetching with rate limiting
    # Handle Polygon's aggregates API response format
    # Return standardized DataFrame with OHLCV + VWAP
    """
    pass
```

### Type Hints for Polygon Data

```python
from typing import Dict, List, Optional, Union
from datetime import datetime
import pandas as pd

PolygonSymbol = str  # Format: C:AAPL, O:SPY230818C00325000
PolygonResponse = Dict[str, Union[str, List[Dict], int]]
OHLCVData = pd.DataFrame  # Columns: [timestamp, open, high, low, close, volume, vwap]
```

## Validation Testing Commands

```bash
# Test all Copilot prompts
python devtools/test_polygon_prompts.py --all

# Test specific functionality
python devtools/test_polygon_prompts.py --test rate_limiting
python devtools/test_polygon_prompts.py --test websocket
python devtools/test_polygon_prompts.py --test options

# Validate prompt formatting
python devtools/test_polygon_prompts.py --validate-prompts
```

## Usage Workflow

1. **Copy relevant prompt** from this document
2. **Paste as comment** in your Python file
3. **Let GitHub Copilot** generate the implementation
4. **Run validation tests** to ensure correctness
5. **Refine based on test results**

## Best Practices

### For Developers
- Always include the prompt as a comment before implementing
- Test Copilot-generated code with the validation suite
- Update prompts based on real-world implementation challenges

### For Copilot
- Be specific about Polygon.io API endpoints and response formats
- Include error handling requirements in prompts
- Specify performance and scalability requirements
- Always mention PostgreSQL integration needs

## Troubleshooting

### Common Issues
1. **Copilot doesn't understand Polygon formats**: Add more specific examples in prompts
2. **Generated code lacks error handling**: Emphasize fault tolerance in prompts
3. **Performance issues**: Include specific performance requirements
4. **Integration problems**: Specify PostgreSQL schema expectations

### Solutions
- Use the `insert_copilot_headers.py` script to ensure consistent prompts
- Run `test_polygon_prompts.py` to validate implementations
- Update prompts based on test failures and real-world usage

---

**Last Updated**: August 2025  
**Version**: 1.0  
**Compatibility**: GitHub Copilot, ChatGPT, Claude
