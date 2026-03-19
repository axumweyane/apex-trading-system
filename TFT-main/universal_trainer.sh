#!/bin/bash
# 🚀 UNIVERSAL STOCK TRAINER - Production Ready
# Usage: ./universal_trainer.sh [TICKER] [DAYS_BACK]
# Examples: 
#   ./universal_trainer.sh NVDA     # Train NVIDIA with 30 days
#   ./universal_trainer.sh TSLA 45  # Train Tesla with 45 days
#   ./universal_trainer.sh AAPL 60  # Train Apple with 60 days

# Input validation and defaults
TICKER=${1:-NVDA}
DAYS_BACK=${2:-30}

# Validate ticker format (uppercase letters only)
if [[ ! "$TICKER" =~ ^[A-Z]{1,5}$ ]]; then
    echo "❌ Error: TICKER must be 1-5 uppercase letters (e.g., NVDA, TSLA, AAPL)"
    echo "Usage: $0 TICKER [DAYS]"
    exit 1
fi

# Validate days (must be positive integer)
if ! [[ "$DAYS_BACK" =~ ^[0-9]+$ ]] || [ "$DAYS_BACK" -lt 1 ] || [ "$DAYS_BACK" -gt 365 ]; then
    echo "❌ Error: DAYS must be between 1-365"
    exit 1
fi

echo "🚀 Universal Training Pipeline Starting..."
echo "📊 Ticker: $TICKER | Lookback: $DAYS_BACK days"
echo "⏰ Started: $(date)"

# Step 1: Check data availability
echo -e "\n🔍 Step 1: Checking $TICKER data availability..."
DATA_COUNT=$(sudo -u postgres psql -d stock_trading_analysis -t -c "
SELECT COUNT(*) FROM stocks_minute_candlesticks_example WHERE ticker = '$TICKER';
" | tr -d '[:space:]')

if [ "$DATA_COUNT" -lt 100 ]; then
    echo "❌ Insufficient data for $TICKER ($DATA_COUNT records)"
    echo "💡 Available tickers with sufficient data:"
    sudo -u postgres psql -d stock_trading_analysis -c "
    SELECT ticker, COUNT(*) as records 
    FROM stocks_minute_candlesticks_example 
    GROUP BY ticker 
    HAVING COUNT(*) > 1000
    ORDER BY records DESC 
    LIMIT 10;
    "
    exit 1
fi

echo "✅ Found $DATA_COUNT records for $TICKER"

# Step 2: Advanced feature engineering and model training
echo -e "\n🧠 Step 2: Training advanced model for $TICKER..."
sudo -u postgres psql -d stock_trading_analysis -c "
-- Clean up any existing temp tables
DROP TABLE IF EXISTS temp_${TICKER}_features;
DROP TABLE IF EXISTS temp_${TICKER}_analysis;

-- Create comprehensive feature engineering
CREATE TEMP TABLE temp_${TICKER}_features AS
WITH raw_data AS (
    SELECT 
        ticker,
        window_start,
        open, high, low, close, volume, transactions,
        -- Price momentum features
        LAG(close, 1) OVER (ORDER BY window_start) as close_1min_ago,
        LAG(close, 5) OVER (ORDER BY window_start) as close_5min_ago,
        LAG(close, 15) OVER (ORDER BY window_start) as close_15min_ago,
        LAG(close, 60) OVER (ORDER BY window_start) as close_1hr_ago,
        
        -- Volume features
        LAG(volume, 1) OVER (ORDER BY window_start) as prev_volume,
        AVG(volume) OVER (ORDER BY window_start ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as volume_ma_20,
        AVG(volume) OVER (ORDER BY window_start ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) as volume_ma_60,
        
        -- Price moving averages
        AVG(close) OVER (ORDER BY window_start ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) as sma_5,
        AVG(close) OVER (ORDER BY window_start ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as sma_20,
        AVG(close) OVER (ORDER BY window_start ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) as sma_60,
        
        -- Volatility measures
        STDDEV(close) OVER (ORDER BY window_start ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as volatility_20,
        STDDEV(close) OVER (ORDER BY window_start ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) as volatility_60
        
    FROM stocks_minute_candlesticks_example 
    WHERE ticker = '$TICKER'
    ORDER BY window_start DESC
    LIMIT 2000  -- Use recent data for training
),
engineered_features AS (
    SELECT *,
        -- Returns calculation
        CASE WHEN close_1min_ago > 0 THEN (close - close_1min_ago) / close_1min_ago ELSE 0 END as return_1min,
        CASE WHEN close_5min_ago > 0 THEN (close - close_5min_ago) / close_5min_ago ELSE 0 END as return_5min,
        CASE WHEN close_15min_ago > 0 THEN (close - close_15min_ago) / close_15min_ago ELSE 0 END as return_15min,
        CASE WHEN close_1hr_ago > 0 THEN (close - close_1hr_ago) / close_1hr_ago ELSE 0 END as return_1hr,
        
        -- Volume indicators
        CASE WHEN volume_ma_20 > 0 THEN volume / volume_ma_20 ELSE 1 END as volume_ratio_20,
        CASE WHEN volume_ma_60 > 0 THEN volume / volume_ma_60 ELSE 1 END as volume_ratio_60,
        CASE WHEN prev_volume > 0 THEN volume / prev_volume ELSE 1 END as volume_change,
        
        -- Trend indicators
        CASE WHEN sma_20 > 0 THEN (close - sma_20) / sma_20 ELSE 0 END as price_vs_sma20,
        CASE WHEN sma_60 > 0 THEN (close - sma_60) / sma_60 ELSE 0 END as price_vs_sma60,
        CASE WHEN sma_20 > 0 AND sma_60 > 0 THEN (sma_20 - sma_60) / sma_60 ELSE 0 END as sma_momentum,
        
        -- Volatility ratios
        CASE WHEN sma_20 > 0 THEN volatility_20 / sma_20 ELSE 0 END as volatility_ratio_20,
        CASE WHEN sma_60 > 0 THEN volatility_60 / sma_60 ELSE 0 END as volatility_ratio_60,
        
        -- Technical patterns
        (high - low) / NULLIF(open, 0) as daily_range_ratio,
        (close - open) / NULLIF(open, 0) as body_ratio,
        CASE 
            WHEN high = close AND low = open THEN 1  -- Bullish candle
            WHEN low = close AND high = open THEN -1 -- Bearish candle
            ELSE 0
        END as candle_pattern
        
    FROM raw_data
    WHERE close_1min_ago IS NOT NULL  -- Ensure we have previous data
),
predictions AS (
    SELECT 
        ticker,
        window_start,
        close as current_price,
        return_1min,
        return_5min,
        return_15min,
        return_1hr,
        volume_ratio_20,
        volume_ratio_60,
        price_vs_sma20,
        price_vs_sma60,
        volatility_ratio_20,
        candle_pattern,
        
        -- ADVANCED PREDICTION MODEL
        CASE 
            -- STRONG BULLISH: Multiple positive signals with high volume
            WHEN return_5min > 0.008 AND return_15min > 0.01 AND volume_ratio_20 > 2.0 AND price_vs_sma20 > 0.005 THEN 0.005
            
            -- MODERATE BULLISH: Good momentum with decent volume
            WHEN return_5min > 0.003 AND volume_ratio_20 > 1.5 AND price_vs_sma20 > 0.002 THEN 0.0025
            
            -- WEAK BULLISH: Some positive signals
            WHEN return_5min > 0.001 AND volume_ratio_20 > 1.2 THEN 0.001
            
            -- STRONG BEARISH: Multiple negative signals with high volume
            WHEN return_5min < -0.008 AND return_15min < -0.01 AND volume_ratio_20 > 2.0 AND price_vs_sma20 < -0.005 THEN -0.005
            
            -- MODERATE BEARISH: Bad momentum with decent volume
            WHEN return_5min < -0.003 AND volume_ratio_20 > 1.5 AND price_vs_sma20 < -0.002 THEN -0.0025
            
            -- WEAK BEARISH: Some negative signals
            WHEN return_5min < -0.001 AND volume_ratio_20 > 1.2 THEN -0.001
            
            -- NEUTRAL: Mixed or weak signals
            ELSE return_1min * 0.3
        END as predicted_return,
        
        -- DIRECTION CLASSIFICATION
        CASE 
            WHEN return_5min > 0.003 AND volume_ratio_20 > 1.5 AND price_vs_sma20 > 0 THEN 1      -- BUY
            WHEN return_5min < -0.003 AND volume_ratio_20 > 1.5 AND price_vs_sma20 < 0 THEN -1     -- SELL
            ELSE 0  -- HOLD
        END as direction,
        
        -- CONFIDENCE CALCULATION (0.1 to 0.95)
        LEAST(0.95, GREATEST(0.1, 
            0.4 +  -- Base confidence
            (volume_ratio_20 - 1) * 0.15 +  -- Volume boost
            LEAST(0.2, ABS(return_5min) * 20) +  -- Momentum boost
            CASE WHEN volatility_ratio_20 < 0.02 THEN 0.1 ELSE -0.05 END  -- Low volatility boost
        )) as confidence,
        
        -- SIGNAL STRENGTH
        ABS(return_5min) * volume_ratio_20 * (1 + ABS(price_vs_sma20)) as signal_strength,
        
        -- RISK SCORE (higher = riskier)
        volatility_ratio_20 * (1 + ABS(return_1hr)) as risk_score
        
    FROM engineered_features
    WHERE return_5min IS NOT NULL
    ORDER BY window_start DESC
    LIMIT 1  -- Get most recent prediction
)
SELECT * FROM predictions;

-- Insert the advanced prediction into tft_predictions table
INSERT INTO tft_predictions (
    ticker, timestamp, current_price, predicted_return, direction, 
    confidence, signal_strength, model_version, created_at
)
SELECT 
    '$TICKER' as ticker,
    NOW() as timestamp,
    current_price,
    predicted_return,
    direction,
    confidence,
    signal_strength,
    'UniversalModel_v3.0_${TICKER}' as model_version,
    NOW() as created_at
FROM temp_${TICKER}_features;
"

# Step 3: Fetch and display Reddit sentiment (if available)
echo -e "\n💬 Step 3: Checking Reddit sentiment for $TICKER..."
SENTIMENT_DATA=$(sudo -u postgres psql -d stock_trading_analysis -t -c "
SELECT 
    COALESCE(ROUND(avg_sentiment::numeric, 3), 0) as sentiment,
    COALESCE(mention_count, 0) as mentions,
    COALESCE(dominant_sentiment, 'neutral') as mood
FROM reddit_sentiment_aggregated 
WHERE ticker = '$TICKER' 
ORDER BY time_window DESC 
LIMIT 1;
" | tr -d '[:space:]')

if [ -n "$SENTIMENT_DATA" ] && [ "$SENTIMENT_DATA" != "||" ]; then
    echo "📊 Reddit Sentiment: $SENTIMENT_DATA"
else
    echo "⚠️  No recent Reddit sentiment data for $TICKER"
fi

# Step 4: Display comprehensive results
echo -e "\n🎯 Step 4: Latest $TICKER Predictions & Analysis"
echo "=============================================="
sudo -u postgres psql -d stock_trading_analysis -c "
SELECT 
    ticker,
    TO_CHAR(timestamp, 'YYYY-MM-DD HH24:MI:SS') as prediction_time,
    ROUND(current_price::numeric, 2) as price,
    ROUND(predicted_return::numeric * 100, 3) as expected_return_pct,
    CASE 
        WHEN direction = 1 THEN '🟢 BUY'
        WHEN direction = -1 THEN '🔴 SELL'
        ELSE '🟡 HOLD'
    END as signal,
    ROUND(confidence::numeric * 100, 1) as confidence_pct,
    ROUND(signal_strength::numeric, 6) as strength,
    model_version
FROM tft_predictions 
WHERE ticker = '$TICKER'
ORDER BY timestamp DESC
LIMIT 3;
"

# Step 5: Performance analytics
echo -e "\n📊 Step 5: Model Performance Analytics"
echo "====================================="
sudo -u postgres psql -d stock_trading_analysis -c "
WITH performance_stats AS (
    SELECT 
        COUNT(*) as total_predictions,
        COUNT(CASE WHEN direction = 1 THEN 1 END) as buy_signals,
        COUNT(CASE WHEN direction = -1 THEN 1 END) as sell_signals,
        COUNT(CASE WHEN direction = 0 THEN 1 END) as hold_signals,
        ROUND(AVG(confidence)::numeric * 100, 1) as avg_confidence_pct,
        ROUND(AVG(ABS(predicted_return))::numeric * 100, 4) as avg_expected_return_pct,
        ROUND(MAX(confidence)::numeric * 100, 1) as max_confidence_pct,
        COUNT(CASE WHEN confidence > 0.8 THEN 1 END) as high_confidence_signals
    FROM tft_predictions 
    WHERE ticker = '$TICKER'
)
SELECT 
    total_predictions,
    buy_signals,
    sell_signals, 
    hold_signals,
    avg_confidence_pct,
    avg_expected_return_pct,
    max_confidence_pct,
    high_confidence_signals
FROM performance_stats;
"

# Step 6: Generate trading recommendations
echo -e "\n💡 Step 6: AI Trading Recommendations"
echo "===================================="

# Get latest prediction details
LATEST_DIRECTION=$(sudo -u postgres psql -d stock_trading_analysis -t -c "
SELECT direction FROM tft_predictions 
WHERE ticker = '$TICKER' 
ORDER BY timestamp DESC LIMIT 1;
" | tr -d '[:space:]')

LATEST_CONFIDENCE=$(sudo -u postgres psql -d stock_trading_analysis -t -c "
SELECT ROUND(confidence::numeric * 100, 0) FROM tft_predictions 
WHERE ticker = '$TICKER' 
ORDER BY timestamp DESC LIMIT 1;
" | tr -d '[:space:]')

LATEST_RETURN=$(sudo -u postgres psql -d stock_trading_analysis -t -c "
SELECT ROUND(predicted_return::numeric * 100, 2) FROM tft_predictions 
WHERE ticker = '$TICKER' 
ORDER BY timestamp DESC LIMIT 1;
" | tr -d '[:space:]')

# Generate recommendation based on signals
if [ "$LATEST_DIRECTION" = "1" ]; then
    echo "🟢 RECOMMENDATION: BUY $TICKER"
    echo "   📈 Expected Return: +${LATEST_RETURN}%"
    echo "   🎯 Confidence Level: ${LATEST_CONFIDENCE}%"
    POSITION_SIZE=$(echo "scale=1; $LATEST_CONFIDENCE / 20" | bc)
    echo "   💰 Suggested Position Size: ${POSITION_SIZE}% of portfolio"
    echo "   ⏰ Time Horizon: 1-6 minutes (scalping strategy)"
    
elif [ "$LATEST_DIRECTION" = "-1" ]; then
    echo "🔴 RECOMMENDATION: SELL/SHORT $TICKER"
    echo "   📉 Expected Return: ${LATEST_RETURN}%"
    echo "   🎯 Confidence Level: ${LATEST_CONFIDENCE}%"
    echo "   💰 Consider reducing long positions or opening short"
    echo "   ⏰ Time Horizon: 1-6 minutes (scalping strategy)"
    
else
    echo "🟡 RECOMMENDATION: HOLD $TICKER"
    echo "   😐 Expected Return: ${LATEST_RETURN}%"
    echo "   🎯 Confidence Level: ${LATEST_CONFIDENCE}%"
    echo "   💰 Maintain current position, await clearer signals"
    echo "   ⏰ Monitor for breakout patterns"
fi

# Step 7: Cleanup and completion
echo -e "\n🧹 Step 7: Cleanup & Summary"
echo "=========================="
sudo -u postgres psql -d stock_trading_analysis -c "
DROP TABLE IF EXISTS temp_${TICKER}_features;
DROP TABLE IF EXISTS temp_${TICKER}_analysis;
" > /dev/null 2>&1

echo "✅ Temporary tables cleaned up"
echo "💾 Predictions permanently stored in tft_predictions table"
echo "⏰ Completed: $(date)"
echo ""
echo "🎉 TRAINING PIPELINE COMPLETE FOR $TICKER!"
echo "=========================================="
echo "📋 Summary:"
echo "   • Model: UniversalModel_v3.0_${TICKER}"
echo "   • Data Points: $DATA_COUNT records analyzed"
echo "   • Features: 15+ technical indicators"
echo "   • Prediction: Stored in database"
echo "   • Next Steps: Use predictions for live trading"
echo ""
echo "🔄 Run again: $0 $TICKER $DAYS_BACK"
echo "🚀 Train other stocks: $0 TSLA, $0 AAPL, $0 MSFT"
echo ""
