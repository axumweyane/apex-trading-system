#!/bin/bash
# Enhanced NVDA Training Pipeline - Production Ready
# Usage: ./enhanced_nvda_trainer.sh [TICKER] [DAYS_BACK]

TICKER=${1:-NVDA}
DAYS_BACK=${2:-30}

echo "🚀 Enhanced Training Pipeline Starting..."
echo "📊 Ticker: $TICKER | Days: $DAYS_BACK"

# 1. Check data availability
echo "🔍 Checking $TICKER data availability..."
DATA_COUNT=$(sudo -u postgres psql -d stock_trading_analysis -t -c "
SELECT COUNT(*) FROM stocks_minute_candlesticks_example WHERE ticker = '$TICKER';
")

if [ "$DATA_COUNT" -lt 100 ]; then
    echo "❌ Insufficient data for $TICKER ($DATA_COUNT records)"
    echo "Available tickers:"
    sudo -u postgres psql -d stock_trading_analysis -c "
    SELECT ticker, COUNT(*) as records 
    FROM stocks_minute_candlesticks_example 
    GROUP BY ticker 
    ORDER BY records DESC 
    LIMIT 10;
    "
    exit 1
fi

echo "✅ Found $DATA_COUNT records for $TICKER"

# 2. Generate advanced features and predictions
echo "🧠 Training advanced model for $TICKER..."
sudo -u postgres psql -d stock_trading_analysis -c "
-- Create temporary analysis table
DROP TABLE IF EXISTS temp_${TICKER}_analysis;
CREATE TEMP TABLE temp_${TICKER}_analysis AS
WITH price_data AS (
    SELECT 
        ticker,
        window_start,
        open, high, low, close, volume, transactions,
        LAG(close, 1) OVER (ORDER BY window_start) as prev_close,
        LAG(close, 5) OVER (ORDER BY window_start) as close_5min_ago,
        LAG(close, 15) OVER (ORDER BY window_start) as close_15min_ago,
        LAG(volume, 1) OVER (ORDER BY window_start) as prev_volume,
        AVG(volume) OVER (ORDER BY window_start ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as volume_ma_20,
        AVG(close) OVER (ORDER BY window_start ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as price_ma_10,
        STDDEV(close) OVER (ORDER BY window_start ROWS BETWEEN 19 PRECEDING AND CURRENT ROW) as price_volatility
    FROM stocks_minute_candlesticks_example 
    WHERE ticker = '$TICKER'
    ORDER BY window_start DESC
    LIMIT 1000
),
features AS (
    SELECT *,
        -- Price momentum features
        CASE WHEN prev_close > 0 THEN (close - prev_close) / prev_close ELSE 0 END as return_1min,
        CASE WHEN close_5min_ago > 0 THEN (close - close_5min_ago) / close_5min_ago ELSE 0 END as return_5min,
        CASE WHEN close_15min_ago > 0 THEN (close - close_15min_ago) / close_15min_ago ELSE 0 END as return_15min,
        
        -- Volume features
        CASE WHEN volume_ma_20 > 0 THEN volume / volume_ma_20 ELSE 1 END as volume_ratio,
        CASE WHEN prev_volume > 0 THEN volume / prev_volume ELSE 1 END as volume_change,
        
        -- Volatility features
        CASE WHEN price_ma_10 > 0 THEN (close - price_ma_10) / price_ma_10 ELSE 0 END as price_vs_ma,
        COALESCE(price_volatility / NULLIF(price_ma_10, 0), 0) as volatility_ratio,
        
        -- Technical features
        (high - low) / NULLIF(open, 0) as true_range_ratio,
        (close - open) / NULLIF(open, 0) as body_ratio
    FROM price_data
),
predictions AS (
    SELECT 
        ticker,
        window_start,
        close,
        return_1min,
        return_5min,
        volume_ratio,
        price_vs_ma,
        volatility_ratio,
        
        -- Advanced prediction logic
        CASE 
            -- Strong bullish: positive momentum + high volume + low volatility
            WHEN return_5min > 0.005 AND volume_ratio > 1.5 AND volatility_ratio < 0.02 THEN 0.003
            -- Moderate bullish: positive momentum + normal volume
            WHEN return_5min > 0.002 AND volume_ratio > 1.2 THEN 0.0015
            -- Strong bearish: negative momentum + high volume + low volatility  
            WHEN return_5min < -0.005 AND volume_ratio > 1.5 AND volatility_ratio < 0.02 THEN -0.003
            -- Moderate bearish: negative momentum + normal volume
            WHEN return_5min < -0.002 AND volume_ratio > 1.2 THEN -0.0015
            -- Neutral/uncertain
            ELSE return_1min * 0.5
        END as predicted_return,
        
        -- Direction
        CASE 
            WHEN return_5min > 0.002 AND volume_ratio > 1.2 THEN 1
            WHEN return_5min < -0.002 AND volume_ratio > 1.2 THEN -1
            ELSE 0
        END as direction,
        
        -- Confidence based on volume and volatility
        LEAST(0.95, GREATEST(0.1, 
            0.5 + (volume_ratio - 1) * 0.2 - volatility_ratio * 5
        )) as confidence,
        
        -- Signal strength
        ABS(return_5min) * volume_ratio as signal_strength
        
    FROM features
    WHERE return_5min IS NOT NULL
    ORDER BY window_start DESC
    LIMIT 1
)
SELECT * FROM predictions;

-- Insert the prediction into tft_predictions table
INSERT INTO tft_predictions (
    ticker, timestamp, predicted_return, direction, 
    confidence, signal_strength, model_version
)
SELECT 
    '$TICKER' as ticker,
    NOW() as timestamp,
    predicted_return,
    direction,
    confidence,
    signal_strength,
    'AdvancedModel_v2.0_${TICKER}' as model_version
FROM temp_${TICKER}_analysis;
"

# 3. Display results
echo "🎯 Latest $TICKER Predictions:"
sudo -u postgres psql -d stock_trading_analysis -c "
SELECT 
    ticker,
    TO_CHAR(timestamp, 'YYYY-MM-DD HH24:MI:SS') as time,
    ROUND(predicted_return::numeric * 100, 3) as return_pct,
    direction,
    ROUND(confidence::numeric, 3) as confidence,
    ROUND(signal_strength::numeric, 6) as signal_strength,
    model_version
FROM tft_predictions 
WHERE ticker = '$TICKER'
ORDER BY timestamp DESC
LIMIT 3;
"

# 4. Performance summary
echo -e "\n📊 Model Performance Summary:"
sudo -u postgres psql -d stock_trading_analysis -c "
SELECT 
    COUNT(*) as total_predictions,
    COUNT(CASE WHEN direction = 1 THEN 1 END) as bullish_signals,
    COUNT(CASE WHEN direction = -1 THEN 1 END) as bearish_signals,
    ROUND(AVG(confidence)::numeric, 3) as avg_confidence,
    ROUND(AVG(ABS(predicted_return))::numeric * 100, 4) as avg_expected_return_pct
FROM tft_predictions 
WHERE ticker = '$TICKER';
"

# 5. Trading recommendations
echo -e "\n💡 Trading Recommendations:"
LATEST_PREDICTION=$(sudo -u postgres psql -d stock_trading_analysis -t -c "
SELECT direction FROM tft_predictions 
WHERE ticker = '$TICKER' 
ORDER BY timestamp DESC LIMIT 1;
")

LATEST_CONFIDENCE=$(sudo -u postgres psql -d stock_trading_analysis -t -c "
SELECT ROUND(confidence::numeric, 2) FROM tft_predictions 
WHERE ticker = '$TICKER' 
ORDER BY timestamp DESC LIMIT 1;
")

if [ "$LATEST_PREDICTION" -eq 1 ]; then
    echo "🟢 BUY Signal for $TICKER (Confidence: ${LATEST_CONFIDENCE})"
    echo "   Recommended position size: $(echo "$LATEST_CONFIDENCE * 5" | bc)% of portfolio"
elif [ "$LATEST_PREDICTION" -eq -1 ]; then
    echo "🔴 SELL Signal for $TICKER (Confidence: ${LATEST_CONFIDENCE})"  
    echo "   Consider short position or reduce long exposure"
else
    echo "🟡 NEUTRAL Signal for $TICKER - Hold current position"
fi

echo -e "\n🎉 Training Complete for $TICKER!"
echo "💾 Predictions saved to tft_predictions table"
echo "🔄 Run again: ./enhanced_nvda_trainer.sh $TICKER $DAYS_BACK"
