#!/bin/bash
# Simple NVDA data extraction and analysis

echo "🚀 NVDA Analysis Pipeline Starting..."

# 1. Extract NVDA data
echo "📊 Extracting NVDA data..."
sudo -u postgres psql -d stock_trading_analysis -c "
SELECT 
    COUNT(*) as total_records,
    MIN(TO_TIMESTAMP(window_start)) as earliest_data,
    MAX(TO_TIMESTAMP(window_start)) as latest_data,
    AVG(close) as avg_price,
    AVG(volume) as avg_volume
FROM stocks_minute_candlesticks_example 
WHERE ticker = 'NVDA';
" > nvda_summary.txt

echo "✅ Data summary saved to nvda_summary.txt"
cat nvda_summary.txt

# 2. Get recent NVDA data for analysis
echo -e "\n📈 Recent NVDA price action..."
sudo -u postgres psql -d stock_trading_analysis -c "
SELECT 
    TO_TIMESTAMP(window_start) as time,
    open, high, low, close, volume,
    ROUND(((close - open) / open * 100)::numeric, 2) as return_pct
FROM stocks_minute_candlesticks_example 
WHERE ticker = 'NVDA'
ORDER BY window_start DESC
LIMIT 10;
"

# 3. Calculate simple predictions and insert
echo -e "\n🧠 Generating simple predictions..."
sudo -u postgres psql -d stock_trading_analysis -c "
WITH recent_nvda AS (
    SELECT 
        close,
        LAG(close, 1) OVER (ORDER BY window_start) as prev_close,
        LAG(close, 5) OVER (ORDER BY window_start) as close_5min_ago,
        volume,
        AVG(volume) OVER (ORDER BY window_start ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as avg_volume_10
    FROM stocks_minute_candlesticks_example 
    WHERE ticker = 'NVDA'
    ORDER BY window_start DESC
    LIMIT 1
),
prediction AS (
    SELECT 
        'NVDA' as ticker,
        NOW() as timestamp,
        CASE 
            WHEN close > close_5min_ago AND volume > avg_volume_10 THEN 0.001  -- Bullish
            WHEN close < close_5min_ago AND volume > avg_volume_10 THEN -0.001 -- Bearish  
            ELSE 0.0001 -- Neutral
        END as predicted_return,
        CASE 
            WHEN close > close_5min_ago AND volume > avg_volume_10 THEN 1
            WHEN close < close_5min_ago AND volume > avg_volume_10 THEN -1
            ELSE 0
        END as direction,
        CASE 
            WHEN volume > avg_volume_10 * 1.5 THEN 0.8
            WHEN volume > avg_volume_10 THEN 0.6
            ELSE 0.4
        END as confidence,
        ABS(close - close_5min_ago) / close_5min_ago as signal_strength,
        'SimpleModel_v1.0' as model_version
    FROM recent_nvda
)
INSERT INTO tft_predictions (ticker, timestamp, predicted_return, direction, confidence, signal_strength, model_version)
SELECT ticker, timestamp, predicted_return, direction, confidence, signal_strength, model_version
FROM prediction;
"

echo "✅ Simple predictions generated and saved!"

# 4. Show the predictions
echo -e "\n🎯 Latest NVDA predictions:"
sudo -u postgres psql -d stock_trading_analysis -c "
SELECT 
    ticker,
    timestamp,
    predicted_return,
    direction,
    ROUND(confidence::numeric, 3) as confidence,
    ROUND(signal_strength::numeric, 6) as signal_strength,
    model_version
FROM tft_predictions 
WHERE ticker = 'NVDA'
ORDER BY timestamp DESC
LIMIT 5;
"

echo -e "\n🎉 NVDA Analysis Pipeline Complete!"
echo "💡 Your model is now generating predictions in the tft_predictions table"
echo "🚀 Ready to use these predictions for trading!"
