import asyncio
import asyncpg
import pandas as pd

async def test_data_loading():
    conn = await asyncpg.connect(
        host='localhost',
        port=5432,
        user='kibrom',
        password='beriha@123KB!',
        database='stock_trading_analysis'
    )
    
    query = """
    SELECT 
        ticker,
        TO_TIMESTAMP(window_start / 1000000000) as timestamp,
        open::float as open,
        high::float as high, 
        low::float as low,
        close::float as close,
        volume::int as volume,
        transactions::int as transactions
    FROM stocks_minute_candlesticks_example 
    WHERE ticker = ANY($1::text[])
    ORDER BY ticker, window_start
    LIMIT $2
    """
    
    tickers = ['AAPL']
    rows = await conn.fetch(query, tickers, 10)
    
    print("Raw rows:")
    for row in rows:
        print(dict(row))
    
    df = pd.DataFrame(rows)
    print("\nDataFrame info:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"DataFrame:\n{df}")
    
    await conn.close()

if __name__ == "__main__":
    asyncio.run(test_data_loading())
