"""
APEX Paper Trading Runner — Daily Ensemble Execution Service.

Runs the full multi-strategy pipeline on a schedule:
  1. Fetch latest market data (yfinance)
  2. Run all enabled strategies (momentum, pairs, FX, options)
  3. Detect market regime
  4. Combine signals via Bayesian ensemble
  5. Optimize portfolio with risk constraints
  6. Execute trades via Alpaca paper account
  7. Log everything to PostgreSQL
  8. Send daily P&L report to Discord
  9. Serve a live dashboard at /dashboard

Port: 8010
"""

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime, date, timezone, timedelta
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import Json
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# Add parent directory to path for strategy imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from strategies.momentum.cross_sectional import CrossSectionalMomentum
from strategies.statarb.pairs import PairsTrading
from strategies.fx.carry_trend import FXCarryTrend
from strategies.ensemble.combiner import EnsembleCombiner, TFTAdapter
from strategies.ensemble.portfolio_optimizer import PortfolioOptimizer
from strategies.regime.detector import RegimeDetector
from strategies.risk.portfolio_risk import PortfolioRiskManager
from strategies.config import (
    MomentumConfig, StatArbConfig, EnsembleConfig, RegimeConfig, FXConfig,
)
from strategies.base import StrategyPerformance

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("paper-trader")

# Configuration from environment
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "tft_trading")
DB_USER = os.getenv("DB_USER", "tft_user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "tft_password")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
TRADING_SYMBOLS = os.getenv(
    "PAPER_TRADING_SYMBOLS",
    "AAPL,MSFT,GOOGL,AMZN,NVDA,META,TSLA,JPM,BAC,XOM",
).split(",")
FX_PAIRS = os.getenv("PAPER_FX_PAIRS", "EURUSD,GBPUSD,USDJPY,AUDUSD,USDCAD,USDCHF").split(",")
SCHEDULE_HOUR = int(os.getenv("PAPER_SCHEDULE_HOUR", "10"))  # 10 AM ET
SCHEDULE_MINUTE = int(os.getenv("PAPER_SCHEDULE_MINUTE", "0"))
INITIAL_CAPITAL = float(os.getenv("PAPER_INITIAL_CAPITAL", "100000"))

# ============================================================
# DATABASE SCHEMA
# ============================================================
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS paper_trades (
    id              SERIAL PRIMARY KEY,
    trade_date      DATE NOT NULL,
    symbol          VARCHAR(16) NOT NULL,
    side            VARCHAR(8) NOT NULL,
    quantity        DOUBLE PRECISION,
    price           DOUBLE PRECISION,
    order_id        VARCHAR(64),
    strategy_source VARCHAR(64),
    signal_score    DOUBLE PRECISION,
    signal_confidence DOUBLE PRECISION,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS paper_daily_snapshots (
    id              SERIAL PRIMARY KEY,
    snapshot_date   DATE NOT NULL,
    portfolio_value DOUBLE PRECISION,
    cash            DOUBLE PRECISION,
    equity          DOUBLE PRECISION,
    daily_pnl       DOUBLE PRECISION,
    daily_return_pct DOUBLE PRECISION,
    total_return_pct DOUBLE PRECISION,
    positions_count INTEGER,
    regime          VARCHAR(32),
    strategy_weights JSONB DEFAULT '{}',
    positions       JSONB DEFAULT '{}',
    risk_metrics    JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS paper_strategy_signals (
    id              SERIAL PRIMARY KEY,
    signal_date     DATE NOT NULL,
    strategy_name   VARCHAR(64) NOT NULL,
    symbol          VARCHAR(16) NOT NULL,
    score           DOUBLE PRECISION,
    confidence      DOUBLE PRECISION,
    direction       VARCHAR(16),
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_pt_date ON paper_trades(trade_date);
CREATE INDEX IF NOT EXISTS idx_pds_date ON paper_daily_snapshots(snapshot_date);
CREATE INDEX IF NOT EXISTS idx_pss_date ON paper_strategy_signals(signal_date);
"""


# ============================================================
# GLOBAL STATE
# ============================================================
class AppState:
    def __init__(self):
        self.scheduler: Optional[AsyncIOScheduler] = None
        self.last_run: Optional[datetime] = None
        self.last_regime: Optional[str] = None
        self.last_weights: Dict[str, float] = {}
        self.last_positions: List[Dict] = []
        self.last_pnl: float = 0.0
        self.total_return_pct: float = 0.0
        self.portfolio_value: float = INITIAL_CAPITAL
        self.day_count: int = 0
        self.daily_returns: List[float] = []
        self.run_log: List[Dict] = []
        self.is_running: bool = False


state = AppState()


# ============================================================
# DATABASE
# ============================================================
def get_db_conn():
    return psycopg2.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
    )


def init_db():
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
        conn.commit()
        conn.close()
        logger.info("Database schema initialized")
    except Exception as e:
        logger.warning("Database init failed (will retry on first run): %s", e)


def log_trade(trade_date, symbol, side, quantity, price, order_id, strategy, score, confidence, metadata):
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO paper_trades
                   (trade_date, symbol, side, quantity, price, order_id,
                    strategy_source, signal_score, signal_confidence, metadata)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (trade_date, symbol, side, quantity, price, order_id,
                 strategy, score, confidence, Json(metadata or {})),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("Failed to log trade: %s", e)


def log_daily_snapshot(snapshot):
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO paper_daily_snapshots
                   (snapshot_date, portfolio_value, cash, equity, daily_pnl,
                    daily_return_pct, total_return_pct, positions_count,
                    regime, strategy_weights, positions, risk_metrics)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (snapshot["date"], snapshot["portfolio_value"], snapshot["cash"],
                 snapshot["equity"], snapshot["daily_pnl"], snapshot["daily_return_pct"],
                 snapshot["total_return_pct"], snapshot["positions_count"],
                 snapshot["regime"], Json(snapshot.get("strategy_weights", {})),
                 Json(snapshot.get("positions", [])), Json(snapshot.get("risk_metrics", {}))),
            )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("Failed to log snapshot: %s", e)


def log_signals(signal_date, strategy_name, signals_df):
    try:
        conn = get_db_conn()
        with conn.cursor() as cur:
            for _, row in signals_df.iterrows():
                cur.execute(
                    """INSERT INTO paper_strategy_signals
                       (signal_date, strategy_name, symbol, score, confidence, direction, metadata)
                       VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                    (signal_date, strategy_name, row.get("symbol", ""),
                     row.get("score", 0), row.get("confidence", 0),
                     row.get("direction", ""), Json({})),
                )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error("Failed to log signals: %s", e)


# ============================================================
# ALPACA BROKER
# ============================================================
class PaperBroker:
    """Lightweight Alpaca paper trading client."""

    def __init__(self):
        self.base_url = ALPACA_BASE_URL.rstrip("/")
        self.headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
            "Content-Type": "application/json",
        }
        self._session: Optional[aiohttp.ClientSession] = None

    async def connect(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def disconnect(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _call(self, method, endpoint, data=None):
        await self.connect()
        url = f"{self.base_url}{endpoint}"
        async with self._session.request(method, url, headers=self.headers,
                                          json=data, timeout=aiohttp.ClientTimeout(total=15)) as resp:
            if resp.status in (200, 207):
                return await resp.json()
            if resp.status == 204:
                return None
            body = await resp.text()
            logger.error("Alpaca %s %s -> %s: %s", method, endpoint, resp.status, body)
            return None

    async def get_account(self) -> Dict:
        return await self._call("GET", "/v2/account") or {}

    async def get_positions(self) -> List[Dict]:
        return await self._call("GET", "/v2/positions") or []

    async def submit_order(self, symbol, qty, side, order_type="market", time_in_force="day"):
        data = {
            "symbol": symbol,
            "qty": str(abs(qty)),
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force,
        }
        return await self._call("POST", "/v2/orders", data)

    async def close_position(self, symbol):
        return await self._call("DELETE", f"/v2/positions/{symbol}")

    async def cancel_all_orders(self):
        return await self._call("DELETE", "/v2/orders")


broker = PaperBroker()


# ============================================================
# DISCORD NOTIFICATIONS
# ============================================================
async def send_discord_report(report: str):
    if not DISCORD_WEBHOOK_URL or "YOUR_WEBHOOK" in DISCORD_WEBHOOK_URL:
        logger.info("Discord not configured, skipping notification")
        return

    embed = {
        "title": "APEX Paper Trading — Daily Report",
        "description": report,
        "color": 0x4CAF50 if state.last_pnl >= 0 else 0xF44336,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "footer": {"text": f"Day {state.day_count}/30 | Paper Trading"},
    }
    payload = {"embeds": [embed]}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(DISCORD_WEBHOOK_URL, json=payload,
                                     timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status in (200, 204):
                    logger.info("Discord report sent")
                else:
                    logger.error("Discord failed: %s", resp.status)
    except Exception as e:
        logger.error("Discord send error: %s", e)


# ============================================================
# DATA FETCHING
# ============================================================
def fetch_stock_data(symbols: List[str], days: int = 300) -> pd.DataFrame:
    """Fetch recent price data via yfinance."""
    import yfinance as yf
    logger.info("Fetching data for %d symbols...", len(symbols))

    all_symbols = symbols + ["SPY"]  # always include SPY for regime
    data = yf.download(all_symbols, period=f"{days}d", group_by="ticker",
                       auto_adjust=True, progress=False)

    rows = []
    for sym in all_symbols:
        try:
            sym_data = data[sym].dropna() if len(all_symbols) > 1 else data.dropna()
            for dt, row in sym_data.iterrows():
                rows.append({
                    "symbol": sym, "timestamp": dt,
                    "open": float(row["Open"]), "high": float(row["High"]),
                    "low": float(row["Low"]), "close": float(row["Close"]),
                    "volume": int(row["Volume"]),
                })
        except Exception as e:
            logger.warning("Failed to get data for %s: %s", sym, e)

    df = pd.DataFrame(rows)
    logger.info("Fetched %d rows for %d symbols", len(df), df["symbol"].nunique())
    return df


def fetch_fx_data(pairs: List[str], days: int = 200) -> pd.DataFrame:
    """Fetch FX data via yfinance."""
    import yfinance as yf

    yf_map = {
        "EURUSD": "EURUSD=X", "GBPUSD": "GBPUSD=X", "USDJPY": "JPY=X",
        "AUDUSD": "AUDUSD=X", "USDCAD": "CAD=X", "USDCHF": "CHF=X",
    }

    yf_symbols = [yf_map.get(p, f"{p}=X") for p in pairs]
    data = yf.download(yf_symbols, period=f"{days}d", group_by="ticker",
                       auto_adjust=True, progress=False)

    rows = []
    for pair, yf_sym in zip(pairs, yf_symbols):
        try:
            sym_data = data[yf_sym].dropna()
            for dt, row in sym_data.iterrows():
                rows.append({
                    "symbol": pair, "timestamp": dt,
                    "close": float(row["Close"]), "volume": 0,
                })
        except Exception:
            pass

    return pd.DataFrame(rows)


# ============================================================
# MAIN PIPELINE
# ============================================================
async def run_daily_pipeline():
    """Execute the full ensemble pipeline."""
    if state.is_running:
        logger.warning("Pipeline already running, skipping")
        return

    state.is_running = True
    run_start = datetime.now(timezone.utc)
    today = date.today()

    try:
        logger.info("=" * 60)
        logger.info("DAILY PIPELINE START — %s", today)
        logger.info("=" * 60)

        # 1. Fetch data
        stock_data = fetch_stock_data(TRADING_SYMBOLS)
        fx_data = fetch_fx_data(FX_PAIRS)

        if stock_data.empty:
            logger.error("No stock data fetched, aborting")
            return

        # 2. Get current account state
        account = await broker.get_account()
        prev_value = state.portfolio_value
        state.portfolio_value = float(account.get("portfolio_value", prev_value))
        cash = float(account.get("cash", 0))
        equity = float(account.get("equity", 0))

        # 3. Detect regime
        regime_detector = RegimeDetector(RegimeConfig(enabled=True))
        regime_state = regime_detector.detect(stock_data, vix_value=None)
        state.last_regime = regime_state.regime.value

        logger.info("Regime: %s (exposure_scalar=%.0f%%)",
                     regime_state.regime.value, regime_state.exposure_scalar * 100)

        # 4. Run strategies
        strategy_outputs = []

        # Momentum
        try:
            mom = CrossSectionalMomentum(MomentumConfig(
                enabled=True, min_history_days=250, min_avg_dollar_volume=0,
                long_threshold_zscore=1.0, short_threshold_zscore=-1.0,
                max_positions_per_side=5,
            ))
            mom_output = mom.generate_signals(stock_data)
            strategy_outputs.append(mom_output)
            mom_df = mom_output.to_dataframe()
            if not mom_df.empty:
                log_signals(today, "momentum", mom_df)
            logger.info("Momentum: %d signals", len(mom_output.scores))
        except Exception as e:
            logger.error("Momentum strategy failed: %s", e)

        # Pairs trading
        try:
            pairs = PairsTrading(StatArbConfig(
                enabled=True, cointegration_pvalue=0.10,
                same_sector_only=False, max_pairs=10,
            ))
            pairs_output = pairs.generate_signals(stock_data)
            strategy_outputs.append(pairs_output)
            logger.info("Pairs: %d signals", len(pairs_output.scores))
        except Exception as e:
            logger.error("Pairs strategy failed: %s", e)

        # FX
        if not fx_data.empty:
            try:
                fx = FXCarryTrend(FXConfig(enabled=True))
                fx_output = fx.generate_signals(fx_data)
                strategy_outputs.append(fx_output)
                logger.info("FX: %d signals", len(fx_output.scores))
            except Exception as e:
                logger.error("FX strategy failed: %s", e)

        # 5. Combine via ensemble
        combiner = EnsembleCombiner(EnsembleConfig(
            enabled=True, weighting_method="bayesian",
            max_total_positions=20, max_gross_leverage=1.5,
        ))
        combined = combiner.combine(strategy_outputs, regime_state)

        # Track weights
        weight_hist = combiner.get_weight_history(1)
        if weight_hist:
            state.last_weights = {
                name: round(w.final_weight, 3) for name, w in weight_hist[-1].items()
            }

        # 6. Optimize portfolio
        optimizer = PortfolioOptimizer(EnsembleConfig(
            enabled=True, max_total_positions=15,
            max_gross_leverage=1.2, target_volatility=0.15,
        ))
        target = optimizer.optimize(combined, stock_data, regime_state)

        logger.info("Target portfolio: %d positions, gross=%.2f, net=%.2f",
                     target.position_count, target.gross_leverage, target.net_leverage)

        # 7. Execute trades
        current_positions = await broker.get_positions()
        current_holdings = {p["symbol"]: float(p["qty"]) for p in current_positions}

        trades_executed = 0
        for pos in target.positions:
            symbol = pos.symbol
            target_value = pos.target_weight * state.portfolio_value
            current_qty = current_holdings.get(symbol, 0)

            # Get latest price
            sym_data = stock_data[stock_data["symbol"] == symbol]
            if sym_data.empty:
                continue
            price = float(sym_data.sort_values("timestamp")["close"].iloc[-1])
            if price <= 0:
                continue

            target_shares = int(target_value / price)
            diff = target_shares - current_qty

            if abs(diff) < 1:
                continue

            side = "buy" if diff > 0 else "sell"
            qty = abs(diff)

            result = await broker.submit_order(symbol, qty, side)
            if result:
                trades_executed += 1
                order_id = result.get("id", "")
                log_trade(today, symbol, side, qty, price, order_id,
                          "ensemble", pos.combined_score, pos.confidence,
                          {"target_weight": pos.target_weight})
                logger.info("  %s %d %s @ $%.2f (order=%s)",
                            side.upper(), qty, symbol, price, order_id[:8])

        # 8. Snapshot
        account = await broker.get_account()
        state.portfolio_value = float(account.get("portfolio_value", state.portfolio_value))
        daily_pnl = state.portfolio_value - prev_value
        daily_return = daily_pnl / prev_value if prev_value > 0 else 0
        state.last_pnl = daily_pnl
        state.total_return_pct = (state.portfolio_value / INITIAL_CAPITAL - 1) * 100
        state.daily_returns.append(daily_return)
        state.day_count += 1

        positions = await broker.get_positions()
        state.last_positions = [
            {
                "symbol": p["symbol"],
                "qty": float(p["qty"]),
                "market_value": float(p["market_value"]),
                "unrealized_pl": float(p["unrealized_pl"]),
                "unrealized_plpc": float(p.get("unrealized_plpc", 0)),
            }
            for p in positions
        ]

        snapshot = {
            "date": today,
            "portfolio_value": state.portfolio_value,
            "cash": float(account.get("cash", 0)),
            "equity": float(account.get("equity", 0)),
            "daily_pnl": daily_pnl,
            "daily_return_pct": daily_return * 100,
            "total_return_pct": state.total_return_pct,
            "positions_count": len(positions),
            "regime": state.last_regime,
            "strategy_weights": state.last_weights,
            "positions": state.last_positions,
            "risk_metrics": {
                "gross_leverage": target.gross_leverage,
                "net_leverage": target.net_leverage,
                "expected_vol": target.expected_volatility,
                "var_99": target.var_99,
            },
        }
        log_daily_snapshot(snapshot)

        state.run_log.append({
            "date": str(today),
            "pnl": round(daily_pnl, 2),
            "return_pct": round(daily_return * 100, 2),
            "trades": trades_executed,
            "regime": state.last_regime,
            "positions": len(positions),
        })

        # 9. Discord report
        sharpe = 0
        if len(state.daily_returns) >= 5:
            rets = np.array(state.daily_returns)
            sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

        report = (
            f"**Day {state.day_count}/30**\n"
            f"Portfolio: ${state.portfolio_value:,.2f}\n"
            f"Daily P&L: ${daily_pnl:+,.2f} ({daily_return:+.2%})\n"
            f"Total Return: {state.total_return_pct:+.2f}%\n"
            f"Sharpe (est): {sharpe:.2f}\n"
            f"Regime: {state.last_regime}\n"
            f"Positions: {len(positions)} | Trades: {trades_executed}\n"
            f"Gross Lev: {target.gross_leverage:.2f} | Net Lev: {target.net_leverage:.2f}\n"
            f"Weights: {state.last_weights}"
        )
        await send_discord_report(report)

        elapsed = (datetime.now(timezone.utc) - run_start).total_seconds()
        logger.info("Pipeline complete in %.1fs — P&L: $%+.2f (%+.2f%%)",
                     elapsed, daily_pnl, daily_return * 100)

    except Exception as e:
        logger.error("Pipeline failed: %s", e, exc_info=True)
        await send_discord_report(f"**PIPELINE ERROR**\n{str(e)[:500]}")
    finally:
        state.is_running = False


# ============================================================
# FASTAPI APPLICATION
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    await broker.connect()

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        run_daily_pipeline,
        "cron",
        hour=SCHEDULE_HOUR,
        minute=SCHEDULE_MINUTE,
        day_of_week="mon-fri",
        timezone="US/Eastern",
    )
    scheduler.start()
    state.scheduler = scheduler
    logger.info("Scheduler started: daily at %d:%02d ET (Mon-Fri)", SCHEDULE_HOUR, SCHEDULE_MINUTE)

    yield

    # Shutdown
    scheduler.shutdown()
    await broker.disconnect()

app = FastAPI(
    title="APEX Paper Trader",
    description="Multi-strategy paper trading runner with live dashboard",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health():
    return {
        "status": "running",
        "day_count": state.day_count,
        "last_run": str(state.last_run),
        "portfolio_value": state.portfolio_value,
        "is_running": state.is_running,
    }


@app.post("/run-now")
async def run_now():
    """Manually trigger a pipeline run."""
    if state.is_running:
        return {"status": "already_running"}
    asyncio.create_task(run_daily_pipeline())
    return {"status": "started"}


@app.get("/positions")
async def positions():
    return state.last_positions


@app.get("/history")
async def history():
    return state.run_log[-30:]


@app.get("/weights")
async def weights():
    return state.last_weights


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Live dashboard showing positions, P&L, and strategy weights."""
    positions_html = ""
    total_unrealized = 0
    for p in state.last_positions:
        pnl = p.get("unrealized_pl", 0)
        total_unrealized += pnl
        color = "#4CAF50" if pnl >= 0 else "#F44336"
        positions_html += f"""
        <tr>
            <td><strong>{p['symbol']}</strong></td>
            <td>{p['qty']:.0f}</td>
            <td>${p['market_value']:,.2f}</td>
            <td style="color:{color};">${pnl:+,.2f}</td>
            <td style="color:{color};">{p.get('unrealized_plpc',0)*100:+.2f}%</td>
        </tr>"""

    weights_html = ""
    for name, w in state.last_weights.items():
        bar_width = int(w * 400)
        weights_html += f"""
        <div style="margin:4px 0;">
            <span style="display:inline-block;width:220px;">{name}</span>
            <div style="display:inline-block;width:{bar_width}px;height:18px;background:#2196F3;border-radius:3px;"></div>
            <span style="margin-left:8px;">{w:.1%}</span>
        </div>"""

    history_html = ""
    for entry in reversed(state.run_log[-15:]):
        color = "#4CAF50" if entry["pnl"] >= 0 else "#F44336"
        history_html += f"""
        <tr>
            <td>{entry['date']}</td>
            <td style="color:{color};">${entry['pnl']:+,.2f}</td>
            <td style="color:{color};">{entry['return_pct']:+.2f}%</td>
            <td>{entry['trades']}</td>
            <td>{entry['regime']}</td>
            <td>{entry['positions']}</td>
        </tr>"""

    sharpe = 0
    if len(state.daily_returns) >= 5:
        rets = np.array(state.daily_returns)
        sharpe = (rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

    pnl_color = "#4CAF50" if state.last_pnl >= 0 else "#F44336"
    total_color = "#4CAF50" if state.total_return_pct >= 0 else "#F44336"

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>APEX Paper Trader</title>
    <meta http-equiv="refresh" content="60">
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
               background: #0D1117; color: #C9D1D9; margin: 0; padding: 20px; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #58A6FF; margin-bottom: 5px; }}
        .subtitle {{ color: #8B949E; margin-bottom: 20px; }}
        .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }}
        .card {{ background: #161B22; border: 1px solid #30363D; border-radius: 8px; padding: 16px; }}
        .card-title {{ color: #8B949E; font-size: 12px; text-transform: uppercase; }}
        .card-value {{ font-size: 28px; font-weight: bold; margin-top: 5px; }}
        .green {{ color: #4CAF50; }}
        .red {{ color: #F44336; }}
        .blue {{ color: #58A6FF; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{ text-align: left; padding: 8px 12px; color: #8B949E; border-bottom: 1px solid #30363D;
              font-size: 12px; text-transform: uppercase; }}
        td {{ padding: 8px 12px; border-bottom: 1px solid #21262D; }}
        .section {{ margin-top: 25px; }}
        .section-title {{ color: #58A6FF; font-size: 18px; margin-bottom: 10px; }}
        .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px;
                  font-size: 12px; font-weight: bold; }}
        .badge-regime {{ background: #1F2937; color: #F59E0B; }}
        .refresh {{ color: #8B949E; font-size: 12px; }}
    </style>
</head>
<body>
<div class="container">
    <h1>APEX Paper Trader</h1>
    <div class="subtitle">
        Day {state.day_count}/30 |
        Regime: <span class="badge badge-regime">{state.last_regime or 'N/A'}</span> |
        <span class="refresh">Auto-refreshes every 60s</span>
    </div>

    <div class="grid">
        <div class="card">
            <div class="card-title">Portfolio Value</div>
            <div class="card-value blue">${state.portfolio_value:,.2f}</div>
        </div>
        <div class="card">
            <div class="card-title">Today's P&L</div>
            <div class="card-value" style="color:{pnl_color};">${state.last_pnl:+,.2f}</div>
        </div>
        <div class="card">
            <div class="card-title">Total Return</div>
            <div class="card-value" style="color:{total_color};">{state.total_return_pct:+.2f}%</div>
        </div>
        <div class="card">
            <div class="card-title">Est. Sharpe</div>
            <div class="card-value blue">{sharpe:.2f}</div>
        </div>
    </div>

    <div class="grid" style="grid-template-columns: 1fr 1fr;">
        <div class="card section">
            <div class="section-title">Live Positions ({len(state.last_positions)})</div>
            <table>
                <tr><th>Symbol</th><th>Qty</th><th>Value</th><th>P&L</th><th>%</th></tr>
                {positions_html}
                <tr style="border-top:2px solid #30363D;">
                    <td><strong>TOTAL</strong></td><td></td><td></td>
                    <td style="color:{'#4CAF50' if total_unrealized>=0 else '#F44336'};">
                        <strong>${total_unrealized:+,.2f}</strong></td><td></td>
                </tr>
            </table>
        </div>
        <div class="card section">
            <div class="section-title">Strategy Weights</div>
            {weights_html if weights_html else '<p style="color:#8B949E;">No weights yet — run pipeline first</p>'}
        </div>
    </div>

    <div class="card section">
        <div class="section-title">Daily P&L History</div>
        <table>
            <tr><th>Date</th><th>P&L</th><th>Return</th><th>Trades</th><th>Regime</th><th>Positions</th></tr>
            {history_html if history_html else '<tr><td colspan="6" style="color:#8B949E;">No history yet</td></tr>'}
        </table>
    </div>

    <div style="margin-top:20px; color:#8B949E; font-size:12px;">
        Last run: {state.last_run or 'Never'} |
        Schedule: {SCHEDULE_HOUR}:{SCHEDULE_MINUTE:02d} ET Mon-Fri |
        <a href="/run-now" style="color:#58A6FF;">Trigger Manual Run</a> |
        <a href="/health" style="color:#58A6FF;">Health Check</a>
    </div>
</div>
</body>
</html>"""
    return HTMLResponse(content=html)
