# Live Trading Runbook

## Pre-Flight Checklist

- [ ] Paper trading for >= 30 days with positive expectancy
- [ ] Run config validator: `cd TFT-main && python -m trading.config_validator`
  - All 10 checks must PASS (0 critical failures)
- [ ] Audit tables created: `python -c "from trading.persistence.audit import AuditLogger; AuditLogger().create_schema()"`
- [ ] Test Discord/email notifications manually
- [ ] Confirm Redis is running and accessible
- [ ] Review position sizing strategy and parameters
- [ ] Back up current `.env` file
- [ ] Confirm circuit breaker thresholds are appropriate for account size

## Capital Sizing Guidance

| Phase | Strategy | Max Position | Notes |
|-------|----------|-------------|-------|
| Week 1-2 | `fixed_fractional` | 1% risk/trade | Conservative start |
| Week 3-4 | `fixed_fractional` | 1-2% risk/trade | If Sharpe > 1.0 |
| Month 2+ | `kelly_criterion` | Half-Kelly, 5% cap | Only after calibration |
| Advanced | `volatility_scaled` | ATR-based | Requires reliable ATR data |

Never use full Kelly. Half-Kelly is the maximum.

## Paper vs Live Configuration

| Setting | Paper | Live |
|---------|-------|------|
| `TRADING_MODE` | `paper` | `live` |
| `ALPACA_API_KEY` | Paper key | Live key |
| `ALPACA_SECRET_KEY` | Paper secret | Live secret |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` | `https://api.alpaca.markets` |
| `CB_DRAWDOWN_METHODS` | `high_water_mark:5.0,start_of_day:3.0` | Same or tighter |
| `POSITION_RISK_PER_TRADE_PERCENT` | 1.0-2.0 | 0.5-1.0 |

## Switchover Procedure

1. **Stop** the trading engine completely
2. **Update** `.env` with live credentials and settings (see table above)
3. **Run** `python -m trading.config_validator` — must show 0 critical failures
4. **Create** audit schema if not already done
5. **Start** the engine and monitor the first trading session manually
6. **Verify** first few orders execute correctly on Alpaca dashboard
7. After 1 full day of successful live trading, enable automated scheduling

## Daily Monitoring Checklist

- [ ] Check circuit breaker status: Redis key `circuit_breaker:is_tripped` should be `false`
- [ ] Review `circuit_breaker_events` table for any overnight activity
- [ ] Verify portfolio value on Alpaca dashboard matches system state
- [ ] Check notification channels received expected alerts
- [ ] Review `portfolio_snapshots` table for SOD values
- [ ] Monitor system logs for repeated API errors

## Emergency Procedures

### Circuit Breaker Tripped

1. **Do not panic** — all positions have been closed automatically
2. Check `circuit_breaker_events` table for the trip reason
3. Check `circuit_breaker_closures` for per-position details
4. Review market conditions — was the trip justified?
5. If false positive: reset via `circuit_breaker.reset_breaker("your_name", "reason")`
6. If legitimate: investigate root cause before resetting

### Manual Emergency Stop

If the circuit breaker didn't fire but you need to stop:

1. Set Redis key: `SET circuit_breaker:is_tripped true`
2. Cancel all orders on Alpaca dashboard
3. Close all positions on Alpaca dashboard
4. Stop the trading engine process
5. Investigate and document the incident

### Rollback to Paper Trading

1. Stop the trading engine
2. Close all live positions on Alpaca dashboard
3. Restore paper `.env` settings (or use backup)
4. Run `python -m trading.config_validator`
5. Restart with paper configuration
6. Document what went wrong and remediation steps
