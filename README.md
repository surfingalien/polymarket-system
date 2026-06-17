# Polymarket + Kalshi AI Trading Bot

Production-ready prediction market trading bot powered by Claude AI, with Bayesian probability estimation, Kelly Criterion sizing, order-book microstructure analysis, momentum signals, and cross-platform arbitrage detection.

## Architecture

```
polymarket_kalshi_ai_bot/
├── main.py                       ← Unified bot (both platforms, shared AI brain)
├── polymarket_bot/
│   ├── polymarket_client.py      ← CLOB V2 integration (post Apr 2026 migration)
│   └── main.py                   ← Polymarket-only runner
├── kalshi_bot/
│   ├── kalshi_client.py          ← V2 API + RSA-PSS SHA-256 auth
│   └── main.py                   ← Kalshi-only runner
├── shared/
│   ├── claude_agent.py           ← Claude AI (single + batch analysis, arb research)
│   ├── predictive_models.py      ← Bayesian, Kelly, momentum, ensemble, calibration
│   ├── market_analyzer.py        ← Full analysis pipeline orchestrator
│   ├── risk_manager.py           ← Position sizing, stop-losses, daily limits
│   ├── signal_router.py          ← Routes signals to executors
│   └── news_fetcher.py           ← Tavily/NewsAPI aggregation with caching
├── config/
│   ├── settings.py               ← Typed Pydantic configuration
│   └── .env.example              ← Environment template
├── scripts/
│   ├── setup_kalshi_keys.py      ← RSA-2048 key pair generation
│   └── run_backtest.py           ← Strategy backtesting with Sharpe/Brier/ECE
└── tests/                        ← 51 tests, all passing
```

## Predictive Algorithm Stack

### 1. Claude AI Analysis (`shared/claude_agent.py`)
- Calls `claude-opus-4-8` with structured JSON prompts for probability estimation
- Batch mode: analyzes up to 5 markets per API call
- Response caching with configurable TTL
- Graceful fallback to market price on API errors

### 2. Bayesian Probability Estimator (`BayesianEstimator`)
- Beta-Binomial conjugate prior initialized from the market price
- Updates posterior with AI analysis + news sentiment evidence
- Returns posterior mean + 95% credible interval
- Evidence weighted by confidence score

### 3. Kelly Criterion Position Sizer (`KellyCriterion`)
- Full Kelly formula: `f* = (b·p − q) / b`
- Handles both YES (positive edge) and NO (negative edge) bets
- Fractional Kelly (default 0.25×) caps volatility
- Hard cap at `RISK_MAX_KELLY_FRACTION`

### 4. Order Book Microstructure (`OrderBookAnalyzer`)
- Order Book Imbalance: `(bid_vol − ask_vol) / total_vol`
- Depth asymmetry at levels 2–3
- Effective spread quality scoring
- Slippage estimation for a given trade size

### 5. Momentum & Mean-Reversion (`MomentumAnalyzer`)
- EMA crossover (configurable short/long window)
- RSI-based overbought/oversold (mean-reversion)
- VWAP deviation signal
- Price velocity (recent rate of change)

### 6. Time-to-Resolution Decay (`ResolutionDecayModel`)
- Confidence decays logarithmically as resolution approaches
- Near-resolution discount for extreme prices (already well-discovered)
- Pulls estimates toward neutral when < 2 days remain

### 7. Ensemble Predictor (`EnsemblePredictor`)
- Weighted combination of all signals
- Default weights: AI (3.0), Bayesian (2.0), OBI (1.5), Momentum (1.0), Sentiment (1.2)
- Online weight updates based on signal performance (`update_weights`)
- Outputs: estimated probability, edge, Kelly fraction, confidence

### 8. Calibration Tracker (`CalibrationTracker`)
- Brier score: mean squared error of probability estimates
- Expected Calibration Error (ECE) across probability buckets
- Accuracy stats for model monitoring

### 9. Cross-Platform Arbitrage (`CrossMarketCorrelator`)
- Jaccard word-similarity matching of market questions
- Net spread calculated after Polymarket (2%) + Kalshi (7%) fees
- Sorted by net spread; AI verifies equivalence before execution

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp config/.env.example .env
# Fill in your credentials
```

### 3. Generate Kalshi RSA keys
```bash
python scripts/setup_kalshi_keys.py
# Upload keys/kalshi_public.pem to Kalshi dashboard
# Set KALSHI_API_KEY_ID in .env
```

### 4. Run backtest (no credentials needed)
```bash
python scripts/run_backtest.py --source mock --n 500
```

### 5. Run in mock mode (safe)
```bash
MOCK_MODE=true python main.py
```

### 6. Run individual platform
```bash
MOCK_MODE=true python -m polymarket_bot.main
MOCK_MODE=true python -m kalshi_bot.main
```

### 7. Live trading (after extensive testing)
```bash
MOCK_MODE=false RISK_MAX_POSITION_SIZE_USD=25 python main.py
```

## Key Configuration

| Variable | Default | Description |
|---|---|---|
| `MOCK_MODE` | `true` | Global kill switch — no real orders |
| `ANTHROPIC_API_KEY` | — | Claude API key |
| `POLY_PRIVATE_KEY` | — | Polymarket wallet private key |
| `KALSHI_API_KEY_ID` | — | Kalshi API key ID |
| `RISK_MAX_POSITION_SIZE_USD` | 50 | Max per-trade size in USD |
| `RISK_DAILY_LOSS_LIMIT_USD` | 100 | Auto-halt daily loss threshold |
| `RISK_MIN_EDGE_PCT` | 0.05 | Minimum 5% edge to trade |
| `RISK_MIN_CONFIDENCE_SCORE` | 0.65 | Minimum AI confidence threshold |
| `RISK_MAX_KELLY_FRACTION` | 0.25 | Never exceed 25% of full Kelly |

## Running Tests

```bash
pytest tests/ -v
pytest tests/ --cov=shared --cov-report=term-missing
```

## Keeping the dashboard awake

Streamlit Community Cloud puts an app to sleep after ~7 days without traffic.
Two ways to keep it always-on:

1. **In-repo (included):** `.github/workflows/keep-alive.yml` pings the app
   every 10 minutes. Set the app URL once under **Settings → Secrets and
   variables → Actions → Variables**: add `STREAMLIT_APP_URL =
   https://your-app.streamlit.app`. (GitHub disables scheduled workflows after
   60 days of no repo activity — any push re-enables them.)
2. **External monitor (most reliable):** point [UptimeRobot](https://uptimerobot.com)
   or [cron-job.org](https://cron-job.org) at the same URL on a 5-minute
   interval. This survives repo inactivity and also alerts you if the app is
   actually down.

## Authentication Notes

| Platform | Auth Method | Key Change |
|---|---|---|
| Polymarket | HMAC-SHA256 on `ts+method+path+body` | CLOB V2: no `feeRateBps` in orders (Apr 28 2026) |
| Kalshi | RSA-PSS SHA-256 on `ts+method+path` | External API host; V2 orders only (May 2026+) |

**Both bots default to `mock_mode=true`.** Never set `MOCK_MODE=false` without weeks of mock testing, verified API connections, confirmed risk limits, and starting with small position sizes ($5–$10).
