# RL Trading Agents

Deep reinforcement learning agents for multi-stock portfolio management on S&P 500 equities.
Seven agents are trained across two action space formulations — continuous portfolio weights
and discrete buy/hold/sell — and evaluated against an equal-weight buy-and-hold portfolio and
the S&P 500 index.

## Repository layout

```
download_top100.py              # yfinance download: top 100 S&P 500 tickers (OHLCV)
download_sp500_index.py         # yfinance download: ^GSPC index series (baseline)
1_data_preparation.ipynb        # feature engineering -> *_transformed.csv
2_trading_agent_for_colab.ipynb # environments, agents, training, evaluation (Colab)

sp100/                          # raw top-100 data
sp100_transformed/              # top-100 data + 12 engineered features (model input)
sp500/                          # full-universe source data + index series
sp500_transformed/              # earlier full-universe feature run (not used by the notebooks)
```

CSVs are tracked with Git LFS (see `.gitattributes`). Training artifacts
(`models/`, `results/`, `plots/`, `smoke_out/`) are gitignored.

## Data

| File | Contents |
| --- | --- |
| `sp500/sp500_companies.csv` | 502 companies with sector, market cap and index weight |
| `sp500/sp500_stocks.csv` | 502 symbols, 1,891,536 rows, 2010-01-04 → 2024-12-20 |
| `sp500/sp500_index.csv` | ^GSPC daily close, 2,773 rows, 2014-12-22 → 2025-12-31 |
| `sp100/sp500_companies_top100.csv` | top 100 companies by index weight |
| `sp100/sp500_stocks_top100.csv` | 99 symbols, 384,841 rows, 2010-01-04 → 2025-12-31 |
| `sp100_transformed/sp500_stocks_top100_transformed.csv` | same + 12 features, 365,140 rows, 2010-10-18 → 2025-12-31 |
| `sp500_transformed/` | 171 symbols through 2024-12-20, superseded by `sp100_transformed/` |

`FI` is requested in the top-100 list but has no usable price history in the yfinance
download, which is why 100 companies yield 99 symbols. The transformed file starts later
than the raw one because rows without a complete 200-day moving average are dropped.

Note: `1_data_preparation.ipynb` writes its output into `sp100/`; the committed copies live
in `sp100_transformed/`, which is also where the downstream cell reads them from. Move the
two generated files after regenerating them.

## Pipeline

### 1. Download (`download_top100.py`, `download_sp500_index.py`)

`download_top100.py` ranks `sp500/sp500_companies.csv` by index weight, takes the top 100
tickers and downloads 2010-01-01 → 2025-12-31 OHLCV from yfinance, reshaping the wide frame
into long format (`Date, Symbol, Adj Close, Close, High, Low, Open, Volume`). It prints a
per-symbol coverage summary and flags late starters (IPO/spinoff after 2010-01-04).

`download_sp500_index.py` downloads the ^GSPC close series for the index baseline and writes
a two-column `Date, S&P500` file matching the format of `sp500/sp500_index.csv`. It is meant
to run on Colab and triggers a browser download of the result.

### 2. Feature engineering (`1_data_preparation.ipynb`)

Computes 12 technical features per symbol from `Adj Close`, all stationary and
scale-independent so they can be compared across stocks:

| Group | Features |
| --- | --- |
| Returns | `return_1d`, `return_5d`, `return_20d` |
| Trend | `ma20_diff`, `ma50_diff`, `ma200_diff` (relative distance of price to MA) |
| Volatility | `volatility_20`, `rsi_14` (Wilder smoothing), `range_pct`, `atr_pct` |
| Volume | `vol_ratio`, `vol_z_20` (vs. 20-day mean / z-score) |

Raw MA levels, absolute ATR and the 20-day volume average are intermediate only — they are
price-scale dependent and get dropped. Rows where any feature is still in its warm-up window
are removed.

### 3. Training and evaluation (`2_trading_agent_for_colab.ipynb`)

Runs on Google Colab: mounts Drive and reads the transformed CSVs from
`/content/drive/MyDrive/TradingAgent/`. `gymnasium` and `stable_baselines3[extra]` are
pip-installed inline.

## Environments

Both environments share the same observation layout, reward and cost model:

- **Observation** — `[features (N*F) | current weights (N) | cash ratio (1)]`; with N=50 and
  F=12 that is a 651-dim `Box`.
- **Reward** — `log(portfolio_value_next / portfolio_value_current)`.
- **Costs** — proportional fee of `cost_bps` (default 5 bps) on every traded dollar.
- **Episodes** — fixed 504-day windows starting at a random date, so training covers diverse
  market regimes. For evaluation the episode length is set to the full test period and the
  start index is pinned to 0.

**`MultiStockTradingEnv`** (continuous) — `Box(-1, 1, shape=(N,))`, one value per stock.
`WEIGHT_MODE = "direct"` (active) treats the action as the target weight itself: negatives
clip to zero, and if the requested weights sum above 1 they are scaled down proportionally
(no leverage), with the remainder staying in cash. This reaches exactly 100% in one stock and
exactly 100% cash. `WEIGHT_MODE = "softmax"` is kept as the earlier variant — a softmax over
the logits with a fixed cash logit, which caps any single weight at 12.5% for N=50.

**`DiscreteMultiStockTradingEnv`** — `MultiDiscrete([3] * N)`: `0 = sell` (liquidate the whole
position), `1 = hold`, `2 = buy` (invest an equal share of available cash). Execution is
two-phase, all sells before all buys, so proceeds can be reinvested in the same step.

## Agents

| Agent | Action space | Implementation |
| --- | --- | --- |
| PPO | continuous | Stable-Baselines3, 4 vectorized envs, `lr=1e-4`, `clip_range=0.2`, `ent_coef=0.05` |
| A2C | continuous | Stable-Baselines3, 4 vectorized envs, `lr=1e-4`, `ent_coef=0.05`, `n_steps=10` |
| SAC | continuous | Stable-Baselines3, single env, `lr=1e-4`, `learning_starts=5000`, `batch_size=512`, `tau=0.005`, `ent_coef="auto"` |
| PPO | discrete | Stable-Baselines3, same hyperparameters as the continuous variant |
| A2C | discrete | Stable-Baselines3, same hyperparameters as the continuous variant |
| BDQ | discrete | custom PyTorch — branching dueling DQN |
| REINFORCE | discrete | custom PyTorch — Monte Carlo policy gradient |

All agents use a `[512, 512, 256]` MLP.

**BDQ** (Tavakoli et al., AAAI 2018) avoids the 3^50 joint action space with a shared feature
extractor, one value head and one 3-way advantage head per stock, combined as
`Q = V + (A - mean(A))`. Trained with a 100k replay buffer, `batch_size=512`, target update
every 1000 steps, `train_freq=4`, and linear ε-decay from 1.0 to 0.02 over the first 20% of
timesteps.

**REINFORCE** samples one categorical action per stock from a single batched distribution,
sums the log-probabilities, and updates on full episodes with normalized discounted returns
(`gamma=0.99`) and gradient clipping at 0.5. No baseline, no critic.

## Experiment configuration

- **Universe** — 50 stocks, the top 50 by market cap, excluding `PLTR` and `GEV` (too short a
  price history).
- **Period** — train 2013-10-16 → 2022-12-31, test 2023-01-01 → 2025-12-31. The start date is
  the earliest day on which all 50 stocks have complete features (bottleneck: the ABBV IPO in
  2013 plus the 200-day MA warm-up).
- **Budget** — `TIMESTEPS = 1_000_000` per agent per seed, `SEEDS = [0, 1, 2]` → 21 runs.
- **Capital** — $10,000 initial, 5 bps transaction costs.
- **Seeding** — `set_global_seeds` covers Python, NumPy and PyTorch; SB3 additionally receives
  the seed via the model constructor and `make_vec_env`.
- **Smoke test** — `SMOKE_TEST = True` drops to 100k timesteps and one seed to validate the
  full train → evaluate → write path in minutes.
- **Resumability** — every finished `(agent, seed)` appends a row to `metrics_per_seed.csv`,
  which is reloaded on re-run so completed runs are skipped. Each run is wrapped in
  `try/except`, so one failure does not abort the batch.

## Evaluation

Trained models are rolled out deterministically over the full test period and compared
against two baselines: an equal-weight buy-and-hold portfolio over the same 50 stocks, and
the S&P 500 index normalized to the same starting capital. Baselines are single deterministic
curves, so their std is 0.

Performance metrics: cumulative return, annualized return, volatility, Sharpe ratio, max
drawdown, Calmar ratio. Trading behavior metrics: average daily turnover, effective number of
positions (inverse Herfindahl-Hirschman index of the renormalized equity weights) and the
share of days held fully in cash. Both are reconstructed from the recorded daily share
holdings, so they are comparable across agents and baselines.

## Outputs

Written to `MyDrive/TradingAgent/`:

```
models/<agent>_seed<seed>.zip        # SB3 agents
models/<agent>_seed<seed>.pt         # BDQ, REINFORCE

results/metrics_per_seed.csv         # one row per (agent, seed), all metrics
results/curve_<agent>_seed<seed>.csv # daily equity curve per run
results/learncurve_<agent>_seed<seed>.csv
results/metrics_summary.csv          # aggregated mean +/- std over seeds
results/results_table.tex            # performance table
results/turnover_table.tex           # trading behavior table

plots/evaluation_all_models.png      # portfolio value + drawdown, mean over seeds
plots/learning_curves.png            # training episode return vs. timesteps
```

The learning-curve figure pools all seeds, bins by timestep and plots one mean curve per
agent — a curve that flattens before the budget is exhausted indicates convergence.

## Running it

1. *(optional)* Refresh the raw data: `python download_top100.py`, and
   `download_sp500_index.py` on Colab for the index series.
2. Run `1_data_preparation.ipynb` locally, then move the two generated
   `*_transformed.csv` files into `sp100_transformed/`.
3. Upload `sp500_companies_top100_transformed.csv`, `sp500_stocks_top100_transformed.csv` and
   `sp500_index.csv` to `MyDrive/TradingAgent/`.
4. Run `2_trading_agent_for_colab.ipynb` — start with `SMOKE_TEST = True` to verify the
   pipeline, then set it back to `False` for the full run.

## Dependencies

- [Gymnasium](https://gymnasium.farama.org/) — RL environment API
- [Stable Baselines3](https://stable-baselines3.readthedocs.io/) — PPO, A2C, SAC
- [PyTorch](https://pytorch.org/) — BDQ and REINFORCE implementations
- [yfinance](https://github.com/ranaroussi/yfinance) — data download
- Pandas / NumPy — data processing
- Matplotlib — evaluation plots
