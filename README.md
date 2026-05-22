# AI-Quant — Taiwan Stock Market Prediction with Deep Learning

> A multi-timeframe deep learning system that ranks Taiwan Stock Exchange (TWSE) equities by their likelihood of achieving **+12% price appreciation within a 20-day forward window**. Combines an LSTM-Transformer ensemble across four time horizons with a market-level regime gate to determine whether conditions favor active trading.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Model Architecture](#2-model-architecture)
3. [Requirements](#3-requirements)
4. [Installation](#4-installation)
5. [Usage](#5-usage)
   - [5.1 Generate Stock Metadata](#51-step-1--generate-stock-metadata)
   - [5.2 Train the Model](#52-step-2--train-the-model)
   - [5.3 Run Inference](#53-step-3--run-inference)
6. [Understanding the Output](#6-understanding-the-output)
7. [Risk Management & Strategy](#7-risk-management--strategy)
8. [Known Limitations](#8-known-limitations)
9. [Closing Remarks](#9-closing-remarks)
10. [License](#10-license)

---

## 1. Overview

AI-Quant fetches historical price and volume data for a user-defined list of TWSE ticker symbols, trains a GPU-accelerated deep learning model across multiple time horizons, and produces a ranked list of stocks with scores reflecting their likelihood of a +12% price gain within 20 trading days.

**Key design decisions:**

- **Multi-horizon ensemble** — Four separate models are trained on half-year, one-year, two-year, and three-year historical windows. Their outputs are combined via fixed weights (`0.25 / 0.55 / 0.15 / 0.05`) to produce a single final score per stock.

- **Alpha calibration** — Each horizon model applies a calibration exponent (alpha) tuned during validation via simulated backtesting. This sharpens or softens the score distribution to maximize risk-adjusted PnL on the validation set.

- **Market regime gate** — A global "Market Score" (the mean score across all stocks) is compared against a threshold (default: `0.52`). If the market is deemed unfavorable, trading signals are suppressed entirely.

- **Labeling logic** — A sample is labeled positive (`1`) if the closing price hits **+12%** above the entry price at any point within the 20-day window, before hitting the **−6%** stop-loss. Otherwise it is labeled negative (`0`).

- **Scoped universe** — The default stock universe covers TWSE-listed technology equities. OTC (`上櫃`) stocks are excluded due to data extraction issues (see [Known Limitations](#8-known-limitations)).

---

## 2. Model Architecture

The core model is `LSTM_CondTransformer`, a hybrid sequence model defined in `model.py`. It processes a fixed-length lookback window of **40 trading days** and outputs a single logit (binary classification).

### Input Features (7 per timestep)

| # | Feature | Description |
|---|---------|-------------|
| 1 | Price | Normalized to the last closing price in the window |
| 2 | Log Return | Day-over-day log price change |
| 3 | Volume Z-score | Rolling 20-day mean/std normalization |
| 4 | Volatility | Rolling 20-day std of log returns |
| 5 | Momentum | 10-day price rate of change |
| 6 | Month Feature | Sin encoding of calendar month |
| 7 | Weekday Feature | Cos encoding of day of week |

### Architecture

```
Input (B, 40, 7)
      │
      ▼
┌─────────────┐
│ LSTM Encoder│  hidden=128, layers=1
└─────────────┘
      │
      ▼
┌──────────────────┐
│ Cross-Attention  │  Query=LSTM output, Key/Value=projected raw input
└──────────────────┘
      │
      ▼
┌──────────────────────┐
│ Transformer Encoder  │  2 layers, 4 heads, FF dim=256
│ + Positional Embed   │
└──────────────────────┘
      │
   last token
      │
      ▼
┌─────────────────────────────────┐
│ Head: Linear(128→32) → ReLU    │
│       → Dropout → Linear(32→1) │
└─────────────────────────────────┘
      │
   logit (sigmoid at inference)
```

### Training Details

| Parameter | Value |
|-----------|-------|
| Loss | `BCEWithLogitsLoss` with `pos_weight=2.0` |
| Label smoothing | Targets shifted from `{0,1}` → `{0.05, 0.95}` |
| Optimizer | AdamW (`lr=3e-4`) |
| Scheduler | ReduceLROnPlateau (`factor=0.9`, `patience=10`, `min_lr=1e-5`) |
| Gradient clipping | Max norm `1.0` |
| Early stopping | Patience of 50 epochs (based on validation PnL) |
| Data split | 70% train / 15% val / 15% test |

---

## 3. Requirements

### Software

- Python 3.10+
- CUDA-compatible GPU — recommended VRAM: **10–12 GB minimum** *(tested on NVIDIA RTX 4060 Ti 16 GB)*
- PyTorch (CUDA build — see [Installation](#4-installation))

### Python Dependencies

Installed via `requirements.txt`:

```
numpy
pandas
tqdm
twstock
```

### Hardware Note

Training runs **four independent models** (one per time horizon). Total training time is approximately **2–3 hours** on a single RTX 4060 Ti. If your GPU has less than 10 GB of VRAM, reduce the batch size (see [Step 2](#52-step-2--train-the-model)). Note that doing so without adding gradient accumulation will alter training dynamics.

---

## 4. Installation

**Step 1 — Clone the repository**

```bash
git clone https://github.com/Wilson-LL/AI-Quant.git
cd AI-Quant
```

**Step 2 — Create and activate a virtual environment**

```bash
python -m venv .venv
```

```bash
# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**Step 3 — Install PyTorch (CUDA build)**

Visit the official selector to generate the right command for your OS and CUDA version:
👉 https://pytorch.org/get-started/locally/

Example used in the reference environment *(Windows, CUDA 13.2 nightly)*:

```bash
pip3 install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu132
```

> **⚠️ Important:** Install PyTorch **before** running `pip install -r requirements.txt`. Installing in the wrong order can result in a CPU-only PyTorch build being pulled in as a transitive dependency.

**Step 4 — Install remaining dependencies**

```bash
pip install -r requirements.txt
```

---

## 5. Usage

### 5.1 Step 1 — Generate Stock Metadata

**Script:** `generate_stocks_json.py`

This script validates the configured ticker symbols against the `twstock` registry and writes a `stocks.json` metadata file to `./checkpoints/`. This file is consumed by both `train.py` and `inference.py` to determine which stocks to include and which have been marked valid after data fetching.

Open `generate_stocks_json.py` and modify `STOCK_IDS` to define your target universe:

```python
STOCK_IDS = [
    "6770", "3481", "2344", "2485", "2367", "2337",
    "2409", "4989", "2408", "2317", "3189", "2303",
    "2313", "1785", "3006", "6182", "3576", "3049",
    "3105", "2369", "1605", "2399", "2329", "6147",
    "8096", "2353", "8021", "2495", "6443", "4958",
    "4967", "2338", "0052", "2406", "5498", "3231",
    "3702", "6282", "2455", "2327", "6274", "3714",
    "5483", "2489", "4906", "4979", "6213", "2330",
    "8112", "4919", "3701", "0050", "2324", "2301",
    "3036", "3702", "3008", "2345", "2357", "2454",
    "1303", "1519", "2467", "8064", "5536", "2308",
    "6187", "2376", "6150", "2377", "3219", "2425",
    "3661", "3515", "3540", "5386", "2481", "3491",
    "2360", "2347", "3048",
]
```

Then run:

```bash
python generate_stocks_json.py
```

This produces `./checkpoints/stocks.json`. Any ticker not found in the `twstock` registry is skipped with a warning. Stocks that fail data fetching during training are automatically flagged `valid: false` and excluded from inference.

> **Tuning tip:** Adding more tickers increases dataset size but requires the model to generalize across a wider universe, which may reduce per-stock accuracy. Start with a focused list if predictive precision is a priority.

---

### 5.2 Step 2 — Train the Model

**Script:** `train.py`

```bash
python train.py
```

The script trains **four independent models**, one per time horizon:

| Horizon | History Used | Ensemble Weight |
|---------|-------------|-----------------|
| `half_year` | ~6 months | 0.25 |
| `one_year` | ~12 months | 0.55 |
| `two_year` | ~24 months | 0.15 |
| `three_year` | ~36 months | 0.05 |

Checkpoints are saved to `./checkpoints/<horizon>/`. Both `latest.pt` and `best.pt` (highest validation PnL) are preserved per horizon.

**Key parameters** *(configurable at the top of `train.py`)*:

```python
BATCH_SIZE         = 64      # Reduce if GPU OOM (e.g., 32 or 16)
EPOCHS             = 1000    # Subject to early stopping (patience=50)
LOOKBACK_WINDOW    = 40      # Input sequence length (trading days)
PREDICTION_HORIZON = 20      # Forward window for labeling (trading days)
TAKE_PROFIT        = 0.12    # +12% target used in labeling and backtest
STOP_LOSS          = 0.06    # -6% stop used in labeling and backtest
LEARNING_RATE      = 3e-4
RESUME             = False   # Set True to resume from latest checkpoint
```

> **⚠️ Warning:** The current implementation does not include gradient accumulation. Reducing `BATCH_SIZE` below `64` will alter effective gradient updates and may produce results that differ from the reference training environment.

---

### 5.3 Step 3 — Run Inference

**Script:** `inference.py`

After all four horizon models have been trained, generate today's predictions:

```bash
python inference.py
```

The script loads the best checkpoint for each horizon, fetches recent price data for all valid stocks, constructs the feature sequence for today's date, and runs the ensemble. Only stocks marked `valid: true` in `stocks.json` are included.

**Key parameters** *(configurable at the top of `inference.py`)*:

```python
TOP_K             = 30    # Number of stocks to display in ranking
MARKET_THRESHOLD  = 0.52  # Minimum market score to enable trading signals
HORIZON_MODELS    = [
    {"name": "half_year",  "weight": 0.25},
    {"name": "one_year",   "weight": 0.55},
    {"name": "two_year",   "weight": 0.15},
    {"name": "three_year", "weight": 0.05},
]
```

---

## 6. Understanding the Output

**Example output:**

```
Market Score: 0.7115  ✅ Trade Enabled

Top 30:
Rank | Stock | Prob
   1 |  2406 | 0.9726
   2 |  3049 | 0.9625
   3 |  2345 | 0.9529
   4 |  2454 | 0.9431
   5 |  6770 | 0.9387
  ...

💰 Trading Signals:
2406 | prob=0.9726
3049 | prob=0.9625
2345 | prob=0.9529
...
```

### Field Descriptions

**`Market Score`**
The mean score across all valid stocks in the universe. Acts as a macro regime indicator. When this value clears the threshold (default: `0.52`), the system switches to **Trade Enabled** and emits the top 10 trading signals. Below the threshold, no signals are emitted.

**`Prob` (Score)**
The ensemble score for a given stock — a weighted combination of the four horizon models' sigmoid outputs, each raised to its calibrated alpha exponent before weighting.

> **⚠️ Important:** These are **NOT** calibrated statistical probabilities. A score of `0.97` does not imply a 97% chance of the +12% outcome. The alpha calibration and ensemble weighting distort the raw probability scale. **Treat these values as a relative ranking signal only.**

**`Top 30`**
The 30 highest-scoring stocks from the valid universe, displayed regardless of whether the market regime gate is open.

**`Trading Signals`**
The top 10 stocks from the Top 30, emitted only when trading is enabled. These represent the model's highest-conviction candidates for the session.

---

## 7. Risk Management & Strategy

The labeling and backtesting logic encodes an explicit exit rule:

| Signal | Threshold | Action |
|--------|-----------|--------|
| Take Profit | **+12%** | Exit when position gains 12% from entry |
| Stop Loss | **−6%** | Exit when position loses 6% from entry |

This asymmetric **2:1 reward-to-risk ratio** is baked into both the training labels and the validation backtest. Any live application of the model's signals should apply the same exit rules to remain consistent with the conditions under which the model was trained.

### Author's Personal Approach

All stocks in the default universe are technology-sector equities. When the Market Score is sufficiently high and trading is enabled, the author's preferred approach is to consider broader market positions in:

- **2330** — TSMC
- **0050** — Yuanta/FTSE TWSE Taiwan 50 ETF

These serve as diversified proxies for the technology sector rather than concentrating exposure in individual model signals.

> **Disclaimer:** This is not financial advice. Always conduct independent research and manage risk appropriately before making any investment decisions.

---

## 8. Known Limitations

**OTC (`上櫃`) Stock Incompatibility**
Data extraction for OTC-listed stocks (traded on the Taipei Exchange, as opposed to TWSE) does not function correctly. The root cause has not been identified. Problematic tickers are filtered out automatically via the `valid` flag in `stocks.json`; some OTC tickers may still appear in `STOCK_IDS` but will be excluded at runtime.

**No Gradient Accumulation**
The training loop does not implement gradient accumulation. Reducing `BATCH_SIZE` below `64` is a valid workaround for VRAM constraints, but it changes training dynamics and may degrade model performance compared to the reference setup.

**Scores Are Not Calibrated Probabilities**
Due to alpha exponentiation and ensemble weighting, output scores should be treated as ordinal rankings only. See [Section 6](#6-understanding-the-output) for details.

**Static Universe**
Models are trained on a fixed set of tickers. The model does not generalize to tickers absent from `STOCK_IDS` at training time without full retraining.

**Single-Stock Evaluation**
The model evaluates each stock independently, with no cross-sectional context. Correlations or sector-level dynamics between stocks are not explicitly modeled.

---

## 9. Closing Remarks

This is an independent research project and should be treated as such. Training results will vary across hardware, data availability, and market conditions — production-grade consistency is not guaranteed.

Experimentation is encouraged. Feel free to adjust any hyperparameters, modify the training pipeline, or swap in alternative architectures. If you have access to more powerful hardware, here are concrete directions for scaling up:

- Increase `LSTM_HIDDEN` and `TRANS_HIDDEN` (e.g., `256` or `512`)
- Add more `LSTM_LAYERS` or `TRANS_LAYERS`
- Increase `TRANS_FF` (e.g., `512` or `1024`)
- Expand `TRANS_HEADS` (e.g., `8`)
- Add gradient accumulation to support larger effective batch sizes
- Extend `LOOKBACK_WINDOW` for longer temporal context

Contributions, forks, and feedback are welcome.

---

## 10. License

See [LICENSE](./LICENSE) file in the repository root for full terms of use.

---

<div align="center">
  <sub>AI-Quant &nbsp;·&nbsp; <a href="https://github.com/Wilson-LL/AI-Quant">github.com/Wilson-LL/AI-Quant</a></sub>
</div>
