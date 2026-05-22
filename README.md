================================================================================
  AI-QUANT — Taiwan Stock Market Prediction with Deep Learning
  https://github.com/Wilson-LL/AI-Quant
================================================================================

  A multi-timeframe deep learning system that ranks Taiwan Stock Exchange (TWSE)
  equities by their probability of achieving +12% price appreciation within a
  20-day forward window. Combines an LSTM-Transformer ensemble across four time
  horizons with a market-level regime gate to determine whether conditions favor
  active trading.

--------------------------------------------------------------------------------
  TABLE OF CONTENTS
--------------------------------------------------------------------------------

  1. Overview
  2. Model Architecture
  3. Requirements
  4. Installation
  5. Usage
       5.1 Step 1 — Generate Stock Metadata   (generate_stocks_json.py)
       5.2 Step 2 — Train the Model           (train.py)
       5.3 Step 3 — Run Inference             (inference.py)
  6. Understanding the Output
  7. Risk Management & Strategy
  8. Known Limitations
  9. Closing Remarks
  10. License

--------------------------------------------------------------------------------
  1. OVERVIEW
--------------------------------------------------------------------------------

  AI-Quant fetches historical price and volume data for a user-defined list of
  TWSE ticker symbols, trains a GPU-accelerated deep learning model across
  multiple time horizons, and produces a ranked list of stocks with associated
  scores reflecting their likelihood of a +12% price gain within 20 trading days.

  Key design decisions:

    - Multi-horizon ensemble: Four separate models are trained on half-year,
      one-year, two-year, and three-year historical windows. Their outputs are
      combined via fixed weights (0.25 / 0.55 / 0.15 / 0.05) to produce a
      single final score per stock.

    - Alpha calibration: Each horizon model applies a calibration exponent
      (alpha) tuned during validation via simulated backtesting. This sharpens
      or softens the probability distribution to maximize risk-adjusted PnL on
      the validation set.

    - Market regime gate: A global "Market Score" (the mean probability across
      all stocks) is compared against a threshold (default: 0.52). If the market
      is deemed unfavorable, trading signals are suppressed entirely.

    - Labeling logic: A sample is labeled positive (1) if the closing price
      hits +12% above the entry price at any point within the 20-day window,
      before hitting the -6% stop-loss. Otherwise it is labeled negative (0).

    - Scoped universe: The default stock universe covers TWSE-listed technology
      equities. OTC ("上櫃") stocks are excluded due to data extraction issues.

--------------------------------------------------------------------------------
  2. MODEL ARCHITECTURE
--------------------------------------------------------------------------------

  The core model is LSTM_CondTransformer, a hybrid sequence model defined in
  model.py. It processes a fixed-length lookback window of 40 trading days and
  outputs a single logit (binary classification).

  Input Features (7 per timestep)
  --------------------------------
    1. Price (normalized to the last closing price in the window)
    2. Log return
    3. Volume Z-score  (rolling 20-day mean/std normalization)
    4. Volatility      (rolling 20-day std of log returns)
    5. Momentum        (10-day price rate of change)
    6. Month feature   (sin encoding of calendar month)
    7. Weekday feature (cos encoding of day of week)

  Architecture
  ------------
    LSTM Encoder
      Processes the input sequence and captures temporal dependencies.
      Hidden size: 128  |  Layers: 1

    Cross-Attention Layer
      Queries from LSTM hidden states attend to a projection of the raw input
      (Key/Value). This allows the model to re-weight raw features relative to
      the learned LSTM representation.

    Transformer Encoder
      Two-layer encoder (4 attention heads, FF dim: 256) applied on top of the
      cross-attended LSTM output, with learned positional embeddings.

    Classification Head
      Final hidden state → Linear(128→32) → ReLU → Dropout → Linear(32→1)
      Output is a raw logit; sigmoid is applied at inference time.

  Training Details
  ----------------
    - Loss: BCEWithLogitsLoss with pos_weight=2.0 to handle class imbalance
    - Label smoothing: targets shifted from {0,1} to {0.05, 0.95}
    - Optimizer: AdamW  (lr=3e-4)
    - Scheduler: ReduceLROnPlateau (factor=0.9, patience=10, min_lr=1e-5)
    - Gradient clipping: max norm 1.0
    - Early stopping: patience of 50 epochs (based on validation PnL)
    - Train / Val / Test split: 70% / 15% / 15%

--------------------------------------------------------------------------------
  3. REQUIREMENTS
--------------------------------------------------------------------------------

  Software
  --------
    - Python 3.10+
    - CUDA-compatible GPU (recommended VRAM: 10–12 GB minimum)
      Tested on: NVIDIA RTX 4060 Ti (16 GB)
    - PyTorch (CUDA build — see Installation for details)

  Python Dependencies (installed via requirements.txt)
  -----------------------------------------------------
    - numpy
    - pandas
    - tqdm
    - twstock

  Hardware Note
  -------------
    Training runs four separate models (one per time horizon). Total training
    time is approximately 2–3 hours on a single RTX 4060 Ti. If your GPU has
    less than 10 GB of VRAM, reduce the batch size (see Section 5.2). Note that
    doing so without adding gradient accumulation will alter training dynamics
    and may affect results relative to the reference environment.

--------------------------------------------------------------------------------
  4. INSTALLATION
--------------------------------------------------------------------------------

  Step 1 — Clone the repository

    git clone https://github.com/Wilson-LL/AI-Quant.git
    cd AI-Quant

  Step 2 — Create and activate a virtual environment

    python -m venv .venv

    # Windows
    .\.venv\Scripts\activate

    # macOS / Linux
    source .venv/bin/activate

  Step 3 — Install PyTorch (CUDA build)

    Visit the official PyTorch installation selector and generate the command
    appropriate for your OS, CUDA version, and package manager:

      https://pytorch.org/get-started/locally/

    Example command used in the reference environment (Windows, CUDA 13.2 nightly):

      pip3 install --pre torch torchvision \
          --index-url https://download.pytorch.org/whl/nightly/cu132

    IMPORTANT: Install PyTorch BEFORE running pip install -r requirements.txt.
    Installing in the wrong order can result in a CPU-only PyTorch build being
    pulled in as a transitive dependency.

  Step 4 — Install remaining dependencies

    pip install -r requirements.txt

--------------------------------------------------------------------------------
  5. USAGE
--------------------------------------------------------------------------------

  5.1  STEP 1 — Generate Stock Metadata  (generate_stocks_json.py)
  -----------------------------------------------------------------

  This script validates the configured ticker symbols against the twstock
  registry and writes a stocks.json metadata file to ./checkpoints/. This file
  is consumed by both train.py and inference.py to determine which stocks to
  include and which have been marked valid after data fetching.

  Open generate_stocks_json.py and modify the STOCK_IDS list to define your
  target universe:

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

  Then run:

    python generate_stocks_json.py

  This produces ./checkpoints/stocks.json. Any ticker not found in the twstock
  registry is skipped with a warning. Stocks that fail data fetching during
  training are automatically flagged as invalid (valid: false) in this file and
  excluded from inference.

  Tuning tip:
    Adding more tickers increases dataset size but requires the model to
    generalize across a wider universe, which may reduce per-stock accuracy.
    Start with a focused list if predictive precision is a priority.


  5.2  STEP 2 — Train the Model  (train.py)
  ------------------------------------------

  Run the training script after generating the stock metadata:

    python train.py

  The script trains four independent models, one for each time horizon:

    Horizon     | History Used  | Ensemble Weight
    ------------|---------------|----------------
    half_year   | ~6 months     | 0.25
    one_year    | ~12 months    | 0.55
    two_year    | ~24 months    | 0.15
    three_year  | ~36 months    | 0.05

  Checkpoints are saved to ./checkpoints/<horizon>/. Both the latest and best
  (highest validation PnL) checkpoints are preserved per horizon.

  Key training parameters (configurable at the top of train.py):

    BATCH_SIZE          = 64      # Reduce if GPU OOM (e.g., 32 or 16)
    EPOCHS              = 1000    # Subject to early stopping (patience=50)
    LOOKBACK_WINDOW     = 40      # Input sequence length (trading days)
    PREDICTION_HORIZON  = 20      # Forward window for labeling (trading days)
    TAKE_PROFIT         = 0.12    # +12% target used in labeling and backtest
    STOP_LOSS           = 0.06    # -6% stop used in labeling and backtest
    LEARNING_RATE       = 3e-4
    RESUME              = False   # Set True to resume from latest checkpoint

  If you encounter out-of-memory (OOM) errors:
    Reduce BATCH_SIZE in train.py. The default is 64.

  WARNING: The current implementation does not include gradient accumulation.
  Reducing the batch size will alter effective gradient updates and may produce
  results that differ from the reference training environment.


  5.3  STEP 3 — Run Inference  (inference.py)
  --------------------------------------------

  After all four horizon models have been trained, generate today's predictions:

    python inference.py

  The script loads the best checkpoint for each horizon, fetches recent price
  data for all valid stocks, constructs the feature sequence for today's date,
  and runs the ensemble to produce final scores. Only stocks marked valid: true
  in stocks.json are included.

  Key inference parameters (configurable at the top of inference.py):

    TOP_K              = 30     # Number of stocks to display in ranking
    MARKET_THRESHOLD   = 0.52   # Minimum market score to enable trading signals
    HORIZON_MODELS weights:
      half_year  → 0.25
      one_year   → 0.55
      two_year   → 0.15
      three_year → 0.05

--------------------------------------------------------------------------------
  6. UNDERSTANDING THE OUTPUT
--------------------------------------------------------------------------------

  Example output:

    Market Score: 0.7115  Trade Enabled

    Top 30:
    Rank | Stock | Prob
       1 |  2406 | 0.9726
       2 |  3049 | 0.9625
       3 |  2345 | 0.9529
       4 |  2454 | 0.9431
       5 |  6770 | 0.9387
      ...

    Trading Signals:
    2406 | prob=0.9726
    3049 | prob=0.9625
    ...

  Field descriptions:

    Market Score
      The mean probability score across all valid stocks in the universe.
      Acts as a macro regime indicator. When this value clears the threshold
      (default: 0.52), the system switches to "Trade Enabled" and emits the
      top 10 trading signals. Below the threshold, no signals are emitted.

    Prob (Score)
      The ensemble score for a given stock — a weighted combination of the
      four horizon models' sigmoid outputs, each raised to its calibrated
      alpha exponent before weighting.

      IMPORTANT INTERPRETATION NOTE:
      These are NOT calibrated statistical probabilities. A score of 0.97
      does not imply a 97% chance of the +12% outcome. The alpha calibration
      and ensemble weighting distort the raw probability scale. Treat these
      values as a relative ranking signal only.

    Top 30
      The 30 highest-scoring stocks from the valid universe, regardless of
      whether the market regime gate is open.

    Trading Signals
      The top 10 stocks from the Top 30, emitted only when Trade is Enabled.
      These represent the model's highest-conviction candidates for the session.

--------------------------------------------------------------------------------
  7. RISK MANAGEMENT & STRATEGY
--------------------------------------------------------------------------------

  The labeling and backtesting logic encodes an explicit exit rule:

    Take Profit : +12%  — exit when the position gains 12% from entry
    Stop Loss   :  -6%  — exit when the position loses 6% from entry

  This asymmetric 2:1 reward-to-risk ratio is baked into both the training
  labels (which stock "won" the 20-day window) and the validation backtest
  (which computes simulated PnL using these thresholds). Any live application
  of the model's signals should apply the same exit rules to remain consistent
  with the conditions under which the model was trained.

  Author's personal approach:
    All stocks in the default universe are technology-sector equities. When
    the Market Score is sufficiently high and trading is enabled, the author's
    preferred approach is to consider broader market positions in:

      - 2330  (TSMC)
      - 0050  (Yuanta/FTSE TWSE Taiwan 50 ETF)

    These serve as diversified proxies for the technology sector rather than
    concentrating exposure in individual signals.

  This is not financial advice. Always conduct independent research and manage
  risk appropriately before making any investment decisions.

--------------------------------------------------------------------------------
  8. KNOWN LIMITATIONS
--------------------------------------------------------------------------------

  OTC ("上櫃") Stock Incompatibility
    Data extraction for OTC-listed stocks (traded on the Taipei Exchange,
    as opposed to TWSE) does not function correctly. The root cause has not
    been identified. Problematic tickers are filtered out automatically during
    dataset construction via the valid flag in stocks.json; some OTC tickers
    may still appear in STOCK_IDS but will be excluded at runtime.

  No Gradient Accumulation
    The training loop does not implement gradient accumulation. Reducing BATCH
    below 64 is a valid workaround for VRAM constraints, but it changes training
    dynamics and may degrade model performance compared to the reference setup.

  Scores Are Not Calibrated Probabilities
    Due to alpha exponentiation and ensemble weighting, output scores should be
    treated as ordinal rankings only. See Section 6 for details.

  Static Universe
    Models are trained on a fixed set of tickers. The model does not generalize
    to tickers absent from STOCK_IDS at training time without full retraining.

  Single-Stock Input
    The model evaluates each stock independently, with no cross-sectional
    context. Correlations or sector-level dynamics between stocks are not
    explicitly modeled.

--------------------------------------------------------------------------------
  9. CLOSING REMARKS
--------------------------------------------------------------------------------

  This is an independent research project and should be treated as such.
  Training results will vary across hardware, data availability, and market
  conditions — production-grade consistency is not guaranteed.

  Experimentation is encouraged. Feel free to adjust any hyperparameters,
  modify the training pipeline, or swap in alternative architectures. Concrete
  directions for scaling up if you have access to more powerful hardware:

    - Increase LSTM_HIDDEN and TRANS_HIDDEN (e.g., 256 or 512)
    - Add more LSTM_LAYERS or TRANS_LAYERS
    - Increase TRANS_FF (e.g., 512 or 1024)
    - Expand TRANS_HEADS (e.g., 8)
    - Add gradient accumulation to support larger effective batch sizes
    - Extend LOOKBACK_WINDOW for longer temporal context

  Contributions, forks, and feedback are welcome.

--------------------------------------------------------------------------------
  10. LICENSE
--------------------------------------------------------------------------------

  See LICENSE file in the repository root for full terms of use.

================================================================================
  AI-Quant  |  https://github.com/Wilson-LL/AI-Quant
================================================================================
