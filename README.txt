================================================================================
  AI-QUANT — Taiwan Stock Market Prediction with Deep Learning
  https://github.com/Wilson-LL/AI-Quant
================================================================================

  A multi-timeframe deep learning system that ranks Taiwan Stock Exchange (TWSE)
  equities by their probability of achieving +12% price appreciation within a
  20-day forward window. Combines ensemble modeling across time scales with a
  market-level regime gate to determine whether conditions favor active trading.

--------------------------------------------------------------------------------
  TABLE OF CONTENTS
--------------------------------------------------------------------------------

  1. Overview
  2. Requirements
  3. Installation
  4. Usage
       4.1 Step 1 — Configure Stock Universe  (generate_stocks_json.py)
       4.2 Step 2 — Train the Model           (train.py)
       4.3 Step 3 — Run Inference             (inference.py)
  5. Understanding the Output
  6. Known Limitations
  7. Author Notes & Strategy
  8. License

--------------------------------------------------------------------------------
  1. OVERVIEW
--------------------------------------------------------------------------------

  AI-Quant fetches historical price data for a user-defined list of TWSE ticker
  symbols, trains a GPU-accelerated deep learning model, and produces a ranked
  list of stocks with associated "probability" scores.

  Key design decisions:
    - Scores are produced by an ensemble of models trained at different time
      scales; the final probability is a weighted combination and should NOT be
      interpreted as a calibrated statistical probability.
    - A global "Market Score" gates the output: if market conditions are deemed
      unfavorable, trading signals are suppressed regardless of individual scores.
    - The system is intentionally scoped to technology-sector equities listed on
      TWSE. OTC ("上櫃") stocks are excluded due to data extraction issues
      (see Section 6).

--------------------------------------------------------------------------------
  2. REQUIREMENTS
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
    Training takes approximately 2–3 hours on a single RTX 4060 Ti.
    If your GPU has less than 10 GB of VRAM, reduce the batch size
    (see Section 4.2 for details). Note that reducing the batch size
    without adding gradient accumulation may affect model quality
    relative to the reference training environment.

--------------------------------------------------------------------------------
  3. INSTALLATION
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
    Installing them in the wrong order can result in a CPU-only PyTorch build
    being pulled in as a dependency.

  Step 4 — Install remaining dependencies

    pip install -r requirements.txt

--------------------------------------------------------------------------------
  4. USAGE
--------------------------------------------------------------------------------

  4.1  STEP 1 — Configure Stock Universe  (generate_stocks_json.py)
  -----------------------------------------------------------------

  Open generate_stocks_json.py and locate the STOCK_IDS list under the
  `if __name__ == "__main__":` block. Add or remove TWSE ticker symbols
  to define the universe of stocks the model will train and predict on.

  Default configuration:

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

  Then run the script to fetch and cache stock data:

    python generate_stocks_json.py

  Tuning tip:
    Adding more tickers increases coverage but can reduce per-stock prediction
    accuracy, as the model must generalize across a larger and more diverse
    universe. Start with a focused list if accuracy is a priority.


  4.2  STEP 2 — Train the Model  (train.py)
  ------------------------------------------

  Run the training script after generating the stock data:

    python train.py

  Training configuration:
    - Default batch size: BATCH = 64  (defined at line 288 in train.py)
    - Estimated training time: 2–3 hours on RTX 4060 Ti
    - Minimum recommended VRAM: 10–12 GB

  If you encounter out-of-memory (OOM) errors:
    Reduce the BATCH variable in train.py (e.g., 32 or 16).

  WARNING: The current implementation does not include gradient accumulation.
  Reducing the batch size will alter effective gradient updates and may produce
  results that differ from the reference training environment.


  4.3  STEP 3 — Run Inference  (inference.py)
  --------------------------------------------

  After training completes, generate predictions for the current date:

    python inference.py

--------------------------------------------------------------------------------
  5. UNDERSTANDING THE OUTPUT
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
      A composite regime indicator. When this score clears the internal
      threshold, the system switches to "Trade Enabled" mode and emits
      trading signals. Below the threshold, signals are suppressed.

    Prob (Probability Score)
      The model's estimate that a given stock will achieve a +12% price
      increase within the next 20 trading days.

      IMPORTANT INTERPRETATION NOTE:
      These scores are a weighted combination of outputs from multiple
      models trained on different time horizons. They are NOT calibrated
      statistical probabilities. A score of 0.97 does not mean a 97%
      chance of the +12% outcome. Use these values as a relative ranking
      tool, not as absolute probability estimates.

    Top 30
      The 30 highest-ranked stocks from the configured universe.

    Trading Signals
      The subset of the Top 30 that clears the internal confidence
      threshold under the current market regime.

--------------------------------------------------------------------------------
  6. KNOWN LIMITATIONS
--------------------------------------------------------------------------------

  OTC ("上櫃") Stock Incompatibility
    Data extraction for OTC-listed stocks (traded on the Taipei Exchange,
    as opposed to TWSE) does not function correctly. The root cause has not
    been identified. As a workaround, problematic tickers are filtered out
    during dataset construction; some may still appear in STOCK_IDS but will
    be excluded automatically at runtime.

  No Gradient Accumulation
    The training loop does not implement gradient accumulation. Reducing
    BATCH below 64 is a valid workaround for VRAM constraints, but it will
    change training dynamics and may degrade model performance.

  No Probability Calibration
    Scores should be treated as ordinal rankings, not calibrated probabilities
    (see Section 5 for details).

  Static Universe
    The model is trained on a fixed set of tickers. It does not generalize
    to tickers not present in STOCK_IDS at training time without retraining.

--------------------------------------------------------------------------------
  7. AUTHOR NOTES & STRATEGY
--------------------------------------------------------------------------------

  All stocks in the default universe are technology-sector equities.

  Personal approach:
    When the Market Score is sufficiently high and trading is enabled, the
    author's preferred approach is to consider positions in:

      - 2330  (TSMC)
      - 0050  (Yuanta/FTSE TWSE Taiwan 50 ETF)

    These are used as broad market proxies rather than relying solely on
    individual stock signals from the model.

  This is not financial advice. Always conduct independent research and
  manage risk appropriately before making investment decisions.

--------------------------------------------------------------------------------
  8. CLOSING REMARKS
--------------------------------------------------------------------------------

  This is an independent research project and should be treated as such.
  Training results may vary across hardware, configurations, and data
  availability — do not expect production-grade consistency out of the box.

  Experimentation is encouraged. Feel free to adjust any hyperparameters,
  modify the training pipeline, or swap in alternative architectures to suit
  your needs. If you have access to more powerful hardware, scaling up the
  model capacity (deeper layers, wider embeddings, larger batch sizes with
  gradient accumulation) is a reasonable starting point for improving
  predictive performance.

--------------------------------------------------------------------------------
  9. LICENSE
--------------------------------------------------------------------------------

  See LICENSE file in the repository root for full terms of use.

================================================================================
  AI-Quant  |  https://github.com/Wilson-LL/AI-Quant
================================================================================
