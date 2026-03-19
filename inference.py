# =========================
# inference.py - 今日推論版本
# =========================
import torch
import pandas as pd
import numpy as np
from dataset import fetch_stock_history, build_samples
from model import LSTM_CondTransformer
from datetime import datetime


# =========================
# Config
# =========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 可以是單支股票或多支股票列表
# STOCK_ID = ["2330", "0050", "2317",
#             "2303", "2421", "3017",
#             "2419", "3037", "6488",
#             "3105", "6285", "2314",
#             "3491", "6669", "2356",
#             "2357", "2382", "3036",
#             "2891"]

# [6488, 3105, ]
STOCK_IDS = ["6669", "2356", "2357", "2382", "3036", "2891"]  

X = 40                  # 序列長度
Y = 20                  # label horizon，推論不需要
H = 0.12
L = 0.06
MODEL_PATH = "checkpoints/best.pt"
THRESHOLD = 0.5         # 硬買入信號閾值

# =========================
# Build model
# =========================
def build_model(input_dim=5, seq_len=40):
    model = LSTM_CondTransformer(
        input_dim=input_dim,
        lstm_hidden=64,
        lstm_layers=1,
        trans_hidden=64,
        trans_heads=4,
        trans_layers=2,
        trans_ff=128,
        dropout=0.1,
        seq_len=seq_len
    ).to(DEVICE)
    return model

# =========================
# Load checkpoint
# =========================
def load_model_checkpoint(model, path):
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"✅ Loaded model checkpoint from epoch {checkpoint['epoch']}")
    return model

def build_today_sample(df, X=40):
    """
    建立今日推論用的單筆 sample (只使用過去 X 天資料)
    df: 包含 'close' 與 'volume'
    X: LSTM/Transformer 序列長度
    return: numpy array shape (X, feature_dim=5)
    """
    prices = df["close"].values.astype(np.float32)
    volume = df["volume"].values.astype(np.float32)

    if len(prices) < X:
        raise ValueError(f"Not enough historical data to build sample. Require {X} days, got {len(prices)}.")

    # ===== return =====
    log_return = np.zeros_like(prices)
    log_return[1:] = np.log(prices[1:] / (prices[:-1] + 1e-8))

    # ===== volume (Z-score) =====
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(20).mean()
    vol_std20 = vol_series.rolling(20).std()
    vol_z = ((vol_series - vol_ma20) / (vol_std20 + 1e-8)).values

    # ===== volatility (return std) =====
    volatility = pd.Series(log_return).rolling(20).std().values

    # ===== momentum =====
    momentum = (prices / (pd.Series(prices).shift(10) + 1e-8) - 1).values

    # ===== 取最後 X 天 =====
    past_price = prices[-X:]
    past_return = log_return[-X:]
    past_vol = vol_z[-X:]
    past_volatility = volatility[-X:]
    past_momentum = momentum[-X:]

    base = past_price[-1]
    price_norm = past_price / (base + 1e-8)

    x = np.stack([
        price_norm,
        past_return,
        past_vol,
        past_volatility,
        past_momentum
    ], axis=1)

    # 防止 NaN
    x = np.nan_to_num(x)

    return x.astype(np.float32)

# =========================
# Predict function
# =========================
def predict_today(stock_id, model, X, H, L, threshold=0.5):
    # 計算從今天往回需要抓幾個月資料，保證至少抓到 X 個交易日
    today = datetime.today()
    months_back = max(1, (X // 20) + 1)  # 粗估每月 20 交易日
    from_month = today.month - months_back
    from_year = today.year
    if from_month <= 0:
        from_year -= 1
        from_month += 12

    # 取得歷史資料
    df = fetch_stock_history(stock_id, from_year, from_month)

    # df: 最近抓到的股價資料
    x_today = build_today_sample(df, X=X)
    x_today = torch.from_numpy(x_today).unsqueeze(0).to(DEVICE)  # (1, X, feature_dim)

    # 推論
    with torch.no_grad():
        logit = model(x_today)
        prob = torch.sigmoid(logit).item()
        buy_signal = int(prob > threshold)

    recent_prices = df['close'].values[-X:]

    return {
        "stock_id": stock_id,
        "prob": prob,
        "buy_signal": buy_signal,
        "recent_prices": recent_prices
    }

# =========================
# Main
# =========================
if __name__ == "__main__":

    model = build_model(input_dim=5, seq_len=X)
    model = load_model_checkpoint(model, MODEL_PATH)

    for stock_id in STOCK_IDS:
        result = predict_today(stock_id, model, X, H, L, threshold=THRESHOLD)
        print("=========================================")
        print(f"📊 STOCK_ID: {result['stock_id']}")
        print(f"Probability to BUY today: {result['prob']:.6f}")
        print(f"Buy signal (threshold={THRESHOLD}): {result['buy_signal']}")
        print(f"Recent {X} days prices: {result['recent_prices']}")