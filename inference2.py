import torch
import numpy as np
import pandas as pd
import os
import json
from datetime import datetime
from tqdm import tqdm

from dataset import fetch_stock_history
from model import LSTM_CondTransformer

def load_valid_stocks(json_path="./checkpoints/stocks.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    stocks = [s["id"] for s in data["stocks"] if s.get("valid", False)]
    print(f"Loaded {len(stocks)} valid stocks")

    return stocks

def load_model(model, path):
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    return model

def load_all_models(horizon_models,
                    features, lstm_hidden, lstm_layers,
                    trans_hidden, trans_heads, trans_layers, trans_ff,
                    seq_len, device):

    models = []

    for h in horizon_models:
        path = f"checkpoints/{h['name']}/best.pt"
        model = LSTM_CondTransformer(
            input_dim=features,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            trans_hidden=trans_hidden,
            trans_heads=trans_heads,
            trans_layers=trans_layers,
            trans_ff=trans_ff,
            dropout=0,
            seq_len=seq_len
        ).to(device)

        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print(f"Loaded {h['name']} (epoch {checkpoint['epoch']})")

        models.append({
            "model": model,
            "weight": h["weight"],
            "name": h["name"],
            "best_alpha": checkpoint["best_alpha"]
        })

    return models

def build_today_sample(df, X=40):

    prices = df["close"].values.astype(np.float32)
    volume = df["volume"].values.astype(np.float32)

    if len(prices) < X:
        raise ValueError("Not enough data")

    # ===== return =====
    log_return = np.zeros_like(prices)
    log_return[1:] = np.log(prices[1:] / (prices[:-1] + 1e-8))

    # ===== volume z-score =====
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(20).mean()
    vol_std20 = vol_series.rolling(20).std()
    vol_z = ((vol_series - vol_ma20) / (vol_std20 + 1e-8)).values

    # ===== volatility =====
    volatility = pd.Series(log_return).rolling(20).std().values

    # ===== momentum =====
    momentum = (prices / (pd.Series(prices).shift(10) + 1e-8) - 1).values

    # ===== time feature =====
    dates = pd.to_datetime(df["date"])

    month_feat = np.sin(2 * np.pi * dates.dt.month / 12).values
    weekday_feat = np.cos(2 * np.pi * dates.dt.weekday / 7).values

    # ===== last X =====
    past_price = prices[-X:]
    past_return = log_return[-X:]
    past_vol = vol_z[-X:]
    past_volatility = volatility[-X:]
    past_momentum = momentum[-X:]
    past_month = month_feat[-X:]
    past_weekday = weekday_feat[-X:]

    base = past_price[-1]
    price_norm = past_price / (base + 1e-8)

    x = np.stack([
        price_norm,
        past_return,
        past_vol,
        past_volatility,
        past_momentum,
        past_month,
        past_weekday
    ], axis=1)

    x = np.nan_to_num(x)

    return x.astype(np.float32)

def predict_stock(stock_id, models, lookback_window, device):

    today = datetime.today()
    months_back = max(3, (lookback_window // 20) + 2)

    from_month = today.month - months_back
    from_year = today.year

    if from_month <= 0:
        from_year -= 1
        from_month += 12

    df = fetch_stock_history(stock_id, from_year, from_month)

    try:
        x = build_today_sample(df, lookback_window)
    except:
        return None

    x = torch.from_numpy(x).unsqueeze(0).to(device)

    final_prob = 0.0
    total_weight = 0.0

    with torch.no_grad():
        for m in models:
            logit = m["model"](x)
            prob = torch.sigmoid(logit).item()
            prob = prob ** m["best_alpha"]

            final_prob += prob * m["weight"]
            total_weight += m["weight"]

    final_prob /= total_weight

    return {
        "stock_id": stock_id,
        "prob": final_prob
    }

def predict_market(stock_ids, models, lookback_window, device):

    results = []

    for stock_id in tqdm(stock_ids):
        r = predict_stock(stock_id, models, lookback_window, device)
        if r is not None:
            results.append(r)

    return results

def trading_decision(results, top_k=30, market_threshold=0.52):

    probs = np.array([r["prob"] for r in results])

    # 排序
    sorted_idx = np.argsort(-probs)

    # top-k ranking（永遠輸出）
    topk = []
    for rank, idx in enumerate(sorted_idx[:top_k], 1):
        topk.append({
            "rank": rank,
            "stock_id": results[idx]["stock_id"],
            "prob": probs[idx]
        })

    market_score = probs.mean()

    # 是否交易
    if market_score < market_threshold:
        return {
            "market_score": market_score,
            "trade": False,
            "topk": topk,
            "trades": []
        }

    # 如果允許交易
    trades = topk[:10]  # 真正下單用 top 10

    return {
        "market_score": market_score,
        "trade": True,
        "topk": topk,
        "trades": trades
    }

if __name__ == "__main__":

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float32

    # Path location
    CHECKPOINT_PATH = "./checkpoints"
    HORIZON_MODELS = [
        {"name": "half_year",  "weight": 0.55},
        {"name": "one_year",   "weight": 0.30},
        {"name": "two_year",   "weight": 0.10},
        {"name": "three_year", "weight": 0.05},
    ]
    # Strategy design
    LOOKBACK_WINDOW     = 40
    PREDICTION_HORIZON  = 20
    TAKE_PROFIT         = 0.12
    STOP_LOSS           = 0.06
    TOP_K = 30
    MARKET_THRESHOLD = 0.52

    # Model structure
    FEATURES        = 7
    LSTM_HIDDEN     = 128
    LSTM_LAYERS     = 1
    TRANS_HIDDEN    = 128
    TRANS_HEADS     = 4
    TRANS_LAYERS    = 2
    TRANS_FF        = 256

    print(" --- Inference Start --- ")

    stock_ids = load_valid_stocks(os.path.join(CHECKPOINT_PATH, "stocks.json"))

    models = load_all_models(HORIZON_MODELS,
                             FEATURES, LSTM_HIDDEN, LSTM_LAYERS,
                             TRANS_HIDDEN, TRANS_HEADS, TRANS_LAYERS, TRANS_FF,
                             LOOKBACK_WINDOW, DEVICE)

    results = predict_market(stock_ids, models, LOOKBACK_WINDOW, DEVICE)

    decision = trading_decision(results, top_k=TOP_K)
    
    print(f"\nMarket Score: {decision['market_score']:.4f}")

    if not decision["trade"]:
        print("No Trade (below threshold)")
    else:
        print("✅ Trade Enabled")

    print(f"\nTop {TOP_K}:")
    print("Rank | Stock | Prob")

    for t in decision["topk"]:
        print(f"{t['rank']:>4} | {t['stock_id']} | {t['prob']:.4f}")

    if decision["trade"]:
        print("\n💰 Trading Signals:")
        for t in decision["trades"]:
            print(f"{t['stock_id']} | prob={t['prob']:.4f}")
