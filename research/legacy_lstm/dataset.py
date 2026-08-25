import twstock
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

import torch
from torch.utils.data import Dataset, DataLoader

def fetch_stock_history(stock_id, from_year, from_month):
    stock = twstock.Stock(stock_id)
    data = stock.fetch_from(from_year, from_month)

    df = pd.DataFrame({
        "date": [d.date for d in data],
        "close": [d.close for d in data],
        "volume": [d.capacity for d in data]
    })

    df = df.dropna().reset_index(drop=True)
    return df

def build_samples(df, X=40, Y=20, H=0.12, L=0.06, add_time_feature=True):

    prices = df["close"].values.astype(np.float32)
    volume = df["volume"].values.astype(np.float32)

    # return
    log_return = np.zeros_like(prices)
    log_return[1:] = np.log(prices[1:] / prices[:-1] + 1e-8)

    # volume (Z-score)
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(20).mean()
    vol_std20 = vol_series.rolling(20).std()
    vol_z = ((vol_series - vol_ma20) / (vol_std20 + 1e-8)).values

    volatility = pd.Series(log_return).rolling(20).std().values
    momentum = (prices / (pd.Series(prices).shift(10) + 1e-8) - 1).values

    # to clean head NaN
    valid_start = 20

    prices = prices[valid_start:]
    log_return = log_return[valid_start:]
    vol_z = vol_z[valid_start:]
    volatility = volatility[valid_start:]
    momentum = momentum[valid_start:]

    if add_time_feature:
        dates = pd.to_datetime(df["date"].iloc[valid_start:])

        month_feat = np.sin(2 * np.pi * dates.dt.month / 12).values
        weekday_feat = np.cos(2 * np.pi * dates.dt.weekday / 7).values
    else:
        month_feat = np.zeros_like(prices)
        weekday_feat = np.zeros_like(prices)

    samples = []
    for t in range(X - 1, len(prices) - Y):

        # past
        past_price = prices[t-(X-1):t+1]
        past_return = log_return[t-(X-1):t+1]
        past_vol = vol_z[t-(X-1):t+1]
        past_volatility = volatility[t-(X-1):t+1]
        past_momentum = momentum[t-(X-1):t+1]

        # time feature
        past_month = month_feat[t-(X-1):t+1]
        past_weekday = weekday_feat[t-(X-1):t+1]

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
        ], axis=1)  # shape: (X, 7)

        # NaN check
        if not np.isfinite(x).all():
            continue

        # future
        future = prices[t+1:t+Y+1]
        future_norm = future / (base + 1e-8)

        label = 0
        for p in future_norm:
            if p >= 1 + H:
                label = 1
                break
            if p <= 1 - L:
                label = 0
                break

        samples.append((x.astype(np.float32), label))

    return samples

def build_grouped_samples(stock_ids, from_time=(2025, 1),
                          X=40, Y=20, H=0.12, L=0.06):
    """Fetch each stock and build its samples, keeping them grouped per stock
    and in chronological order.

    Returns a list of per-stock sample lists (each inner list is time-ordered).
    Grouping and ordering are preserved so callers can split the data
    chronologically without leaking future information into the past — see
    ``chronological_split``.
    """

    valid_stock_ids = []
    grouped = []
    total = 0
    positives = 0
    iterator = tqdm(stock_ids, desc="📊 Stocks", position=0)
    for stock_id in iterator:
        try:
            df = fetch_stock_history(stock_id, from_time[0], from_time[1])
            samples = build_samples(df, X, Y, H, L)

            if len(samples) != 0:
                grouped.append(samples)
                valid_stock_ids.append(stock_id)
                total += len(samples)
                positives += sum(y for _, y in samples)

                tqdm.write(f"[{stock_id}] rows: {len(df)}")
                tqdm.write(f"[{stock_id}] samples: {len(samples)}")
            else:
                tqdm.write(f"skip {stock_id}")

        except Exception as e:
            tqdm.write(f"skip {stock_id}: {e}")

    # validation
    if total == 0:
        raise ValueError("❌ No samples generated. Check data or filters.")

    # label distribution
    pos_ratio = positives / total

    print("\n===== Dataset Summary =====")
    print(f"Total samples: {total}")
    print(f"Positive ratio: {pos_ratio:.4f}")

    # store valid stock id
    with open("./checkpoints/stocks.json", "r", encoding="utf-8") as f:
        stock_data = json.load(f)
    for stock in stock_data["stocks"]:
        if (stock["valid"] is None or stock["valid"] is True) and stock["id"] in valid_stock_ids:
            stock["valid"] = True
        else:
            stock["valid"] = False

    with open("./checkpoints/stocks.json", "w", encoding="utf-8") as f:
        json.dump(stock_data, f, ensure_ascii=False, indent=4)

    return grouped


def chronological_split(grouped, train_frac=0.7, val_frac=0.15, purge=0):
    """Split per-stock, time-ordered samples into train/val/test chronologically.

    Samples are overlapping sliding windows: consecutive samples share most of
    their input days and their forward label horizons overlap. A random split
    would therefore leak near-identical, look-ahead-correlated windows across
    the sets and inflate validation metrics. Instead we split each stock's
    samples by time and drop a ``purge`` gap of samples after each boundary so
    that no training window (or its forward label period) overlaps a
    validation/test window. Set ``purge`` to ``lookback + horizon``.

    Returns ``(train, val, test)`` as flat lists of ``(x, label)`` samples.
    """

    train, val, test = [], [], []
    for samples in grouped:
        n = len(samples)
        tr_end = int(n * train_frac)
        va_end = tr_end + int(n * val_frac)

        train.extend(samples[:tr_end])
        val.extend(samples[tr_end + purge:va_end])
        test.extend(samples[va_end + purge:])

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise ValueError(
            "❌ Chronological split produced an empty set "
            f"(train={len(train)}, val={len(val)}, test={len(test)}). "
            "Not enough per-stock history for the requested purge gap."
        )

    return train, val, test


def build_dataloader(stock_ids, batch_size=64, from_time=(2025, 1),
                     X=40, Y=20, H=0.12, L=0.06):

    grouped = build_grouped_samples(stock_ids, from_time=from_time,
                                    X=X, Y=Y, H=H, L=L)
    all_samples = [s for group in grouped for s in group]

    dataset = StockDataset(all_samples)

    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=True, drop_last=True)

# Dataset Structure
class StockDataset(Dataset):

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        x, y = self.samples[idx]

        x = torch.from_numpy(x)   # (X, 5)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y
