import torch
import os
from tqdm import tqdm
from datetime import datetime
import json

from dataset import build_dataloader
from model import LSTM_CondTransformer

def build_model(input_dim, lstm_hidden, lstm_layers,
                trans_hidden, trans_heads, trans_layers, trans_ff,
                dropout, seq_len):
    model = LSTM_CondTransformer(
        input_dim=input_dim,
        lstm_hidden=lstm_hidden,lstm_layers=lstm_layers,trans_hidden=trans_hidden,
        trans_heads=trans_heads, trans_layers=trans_layers, trans_ff=trans_ff,
        dropout=dropout, seq_len=seq_len
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    return model, optimizer

def build_dataloaders(stock_ids, batch_size, from_time, x, y, h, l):
    full_loader = build_dataloader(
        stock_ids,
        batch_size=batch_size,
        from_time=from_time,
        X=x, Y=y, H=h, L=l
    )

    dataset = full_loader.dataset

    n = len(dataset)
    train_n = int(n * 0.7)
    val_n = int(n * 0.15)
    test_n = n - train_n - val_n

    train_set, val_set, test_set = torch.utils.data.random_split(
        dataset, [train_n, val_n, test_n]
    )

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(val_set, batch_size=batch_size)
    test_loader  = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    return train_loader, val_loader, test_loader

def save_checkpoint(model, optimizer, epoch, best_loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_loss": best_loss
    }, path)


def load_checkpoint(model, optimizer, path, device):

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_loss = checkpoint.get("best_loss", float("inf"))

    print(f"✅ Resume from epoch {checkpoint['epoch']}")

    return model, optimizer, start_epoch, best_loss

def evaluate(loader, model, device, dtype):

    model.eval()

    total_loss = 0
    total_samples = 0
    total_correct = 0

    all_probs = []
    all_labels = []

    criterion = torch.nn.BCEWithLogitsLoss()

    with torch.no_grad():
        for x, y in loader:

            x = x.to(device, dtype=dtype)
            y = y.to(device, dtype=dtype)

            logits = model(x)
            loss = criterion(logits, y)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()

            total_correct += (preds == y).sum().item()
            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples

    return avg_loss, acc, torch.cat(all_probs), torch.cat(all_labels)

def backtest(probs, labels, threshold=0.5, tp=0.12, sl=0.06):

    preds = (probs > threshold).float()

    trades = preds.sum().item()

    if trades == 0:
        return {"trades": 0, "winrate": 0, "pnl": 0}

    wins = ((preds == 1) & (labels == 1)).sum().item()
    losses = trades - wins

    pnl = wins * tp - losses * sl
    winrate = wins / trades

    return {"trades": trades, "winrate": winrate, "pnl": pnl}

def train(train_loader, val_loader, model, optimizer, epochs,
          device="cuda", dtype=torch.float32,
          start_epoch=0, best_loss=float("inf"),
          save_dir="checkpoints"):

    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)

    # Class Imbalance
    all_labels = torch.cat([y for _, y in train_loader])
    pos_ratio = max(all_labels.mean().item(), 1e-6)
    pos_weight = torch.tensor([(1 - pos_ratio) / pos_ratio], device=device)

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    print(f"Positive ratio: {pos_ratio:.4f}")

    # train
    for epoch in range(start_epoch, start_epoch + epochs):

        print(f"\n===== Epoch {epoch} =====")

        model.train()

        total_loss = 0
        total_samples = 0

        ema_loss = 0
        alpha = 0.98

        pbar = tqdm(train_loader, leave=False)

        for x, y in pbar:

            x = x.to(device, dtype=dtype)
            y = y.to(device, dtype=dtype)

            logits = model(x)
            loss = criterion(logits, y)

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            ema_loss = alpha * ema_loss + (1 - alpha) * loss.item()

            # tqdm progress bar update
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "ema": f"{ema_loss:.4f}",
                              "lr": f"{optimizer.param_groups[0]['lr']:.1e}", "grad": f"{grad_norm:.2f}"})

        avg_loss = total_loss / total_samples
        print(f"Train Loss: {avg_loss:.4f}")

        # Validation & Backtest
        val_loss, val_acc, val_probs, val_labels = evaluate(val_loader, model, device, dtype)
        bt = backtest(val_probs, val_labels)
        print(f"VAL Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        print(f"💰 Trades: {bt['trades']} | WinRate: {bt['winrate']:.2f} | PnL: {bt['pnl']:.4f}")

        # Save Checkpoint
        save_checkpoint(model, optimizer, epoch, best_loss, os.path.join(save_dir, "latest.pt"))
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, epoch, best_loss, os.path.join(save_dir, "best.pt"))
            print("🔥 Saved BEST model")

if __name__ == "__main__":

    # Train Parameter
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float32
    EPOCHS = 1000
    BATCH_SIZE = 64
    RESUME = True

    # Data Configuration
    with open("stocks.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        stocks = data["stocks"]
    STOCK_ID = [s["id"] for s in stocks]
    FROM_TIME = (datetime.today().year - 3, datetime.today().month)

    # Trading Strategy Configuration
    LOOKBACK_WINDOW     = 40        # past context length
    PREDICTION_HORIZON  = 20        # future evaluation window
    TAKE_PROFIT         = 0.12      # +12%
    STOP_LOSS           = 0.06      # -6%

    # Model Structure
    FEATURES        = 5
    LSTM_HIDDEN     = 384
    LSTM_LAYERS     = 2
    TRANS_HIDDEN    = 384
    TRANS_HEADS     = 6
    TRANS_LAYERS    = 3
    TRANS_FF        = 1024
    DROPOUT         = 0.1

    train_loader, val_loader, test_loader = build_dataloaders(
        STOCK_ID, BATCH_SIZE, FROM_TIME, LOOKBACK_WINDOW, PREDICTION_HORIZON, TAKE_PROFIT, STOP_LOSS
    )

    model, optimizer = build_model(input_dim=FEATURES, lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS,
                trans_hidden=TRANS_HIDDEN, trans_heads=TRANS_HEADS, trans_layers=TRANS_LAYERS, trans_ff=TRANS_FF,
                dropout=DROPOUT, seq_len=LOOKBACK_WINDOW)

    start_epoch = 0
    best_loss = float("inf")
    if RESUME:
        path = "checkpoints/latest.pt"
        if os.path.exists(path):
            model, optimizer, start_epoch, best_loss = load_checkpoint(model, optimizer, path, DEVICE)

    print("🚀 Start Training")
    train(train_loader, val_loader, model, optimizer,
          epochs=EPOCHS, device=DEVICE, dtype=DTYPE, start_epoch=start_epoch, best_loss=best_loss)