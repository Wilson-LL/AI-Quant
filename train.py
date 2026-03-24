import torch
import os
from tqdm import tqdm
from datetime import datetime
import json

from dataset import build_dataloader
from model import LSTM_CondTransformer


def build_model(input_dim, lstm_hidden, lstm_layers,
                trans_hidden, trans_heads, trans_layers, trans_ff,
                dropout, lr, seq_len):

    model = LSTM_CondTransformer(
        input_dim=input_dim,
        lstm_hidden=lstm_hidden,
        lstm_layers=lstm_layers,
        trans_hidden=trans_hidden,
        trans_heads=trans_heads,
        trans_layers=trans_layers,
        trans_ff=trans_ff,
        dropout=dropout,
        seq_len=seq_len
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.9,
        patience=10,
        min_lr=1e-5
    )

    return model, optimizer, scheduler

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

def save_checkpoint(model, optimizer, scheduler, epoch, best_score, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_score": best_score
    }, path)


def load_checkpoint(model, optimizer, scheduler, path, device):

    checkpoint = torch.load(path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    if "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    best_score = checkpoint.get("best_score", -float("inf"))

    print(f"✅ Resume from epoch {checkpoint['epoch']}")

    return model, optimizer, scheduler, start_epoch, best_score

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
            logits = torch.clamp(logits, -5, 5)

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

def backtest(probs, labels, tp=0.12, sl=0.06, top_k=30, calibrate=True):

    probs = probs.clone()

    if calibrate:
        probs = probs ** 1.5

    top_idx = torch.topk(probs, min(top_k, len(probs))).indices
    selected_probs = probs[top_idx]
    selected_labels = labels[top_idx]

    trades = len(selected_probs)
    pnl = 0.0

    for p, y in zip(selected_probs, selected_labels):
        weight = p.item()
        if y.item() == 1:
            pnl += weight * tp
        else:
            pnl -= weight * sl

    winrate = (selected_labels == 1).sum().item() / trades if trades > 0 else 0

    return {
        "trades": trades,
        "winrate": winrate,
        "pnl": pnl,
        "market_score": probs.mean().item()
    }

def train(train_loader, val_loader, model, optimizer, scheduler, epochs,
          device="cuda", dtype=torch.float32,
          start_epoch=0, best_score=-float("inf"),
          save_dir="checkpoints"):

    os.makedirs(save_dir, exist_ok=True)

    model = model.to(device)

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([2.0], device=device)
    )

    patience = 50
    no_improve = 0
    for epoch in range(start_epoch, start_epoch + epochs):

        print(f"\n===== Epoch {epoch} =====")

        model.train()

        total_loss = 0
        total_samples = 0

        pbar = tqdm(train_loader, leave=False)

        for x, y in pbar:

            x = x.to(device, dtype=dtype)
            y = y.to(device, dtype=dtype)

            y = y * 0.9 + 0.05

            logits = model(x)
            logits = torch.clamp(logits, -5, 5)

            loss = criterion(logits, y)

            if torch.isnan(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_samples += x.size(0)

            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.1e}"
            })

        avg_loss = total_loss / total_samples
        print(f"Train Loss: {avg_loss:.4f}")

        # Validation
        val_loss, val_acc, val_probs, val_labels = evaluate(
            val_loader, model, device, dtype
        )

        bt = backtest(
            val_probs,
            val_labels,
            tp=TAKE_PROFIT,
            sl=STOP_LOSS,
            top_k=30
        )

        print(f"VAL Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")
        print(
            f"Trades: {bt['trades']} | "
            f"WinRate: {bt['winrate']:.2f} | "
            f"PnL: {bt['pnl']:.4f} | "
            f"Market: {bt['market_score']:.3f}"
        )

        score = bt["pnl"]

        scheduler.step(val_loss)

        # save latest
        save_checkpoint(
            model, optimizer, scheduler, epoch, best_score,
            os.path.join(save_dir, "latest.pt")
        )

        # save best
        if score > best_score:
            best_score = score
            no_improve = 0

            save_checkpoint(
                model, optimizer, scheduler, epoch, best_score,
                os.path.join(save_dir, "best.pt")
            )
            print("🔥 Saved BEST model")

        else:
            no_improve += 1

        if no_improve >= patience:
            print(f"🛑 Early stopping at epoch {epoch}")
            break

if __name__ == "__main__":

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.float32
    EPOCHS = 1000
    BATCH_SIZE = 64
    RESUME = False

    with open("stocks.json", "r", encoding="utf-8") as f:
        data = json.load(f)
        stocks = data["stocks"]

    STOCK_ID = [s["id"] for s in stocks]
    FROM_TIME = (datetime.today().year - 3, datetime.today().month)

    LOOKBACK_WINDOW     = 40
    PREDICTION_HORIZON  = 20
    TAKE_PROFIT         = 0.12
    STOP_LOSS           = 0.06

    FEATURES        = 7
    LSTM_HIDDEN     = 128
    LSTM_LAYERS     = 1
    TRANS_HIDDEN    = 128
    TRANS_HEADS     = 4
    TRANS_LAYERS    = 2
    TRANS_FF        = 256
    DROPOUT         = 0.3
    LEARNING_RATE   = 3e-4

    train_loader, val_loader, test_loader = build_dataloaders(
        STOCK_ID, BATCH_SIZE, FROM_TIME,
        LOOKBACK_WINDOW, PREDICTION_HORIZON,
        TAKE_PROFIT, STOP_LOSS
    )

    model, optimizer, scheduler = build_model(
        FEATURES, LSTM_HIDDEN, LSTM_LAYERS,
        TRANS_HIDDEN, TRANS_HEADS, TRANS_LAYERS, TRANS_FF,
        DROPOUT, LEARNING_RATE, LOOKBACK_WINDOW
    )

    start_epoch = 0
    best_score = -float("inf")

    if RESUME:
        path = "checkpoints/latest.pt"
        if os.path.exists(path):
            model, optimizer, scheduler, start_epoch, best_score = load_checkpoint(
                model, optimizer, scheduler, path, DEVICE
            )

    print("🚀 Start Training")

    train(
        train_loader,
        val_loader,
        model,
        optimizer,
        scheduler,
        epochs=EPOCHS,
        device=DEVICE,
        dtype=DTYPE,
        start_epoch=start_epoch,
        best_score=best_score
    )