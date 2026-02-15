import warnings
warnings.filterwarnings("ignore")

# src/train/train_central.py
import os, argparse, json, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.data.dataset import DreamerWindows
from src.models.cnn1d import CNN1D
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

def train_one_epoch(model, loader, loss_fn, opt, device):
    model.train()
    losses = []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return float(np.mean(losses))

@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    y_true, y_prob = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        prob = torch.sigmoid(logits).cpu().numpy().ravel()
        y_prob.extend(prob.tolist())
        y_true.extend(yb.numpy().ravel().tolist())
    y_true = np.array(y_true, dtype=np.int32)
    y_prob = np.array(y_prob, dtype=np.float32)
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)
    return acc, f1, auc, cm, y_true, y_prob

def plot_curve(history, out_png):
    plt.figure()
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png); plt.close()

def plot_cm(cm, out_png):
    import seaborn as sns
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cbar=False)
    plt.xlabel("Pred"); plt.ylabel("True"); plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png); plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mat", type=str, default="/home/rajeshkumarjogi/Desktop/eeg-fl-emotion/Dataset/DREAMER.mat")
    ap.add_argument("--target", type=str, default="arousal", choices=["arousal","valence"])
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = f"results/central/{args.target}"
    os.makedirs(outdir, exist_ok=True)

    ds = DreamerWindows(args.mat, target=args.target, win_sec=2.0, overlap=0.5)
    n = len(ds)
    val_sz = max(1, int(0.1*n))
    train_sz = n - val_sz
    train_ds, val_ds = random_split(ds, [train_sz, val_sz], generator=torch.Generator().manual_seed(42))
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_dl   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = CNN1D(in_ch=14, n_classes=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss":[], "val_loss":[]}
    best_val = 1e9
    patience, bad = 2, 0

    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_dl, loss_fn, opt, device)
        # compute val loss
        model.eval(); losses=[]
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                losses.append(loss.item())
        va_loss = float(np.mean(losses))
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        print(f"Epoch {epoch:02d}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f} time={time.time()-t0:.1f}s")

        if va_loss < best_val - 1e-4:
            best_val = va_loss; bad = 0
            torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))
        else:
            bad += 1
            if bad > patience:
                print("Early stopping.")
                break

    # final eval
    model.load_state_dict(torch.load(os.path.join(outdir, "model.pt"), map_location=device))
    acc, f1, auc, cm, y_true, y_prob = eval_epoch(model, val_dl, device)
    with open(os.path.join(outdir,"report.json"),"w") as f:
        json.dump({"acc":acc,"f1":f1,"auc":float(auc),"val_windows":len(val_ds)}, f, indent=2)

    plot_curve(history, os.path.join(outdir, "learning_curve.png"))
    try:
        plot_cm(cm, os.path.join(outdir, "confusion_matrix.png"))
    except Exception:
        pass

    # simple ROC plot
    try:
        from sklearn.metrics import RocCurveDisplay
        RocCurveDisplay.from_predictions(y_true, y_prob)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir,"roc.png")); plt.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
