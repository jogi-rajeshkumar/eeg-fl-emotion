# src/eval/evaluate.py
import os, json, argparse
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt

from src.models.cnn1d import CNN1D
from src.data.dataset import DreamerWindows
from src.data.load_dreamer import load_dreamer_mat


def set_parameters(model: torch.nn.Module, state_dict: OrderedDict):
    model.load_state_dict(state_dict, strict=True)


def get_val_subject_ids(mat_path: str, k_last: int = 3):
    subjects, _ = load_dreamer_mat(mat_path)
    all_ids = sorted(list(subjects.keys()))
    k_last = min(k_last, len(all_ids))
    return all_ids[-k_last:], all_ids


@torch.no_grad()
def run_inference(model, dl, device):
    model.eval()
    y_true, y_prob = [], []
    loss_fn = nn.BCEWithLogitsLoss()
    losses = []

    for xb, yb in dl:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        losses.append(loss.item())
        prob = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        y_prob.append(prob)
        y_true.append(yb.detach().cpu().numpy().ravel())

    y_prob = np.concatenate(y_prob) if y_prob else np.array([])
    y_true = np.concatenate(y_true) if y_true else np.array([])
    avg_loss = float(np.mean(losses)) if losses else float("nan")
    return y_true, y_prob, avg_loss


def safe_auc(y_true, y_prob):
    try:
        # AUC needs both classes present
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def evaluate_model(target: str, model_path: str, mat_path: str, out_dir: str, batch_size: int = 256):
    os.makedirs(out_dir, exist_ok=True)

    # ---- validation subjects (deterministic) ----
    val_ids, all_ids = get_val_subject_ids(mat_path, k_last=3)

    # ---- dataset / dataloader ----
    # IMPORTANT: DreamerWindows uses subjects_subset, not 'subjects'
    val_ds = DreamerWindows(
        mat_path,
        target=target,
        subjects_subset=val_ids,
        win_sec=2.0,
        overlap=0.5,
    )
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ---- model ----
    device = torch.device("cpu")
    model = CNN1D(in_ch=14, n_classes=1).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Expected a trained central model at results/central/{target}/model.pt."
        )

    state = torch.load(model_path, map_location="cpu")
    set_parameters(model, state if isinstance(state, OrderedDict) else OrderedDict(state))

    # ---- inference ----
    y_true, y_prob, avg_loss = run_inference(model, val_dl, device)

    # metrics
    y_pred = (y_prob >= 0.5).astype(int) if y_prob.size else np.array([], dtype=int)
    acc = float(accuracy_score(y_true, y_pred)) if y_true.size else float("nan")
    f1 = float(f1_score(y_true, y_pred, zero_division=0)) if y_true.size else float("nan")
    auc = safe_auc(y_true, y_prob)

    # confusion matrix (only if both classes possible)
    if y_true.size:
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    else:
        cm = np.array([[0, 0], [0, 0]])

    # ---- save plots ----
    # ROC curve
    roc_path = os.path.join(out_dir, "roc.png")
    try:
        if y_true.size and len(np.unique(y_true)) == 2:
            RocCurveDisplay.from_predictions(y_true, y_prob)
            plt.title(f"ROC — {target}")
            plt.savefig(roc_path, bbox_inches="tight", dpi=150)
            plt.close()
        else:
            # create a blank figure explaining why ROC not plotted
            plt.figure()
            plt.text(0.5, 0.5, "ROC unavailable (single-class in val set)", ha="center", va="center")
            plt.axis("off")
            plt.savefig(roc_path, bbox_inches="tight", dpi=150)
            plt.close()
    except Exception:
        # be resilient
        plt.figure()
        plt.text(0.5, 0.5, "ROC plotting error", ha="center", va="center")
        plt.axis("off")
        plt.savefig(roc_path, bbox_inches="tight", dpi=150)
        plt.close()

    # Confusion matrix
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix — {target}")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(cm_path, bbox_inches="tight", dpi=150)
    plt.close()

    # ---- save metrics/report ----
    report = {
        "target": target,
        "model_path": model_path,
        "val_subjects": val_ids,
        "all_subjects_count": len(all_ids),
        "n_windows": int(y_true.size),
        "loss": avg_loss,
        "acc": acc,
        "f1": f1,
        "auc": auc,
    }
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    # append a single-row CSV (create header if new)
    mpath = os.path.join(out_dir, "metrics.csv")
    new_file = not os.path.exists(mpath)
    import csv
    with open(mpath, "a", newline="") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["loss", "acc", "f1", "auc", "n_windows"])
        w.writerow([avg_loss, acc, f1, auc, y_true.size])

    print(f"[EVAL] {target} | loss={avg_loss:.4f} acc={acc:.4f} f1={f1:.4f} auc={auc:.4f}")
    print(f"[EVAL] wrote: {mpath}, {roc_path}, {cm_path}, {os.path.join(out_dir, 'report.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=str, required=True, choices=["arousal", "valence"])
    ap.add_argument("--mat", type=str, default="data/DREAMER.mat")
    ap.add_argument("--model", type=str, default="")   # optional override
    ap.add_argument("--outdir", type=str, default="")  # optional override
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    # defaults derived from target if not provided
    if not args.outdir or args.outdir.strip() == "":
        out_dir = os.path.join("results", "central", args.target)
    else:
        out_dir = args.outdir

    if not args.model or args.model.strip() == "":
        model_path = os.path.join("results", "central", args.target, "model.pt")
    else:
        model_path = args.model

    evaluate_model(args.target, model_path, args.mat, out_dir, batch_size=args.batch)


if __name__ == "__main__":
    main()
