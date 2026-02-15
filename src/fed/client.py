# src/fed/client.py
import argparse, os, time, numpy as np, torch
import flwr as fl
from collections import OrderedDict
from torch.utils.data import DataLoader, random_split, Subset
import torch.nn as nn

from src.data.dataset import DreamerWindows
from src.models.cnn1d import CNN1D


def get_parameters(model):
    # Return a list of numpy arrays (Flower expects this for NumPyClient)
    return [p.detach().cpu().numpy() for _, p in model.state_dict().items()]


def set_parameters(model, parameters):
    # Set model params from list of numpy arrays (same order as state_dict)
    state_dict = OrderedDict(
        {k: torch.tensor(v) for (k, _), v in zip(model.state_dict().items(), parameters)}
    )
    model.load_state_dict(state_dict, strict=True)


class DreamerClient(fl.client.NumPyClient):
    def __init__(
        self,
        subject_id: str,
        target: str,
        mat_path: str,
        batch: int = 128,
        epochs: int = 1,
        win_sec: float = 2.0,
        overlap: float = 0.5,
        device: str = "cpu",
        max_windows: int = 0,
    ):
        self.subject_id = subject_id
        self.target = target
        self.mat_path = mat_path
        self.batch = batch
        self.local_epochs = epochs
        self.win_sec = win_sec
        self.overlap = overlap
        self.max_windows = max_windows
        self.device = torch.device(device)

        self.model = CNN1D(in_ch=14, n_classes=1).to(self.device)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.train_dl = None
        self.val_dl = None
        self.train_ds = None
        self.val_ds = None

    def _ensure_data(self):
        if self.train_dl is not None:  # already built
            return

        full = DreamerWindows(
            self.mat_path,
            target=self.target,
            subjects_subset=[self.subject_id],
            win_sec=self.win_sec,
            overlap=self.overlap,
        )

        # Cap number of windows to bound memory/compute
        if self.max_windows and len(full) > self.max_windows:
            full = Subset(full, list(range(self.max_windows)))

        n = len(full)
        val_sz = max(1, int(0.1 * n))
        train_sz = n - val_sz
        g = torch.Generator().manual_seed(42)
        self.train_ds, self.val_ds = random_split(full, [train_sz, val_sz], generator=g)

        # Keep memory footprint minimal
        self.train_dl = DataLoader(
            self.train_ds,
            batch_size=self.batch,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )
        self.val_dl = DataLoader(
            self.val_ds,
            batch_size=self.batch,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

    # -------- Flower NumPyClient API --------
    def get_parameters(self, config):
        return get_parameters(self.model)

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        self._ensure_data()
        self.model.train()

        for _ in range(self.local_epochs):
            for xb, yb in self.train_dl:
                # Ensure correct dtype/shapes; keep everything float32 on CPU
                xb = xb.to(self.device, dtype=torch.float32)
                yb = yb.to(self.device, dtype=torch.float32)
                if yb.ndim == 1:
                    yb = yb.unsqueeze(1)  # (B,) -> (B,1) for BCEWithLogitsLoss

                self.opt.zero_grad(set_to_none=True)
                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
                loss.backward()
                self.opt.step()

        return get_parameters(self.model), len(self.train_ds), {}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        self._ensure_data()
        self.model.eval()

        y_true, y_prob, losses = [], [], []
        with torch.no_grad():
            for xb, yb in self.val_dl:
                xb = xb.to(self.device, dtype=torch.float32)
                yb = yb.to(self.device, dtype=torch.float32)
                if yb.ndim == 1:
                    yb = yb.unsqueeze(1)  # (B,) -> (B,1)

                logits = self.model(xb)
                loss = self.loss_fn(logits, yb)
                losses.append(loss.item())

                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                y_prob.extend(probs.tolist())
                y_true.extend(yb.cpu().numpy().ravel().tolist())

        # Metrics
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        loss = float(np.mean(losses)) if losses else 0.0
        if len(y_true) == 0:
            acc = 0.0
            f1 = 0.0
            auc = float("nan")
        else:
            y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
            acc = float(accuracy_score(y_true, y_pred))
            f1 = float(f1_score(y_true, y_pred, zero_division=0))
            try:
                auc = float(roc_auc_score(y_true, y_prob))
            except Exception:
                auc = float("nan")

        # Return (loss, num_examples, metrics_dict)
        return loss, len(self.val_ds), {"acc": acc, "f1": f1, "auc": auc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", type=str, default="127.0.0.1:8080")
    ap.add_argument("--subject", type=str, required=True)
    ap.add_argument("--target", type=str, default="arousal", choices=["arousal", "valence"])
    ap.add_argument(
        "--mat",
        type=str,
        default="/home/rajeshkumarjogi/Desktop/eeg-fl-emotion/Dataset/DREAMER.mat",
    )
    ap.add_argument("--device", type=str, default="cpu")  # e.g., "cpu" or "cuda"
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--win_sec", type=float, default=2.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--max_windows", type=int, default=0)
    ap.add_argument("--retries", type=int, default=30)
    ap.add_argument("--retry_sleep", type=float, default=3.0)
    args = ap.parse_args()

    client = DreamerClient(
        subject_id=args.subject,
        target=args.target,
        mat_path=args.mat,
        batch=args.batch,
        epochs=args.epochs,
        win_sec=args.win_sec,
        overlap=args.overlap,
        device=args.device,
        max_windows=args.max_windows,
    )

    # Robust connect: retry if server not yet accepting connections
    last_err = None
    for attempt in range(1, args.retries + 1):
        try:
            fl.client.start_client(server_address=args.server, client=client.to_client())
            last_err = None
            break
        except Exception as e:
            last_err = e
            print(f"[Client {args.subject}] connect attempt {attempt}/{args.retries} failed: {e}")
            time.sleep(args.retry_sleep)
    if last_err:
        raise last_err


if __name__ == "__main__":
    # Keep PyTorch threads small when you run many clients
    torch.set_num_threads(1)
    main()
