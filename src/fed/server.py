# src/fed/server.py
import argparse, os, csv
from math import ceil
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import parameters_to_ndarrays

from src.data.dataset import DreamerWindows
from src.models.cnn1d import CNN1D
from src.data.load_dreamer import load_dreamer_mat


# ----------------------- helpers -----------------------

def set_parameters(model, parameters_nd):
    """Load weights (list of ndarrays) into model."""
    state_dict = OrderedDict({k: torch.tensor(v) for (k, _), v in zip(model.state_dict().items(), parameters_nd)})
    model.load_state_dict(state_dict, strict=True)

def _to_ndarrays(parameters):
    """Flower may pass a Parameters proto or a list; coerce into list[np.ndarray]."""
    if isinstance(parameters, list):
        return parameters
    try:
        return parameters_to_ndarrays(parameters)
    except Exception:
        return parameters

def model_nbytes(model) -> int:
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))

def choose_subjects(mat_path: str, n_clients: int, n_eval: int = 3):
    subjects, _ = load_dreamer_mat(mat_path)
    all_ids = sorted(list(subjects.keys()))
    # train: first N, eval: last n_eval (disjoint where possible)
    train_ids = all_ids[:n_clients]
    eval_ids = [s for s in all_ids[-n_eval:] if s not in train_ids] if n_eval > 0 else []
    return train_ids, eval_ids

def make_eval_fn(target: str, mat_path: str, eval_subjects):
    """Global evaluation on a fixed validation slice."""
    device = torch.device("cpu")
    val_ds = DreamerWindows(mat_path, target=target, subjects_subset=eval_subjects, win_sec=2.0, overlap=0.5)
    val_dl = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    loss_fn = nn.BCEWithLogitsLoss()

    def evaluate(server_round: int, parameters, config):
        model = CNN1D(in_ch=14, n_classes=1).to(device)
        set_parameters(model, _to_ndarrays(parameters))
        model.eval()

        y_true, y_prob, losses = [], [], []
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                losses.append(loss.item())
                y_prob.extend(torch.sigmoid(logits).cpu().numpy().ravel().tolist())
                y_true.extend(yb.cpu().numpy().ravel().tolist())

        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        y_pred = (np.array(y_prob) >= 0.5).astype(int) if y_prob else np.array([], int)
        acc = accuracy_score(y_true, y_pred) if len(y_true) else 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0) if len(y_true) else 0.0
        try:
            auc = roc_auc_score(y_true, y_prob) if len(y_true) else float("nan")
        except Exception:
            auc = float("nan")
        loss = float(np.mean(losses)) if losses else 0.0
        return loss, {"acc": acc, "f1": f1, "auc": float(auc)}

    return evaluate


# ----------------------- strategy with logging -----------------------

class LoggingFedAvg(fl.server.strategy.FedAvg):
    """FedAvg that also writes:
       - metrics.csv (loss/acc/f1/auc per round)
       - global_round_XXX.pt checkpoints
    """
    def __init__(self, *args, outdir="results/fed", **kwargs):
        super().__init__(*args, **kwargs)
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)
        # header once
        with open(os.path.join(self.outdir, "metrics.csv"), "w", newline="") as f:
            csv.writer(f).writerow(["round", "loss", "acc", "f1", "auc"])

    def aggregate_fit(self, server_round, results, failures):
        agg_params, agg_metrics = super().aggregate_fit(server_round, results, failures)
        if agg_params is not None:
            model = CNN1D(in_ch=14, n_classes=1)
            set_parameters(model, _to_ndarrays(agg_params))
            torch.save(model.state_dict(), os.path.join(self.outdir, f"global_round_{server_round:03d}.pt"))
        return agg_params, agg_metrics

    def aggregate_evaluate(self, server_round, results, failures):
        aggregated_loss, aggregated_metrics = super().aggregate_evaluate(server_round, results, failures)

        # Be defensive: aggregated_metrics can be None or might not contain our keys.
        loss = aggregated_loss if aggregated_loss is not None else float("nan")
        acc = float("nan")
        f1 = float("nan")
        auc = float("nan")

        # If the strategy returns a dict of metrics averaged across clients
        if isinstance(aggregated_metrics, dict) and len(aggregated_metrics):
            acc = aggregated_metrics.get("acc", float("nan"))
            f1 = aggregated_metrics.get("f1", float("nan"))
            auc = aggregated_metrics.get("auc", float("nan"))

        with open(os.path.join(self.outdir, "metrics.csv"), "a", newline="") as f:
            csv.writer(f).writerow([server_round, loss, acc, f1, auc])

        return aggregated_loss, aggregated_metrics


# ----------------------- main -----------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", type=str, default="127.0.0.1:0")  # :0 -> auto-pick free port
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--target", type=str, default="arousal", choices=["arousal", "valence"])
    ap.add_argument("--mat", type=str, default="/home/rajeshkumarjogi/Desktop/eeg-fl-emotion/Dataset/DREAMER.mat")
    ap.add_argument("--fraction_fit", type=float, default=0.5)
    ap.add_argument("--fraction_evaluate", type=float, default=1.0)
    ap.add_argument("--out_dir", type=str, default="results/fed")  # <— allow run_fed to set a timestamped folder
    args = ap.parse_args()

    outdir = args.out_dir or "results/fed"
    os.makedirs(outdir, exist_ok=True)

    # Write which subjects were used
    train_ids, eval_ids = choose_subjects(args.mat, n_clients=args.clients, n_eval=3)
    with open(os.path.join(outdir, "subjects.txt"), "w") as f:
        f.write("clients: " + " ".join(train_ids) + "\n")
        f.write("eval: " + " ".join(eval_ids) + "\n")

    # Evaluation hook
    evaluate_fn = make_eval_fn(args.target, args.mat, eval_subjects=eval_ids)

    # Rough comm estimate per round
    dummy = CNN1D(in_ch=14, n_classes=1)
    per_round_bytes = model_nbytes(dummy)
    comm_path = os.path.join(outdir, "comm_log.csv")
    with open(comm_path, "w", newline="") as f:
        csv.writer(f).writerow(["round", "uplink_bytes", "downlink_bytes", "total_bytes"])

    # Strategy: do NOT require all clients to be connected — only what we sample
    min_fit = max(1, ceil(args.fraction_fit * args.clients))
    strategy = LoggingFedAvg(
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        min_fit_clients=min_fit,
        min_available_clients=min_fit,   # proceed as long as enough clients are alive
        evaluate_fn=evaluate_fn,
        outdir=outdir,
    )

    print(f"[Server] FedAvg | rounds={args.rounds} clients={args.clients} "
          f"target={args.target} fraction_fit={args.fraction_fit} outdir={outdir}")

    # Launch Flower server
    fl.server.start_server(
        server_address=args.server,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )

    # Append a simple (constant) comm estimate per round
    uplink = per_round_bytes * args.clients
    downlink = per_round_bytes * args.clients
    total = uplink + downlink
    with open(comm_path, "a", newline="") as f:
        w = csv.writer(f)
        for r in range(1, args.rounds + 1):
            w.writerow([r, uplink, downlink, total])


if __name__ == "__main__":
    main()
