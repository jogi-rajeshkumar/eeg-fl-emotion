# Robustly compare central vs FL even if AUC is NaN/missing
import os, json, argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_report(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        try:
            return json.load(f)
        except Exception:
            return None

def pick_best_row(df):
    """Pick best FL round using AUC > ACC > F1 > (lowest) loss."""
    if df.empty:
        return None
    for col, kind in [("auc","max"), ("acc","max"), ("f1","max"), ("loss","min")]:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                return df.loc[s.idxmax()] if kind=="max" else df.loc[s.idxmin()]
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--central_arousal", default="results/central/arousal/report.json")
    ap.add_argument("--central_valence", default="results/central/valence/report.json")
    ap.add_argument("--fed_dir", default="results/fed")
    args = ap.parse_args()

    cen_a = load_report(args.central_arousal)
    cen_v = load_report(args.central_valence)

    mpath = os.path.join(args.fed_dir, "metrics.csv")
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"Missing {mpath}")
    fed = pd.read_csv(mpath)
    # numeric coercion
    for col in ["round","loss","acc","f1","auc"]:
        if col in fed.columns:
            fed[col] = pd.to_numeric(fed[col], errors="coerce")

    best = pick_best_row(fed)

    rows = []
    if cen_a:
        rows.append({"model":"Central-Arousal",
                     "acc": cen_a.get("acc", float("nan")),
                     "f1":  cen_a.get("f1",  float("nan")),
                     "auc": cen_a.get("auc", float("nan"))})
    if cen_v:
        rows.append({"model":"Central-Valence",
                     "acc": cen_v.get("acc", float("nan")),
                     "f1":  cen_v.get("f1",  float("nan")),
                     "auc": cen_v.get("auc", float("nan"))})
    if best is not None:
        model_name = f"FedAvg-best(r{int(best['round'])})" if "round" in best else "FedAvg-best"
        rows.append({"model": model_name,
                     "acc": best.get("acc", float("nan")),
                     "f1":  best.get("f1", float("nan")),
                     "auc": best.get("auc", float("nan"))})

    df = pd.DataFrame(rows)
    out_csv = os.path.join(args.fed_dir, "central_vs_fed.csv")
    df.to_csv(out_csv, index=False)

    # Bar plot for AUC (only for rows that have AUC)
    plt.figure()
    plot_df = df.copy()
    plot_df["auc"] = pd.to_numeric(plot_df["auc"], errors="coerce")
    plt.bar(plot_df["model"], plot_df["auc"])
    plt.ylabel("AUC")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    out_png = os.path.join(args.fed_dir, "central_vs_fed_auc.png")
    plt.savefig(out_png)
    plt.close()
    print("Wrote:", out_csv, out_png)

if __name__ == "__main__":
    main()
    
    
