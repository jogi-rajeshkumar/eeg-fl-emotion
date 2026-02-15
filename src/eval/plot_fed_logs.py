import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fed_dir", default="results/fed")
    args = ap.parse_args()

    mpath = os.path.join(args.fed_dir, "metrics.csv")
    if not os.path.exists(mpath):
        raise FileNotFoundError(f"Missing {mpath}")
    m = pd.read_csv(mpath)

    # Ensure numeric (allow NaN) and ensure round exists
    if "round" not in m.columns:
        raise ValueError(f"'round' column missing in {mpath}")
    for col in ["loss", "acc", "f1", "auc"]:
        if col in m.columns:
            m[col] = pd.to_numeric(m[col], errors="coerce")

    # Choose “best” round using whatever metric is available (in this order)
    best_idx = None
    best_metric_txt = "n/a"
    if "auc" in m.columns and m["auc"].notna().any():
        best_idx = m["auc"].idxmax(skipna=True)
        best_metric_txt = f"AUC={m.loc[best_idx,'auc']:.4f}"
    elif "acc" in m.columns and m["acc"].notna().any():
        best_idx = m["acc"].idxmax(skipna=True)
        best_metric_txt = f"ACC={m.loc[best_idx,'acc']:.4f}"
    elif "f1" in m.columns and m["f1"].notna().any():
        best_idx = m["f1"].idxmax(skipna=True)
        best_metric_txt = f"F1={m.loc[best_idx,'f1']:.4f}"
    elif "loss" in m.columns and m["loss"].notna().any():
        best_idx = m["loss"].idxmin(skipna=True)
        best_metric_txt = f"loss={m.loc[best_idx,'loss']:.4f}"

    # Plot everything that exists and has at least one non-NaN
    fig, ax = plt.subplots()
    plotted_any = False
    if "loss" in m.columns and m["loss"].notna().any():
        ax.plot(m["round"], m["loss"], label="loss"); plotted_any = True
    if "auc" in m.columns and m["auc"].notna().any():
        ax.plot(m["round"], m["auc"], label="auc"); plotted_any = True
    if "acc" in m.columns and m["acc"].notna().any():
        ax.plot(m["round"], m["acc"], label="acc"); plotted_any = True
    if "f1" in m.columns and m["f1"].notna().any():
        ax.plot(m["round"], m["f1"], label="f1"); plotted_any = True

    ax.set_xlabel("round")
    title_suffix = ""
    if best_idx is not None and not pd.isna(best_idx):
        title_suffix = f" (best {best_metric_txt} @ round {int(m.loc[best_idx,'round'])})"
    ax.set_title(f"Federated metrics{title_suffix}")
    if plotted_any:
        ax.legend()

    out1 = os.path.join(args.fed_dir, "fed_metrics.png")
    plt.savefig(out1, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out1}")

    # Communication plot (if present)
    clog = os.path.join(args.fed_dir, "comm_log.csv")
    if os.path.exists(clog):
        c = pd.read_csv(clog)
        if {"round","total_bytes"}.issubset(c.columns):
            c["round"] = pd.to_numeric(c["round"], errors="coerce")
            c["total_bytes"] = pd.to_numeric(c["total_bytes"], errors="coerce")
            c = c.dropna(subset=["round","total_bytes"])
            fig2, ax2 = plt.subplots()
            ax2.plot(c["round"], c["total_bytes"])
            ax2.set_xlabel("round"); ax2.set_ylabel("bytes")
            ax2.set_title("Communication per round (total)")
            out2 = os.path.join(args.fed_dir, "fed_comm.png")
            plt.savefig(out2, bbox_inches="tight")
            plt.close(fig2)
            print(f"Saved: {out2}")
        else:
            print("comm_log.csv missing required columns, skipping comm plot.")
    else:
        print("No comm_log.csv found, skipping comm plot.")

if __name__ == "__main__":
    main()
