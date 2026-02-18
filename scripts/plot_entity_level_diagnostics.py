from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CLEAN = Path("outputs/tables/distilbert_span_by_label_clean.csv")
PERT  = Path("outputs/tables/distilbert_span_by_label_perturbed.csv")
OUTDIR = Path("outputs/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

def describe_distribution(df: pd.DataFrame, name: str):
    f1 = df["f1"].astype(float).values
    support = (df["tp"] + df["fn"]).astype(int).values

    macro_f1 = float(np.mean(f1))
    median_f1 = float(np.median(f1))
    q1, q3 = np.quantile(f1, [0.25, 0.75])
    iqr = float(q3 - q1)
    pct_zero = float(np.mean(f1 == 0.0) * 100.0)

    # Weighted by support (approx "micro across labels" using span counts)
    weighted_f1 = float(np.average(f1, weights=np.maximum(support, 1)))

    print(f"\n=== Entity-level stats: {name} ===")
    print(f"Num labels: {len(df)}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Weighted-F1 (by span support): {weighted_f1:.4f}")
    print(f"Median F1: {median_f1:.4f}")
    print(f"IQR(F1): {iqr:.4f}  (Q1={q1:.4f}, Q3={q3:.4f})")
    print(f"% labels with F1=0: {pct_zero:.1f}%")

def plot_f1_ranked(df: pd.DataFrame, title: str, outpath: Path, top_n=20, bottom_n=20):
    # show extremes with support info
    df = df.copy()
    df["support"] = (df["tp"] + df["fn"]).astype(int)
    df = df.sort_values("f1", ascending=False)

    top = df.head(top_n)
    bottom = df.tail(bottom_n).sort_values("f1", ascending=True)

    # Combine with a gap
    combo = pd.concat([top, bottom], axis=0)
    labels = [
        f"{r['label']} (n={int(r['support'])})"
        for _, r in combo.iterrows()
    ]
    values = combo["f1"].astype(float).values

    plt.figure(figsize=(12, 7))
    x = np.arange(len(values))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=75, ha="right")
    plt.ylabel("Span F1")
    plt.ylim(0, 1.05)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.show()

def plot_support_vs_f1(df: pd.DataFrame, title: str, outpath: Path):
    df = df.copy()
    df["support"] = (df["tp"] + df["fn"]).astype(int)
    # avoid log(0)
    x = df["support"].clip(lower=1).astype(float).values
    y = df["f1"].astype(float).values

    plt.figure(figsize=(8.5, 6))
    # size encodes FP+FN (difficulty), clipped for readability
    difficulty = (df["fp"] + df["fn"]).astype(int).clip(lower=1).astype(float).values
    sizes = np.clip(difficulty * 6.0, 20, 400)

    plt.scatter(x, y, s=sizes, alpha=0.7)
    plt.xscale("log")
    plt.xlabel("Entity Support (tp+fn), log scale")
    plt.ylabel("Span F1")
    plt.ylim(0, 1.05)
    plt.title(title)

    # Add reference lines for quartiles
    q1, q2, q3 = np.quantile(y, [0.25, 0.5, 0.75])
    plt.axhline(q2, linestyle="--")
    plt.axhline(q1, linestyle=":")
    plt.axhline(q3, linestyle=":")

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.show()

def plot_delta_f1(clean_df: pd.DataFrame, pert_df: pd.DataFrame, outpath: Path, top_n=25):
    c = clean_df.copy()
    p = pert_df.copy()

    c["support"] = (c["tp"] + c["fn"]).astype(int)
    p["support"] = (p["tp"] + p["fn"]).astype(int)

    merged = c.merge(
        p[["label", "f1"]],
        on="label",
        how="left",
        suffixes=("_clean", "_pert")
    )
    merged["f1_pert"] = merged["f1_pert"].fillna(0.0)
    merged["delta_f1"] = merged["f1_clean"].astype(float) - merged["f1_pert"].astype(float)

    # focus on labels with meaningful support to avoid tiny-count noise
    merged = merged.sort_values(["delta_f1", "support"], ascending=[False, False])
    view = merged.head(top_n)

    labels = [f"{r.label} (n={int(r.support)})" for r in view.itertuples(index=False)]
    values = view["delta_f1"].values

    plt.figure(figsize=(12, 7))
    x = np.arange(len(values))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=75, ha="right")
    plt.ylabel("Δ Span F1 (clean − perturbed)")
    plt.title("Robustness Sensitivity by Entity Type (Largest F1 Drops)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.show()

def main():
    clean_df = pd.read_csv(CLEAN)
    pert_df  = pd.read_csv(PERT)

    # 1) Print dissertation-grade summary stats
    describe_distribution(clean_df, "CLEAN")
    describe_distribution(pert_df,  "PERTURBED")

    # 2) Visualization 2A: extremes of entity-level F1 with support
    plot_f1_ranked(
        clean_df,
        "Entity-Level Span F1 (Clean): Top/Bottom Labels with Support",
        OUTDIR / "v2_entity_f1_extremes_clean.png",
        top_n=20,
        bottom_n=20
    )

    # 3) Visualization 2B: Support vs F1 (imbalance effect)
    plot_support_vs_f1(
        clean_df,
        "Entity Support vs Span F1 (Clean) — Imbalance Sensitivity",
        OUTDIR / "v2_support_vs_f1_clean.png"
    )

    # 4) Visualization 2C: Robustness drop per entity type
    plot_delta_f1(
        clean_df,
        pert_df,
        OUTDIR / "v2_delta_f1_by_entity.png",
        top_n=25
    )

    print(f"\n✅ Figures saved to: {OUTDIR.resolve()}")

if __name__ == "__main__":
    main()
