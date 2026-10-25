
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =========================
# CONFIG
# =========================
ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "train-balanced-sarcasm.csv"
OUT = ROOT_DIR / "results" / "dataset_insights"
OUT.mkdir(parents=True, exist_ok=True)

# Load only columns required for analysis
df = pd.read_csv(
    DATA_PATH,
    usecols=lambda c: c in {
        "label", "comment", "subreddit", "score"
    },
    low_memory=False
)

required = {"label", "comment"}
if not required.issubset(df.columns):
    raise ValueError(f"Required columns missing. Found: {df.columns.tolist()}")

print("\n===== DATASET OVERVIEW =====")
print("Rows:", len(df))
print("Columns:", df.columns.tolist())
print("\nMissing values:")
print(df.isna().sum())

# Normalize labels for basic analysis
df["label"] = pd.to_numeric(df["label"], errors="coerce")
df = df[df["label"].isin([0, 1])].copy()

# Clean text
df["comment"] = df["comment"].fillna("").astype(str)
df["comment_length"] = df["comment"].str.len()
df["word_count"] = df["comment"].str.split().str.len()

# =========================
# 1. CLASS DISTRIBUTION
# =========================
counts = df["label"].value_counts().sort_index()
names = {0: "Non-sarcastic", 1: "Sarcastic"}

print("\n===== CLASS DISTRIBUTION =====")
print(counts.rename(index=names))
print("\nPercentages:")
print((counts / counts.sum() * 100).round(2))

plt.figure(figsize=(8, 5))
bars = plt.bar(
    [names[i] for i in counts.index],
    counts.values
)
for bar, value in zip(bars, counts.values):
    plt.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"{value:,}",
        ha="center",
        va="bottom"
    )
plt.title("Sarcasm Dataset: Class Distribution")
plt.ylabel("Number of comments")
plt.tight_layout()
plt.savefig(OUT / "class_distribution.png", dpi=300)
plt.close()

# =========================
# 2. COMMENT LENGTH
# =========================
print("\n===== COMMENT LENGTH =====")
print(df.groupby("label")["word_count"].describe().round(2))

# Remove extreme lengths for readable visualization only
plot_df = df[df["word_count"] <= 150]

plt.figure(figsize=(9, 5))
for label in [0, 1]:
    subset = plot_df.loc[
        plot_df["label"] == label, "word_count"
    ]
    plt.hist(
        subset,
        bins=50,
        alpha=0.55,
        label=names[label]
    )

plt.title("Comment Length Distribution")
plt.xlabel("Words per comment (up to 150)")
plt.ylabel("Number of comments")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "comment_length_distribution.png", dpi=300)
plt.close()

# =========================
# 3. SUBREDDIT ANALYSIS
# =========================
if "subreddit" in df.columns:
    sub = df.dropna(subset=["subreddit"]).copy()

    # Minimum support to avoid tiny groups
    sub_stats = sub.groupby("subreddit").agg(
        total_comments=("label", "size"),
        sarcastic=("label", "sum")
    )
    sub_stats = sub_stats[sub_stats["total_comments"] >= 500]
    sub_stats["sarcasm_rate"] = (
        sub_stats["sarcastic"] / sub_stats["total_comments"] * 100
    )

    top_sub = sub_stats.sort_values(
        "sarcasm_rate", ascending=False
    ).head(15)

    print("\n===== TOP SUBREDDITS BY SARCASM RATE =====")
    print(top_sub.round(2))

    if not top_sub.empty:
        plt.figure(figsize=(11, 6))
        top_sub["sarcasm_rate"].sort_values().plot(kind="barh")
        plt.title("Top Subreddits by Sarcasm Rate (min 500 comments)")
        plt.xlabel("Sarcastic comments (%)")
        plt.tight_layout()
        plt.savefig(OUT / "subreddit_sarcasm_rate.png", dpi=300)
        plt.close()

# =========================
# 4. SCORE ANALYSIS
# =========================
if "score" in df.columns:
    df["score"] = pd.to_numeric(df["score"], errors="coerce")

    print("\n===== SCORE STATISTICS =====")
    print(df.groupby("label")["score"].describe().round(2))

    score_df = df.dropna(subset=["score"])

    # Clip only for visualization
    lo = score_df["score"].quantile(0.01)
    hi = score_df["score"].quantile(0.99)
    score_plot = score_df[
        score_df["score"].between(lo, hi)
    ]

    plt.figure(figsize=(9, 5))
    for label in [0, 1]:
        vals = score_plot.loc[
            score_plot["label"] == label, "score"
        ]
        plt.hist(
            vals,
            bins=60,
            alpha=0.5,
            label=names[label]
        )

    plt.title("Reddit Score Distribution by Label (1st-99th percentile)")
    plt.xlabel("Score")
    plt.ylabel("Number of comments")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "score_distribution.png", dpi=300)
    plt.close()

# =========================
# 5. DATA QUALITY
# =========================
print("\n===== DATA QUALITY =====")
print("Empty comments:", (df["comment"].str.strip() == "").sum())
print("Exact duplicate comments:", df["comment"].duplicated().sum())
print("Zero-word comments:", (df["word_count"] == 0).sum())

# Exact repeated texts can have different labels.
conflicts = (
    df.groupby("comment")["label"].nunique().gt(1).sum()
)
print("Unique comment texts with conflicting labels:", conflicts)

# Save summary
summary = pd.DataFrame({
    "label": counts.index.map(names),
    "count": counts.values,
    "percentage": (counts.values / counts.sum() * 100)
})
summary.to_csv(OUT / "class_summary.csv", index=False)

print(f"\nAll insights saved in: {OUT.resolve()}")
