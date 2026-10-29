"""Exploratory data analysis: default rates by segment and chi-square tests of association."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu

from data_pipeline import ROOT, build_dataset

REPORTS = ROOT / "reports"
CATEGORICAL = ["sex", "job", "housing", "saving_accounts", "checking_account", "purpose", "age_band"]
NUMERIC = ["age", "credit_amount", "duration", "monthly_burden"]


def run_eda():
    df = build_dataset()
    REPORTS.mkdir(exist_ok=True)
    lines = ["# EDA summary", "", f"Rows: {len(df)}  |  Default rate: {df.is_default.mean():.1%}", ""]

    lines += ["## Categorical features (chi-square test vs default)", "",
              "| Feature | chi2 | p-value | Highest-risk group (default rate) |", "|---|---|---|---|"]
    for col in CATEGORICAL:
        table = df.groupby(col).is_default.agg(["count", "mean"])
        chi2, p, _, _ = chi2_contingency(df.groupby([col, "is_default"]).size().unstack(fill_value=0))
        worst = table["mean"].idxmax()
        lines.append(f"| {col} | {chi2:.1f} | {p:.4f} | {worst} ({table.loc[worst, 'mean']:.0%}) |")

    lines += ["", "## Numeric features (Mann-Whitney U, default vs non-default)", "",
              "| Feature | median good | median bad | p-value |", "|---|---|---|---|"]
    for col in NUMERIC:
        good, bad = df.loc[df.is_default == 0, col], df.loc[df.is_default == 1, col]
        lines.append(f"| {col} | {good.median():.1f} | {bad.median():.1f} | {mannwhitneyu(good, bad).pvalue:.4f} |")

    (REPORTS / "eda_summary.md").write_text("\n".join(lines), encoding="utf-8")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, col in zip(axes, ["checking_account", "saving_accounts", "purpose"]):
        df.groupby(col).is_default.mean().sort_values().plot.barh(ax=ax, title=f"Default rate by {col}")
    plt.tight_layout()
    plt.savefig(REPORTS / "default_rate_by_segment.png", dpi=120)

    plt.figure(figsize=(6, 4))
    sns.heatmap(df[NUMERIC + ["is_default"]].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.tight_layout()
    plt.savefig(REPORTS / "correlation_heatmap.png", dpi=120)
    print("\n".join(lines))


if __name__ == "__main__":
    run_eda()
