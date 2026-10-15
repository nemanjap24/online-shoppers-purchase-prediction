from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def setup_plot_style() -> None:
    """
    Podešava zajednički stil za sve grafikone.
    """
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["figure.dpi"] = 120
    plt.rcParams["axes.titlesize"] = 14
    plt.rcParams["axes.labelsize"] = 11


def save_figure(filename: str) -> None:
    """
    Čuva trenutno aktivan grafikon u results/figures.
    """
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    output_path = FIGURES_DIR / filename
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()

    print(f"Figure saved to: {output_path}")


def print_grouped_revenue_rate(
    df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    """
    Prikazuje stopu kupovine po kategorijama izabranog atributa.
    """
    summary = (
        df.groupby(group_column, dropna=False)["Revenue"]
        .agg(
            sessions="count",
            purchases="sum",
            purchase_rate="mean",
        )
        .sort_values("purchase_rate", ascending=False)
    )

    summary["purchase_rate"] = summary["purchase_rate"] * 100

    print(summary.round(2))

    return summary

def plot_purchase_rate_by_visitor_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prikazuje procenat sesija završenih kupovinom prema tipu posetioca.
    Čuva graf i tabelu sa rezultatima.
    """
    summary = (
        df.groupby("VisitorType", dropna=False)["Revenue"]
        .agg(
            sessions="count",
            purchases="sum",
            purchase_rate="mean",
        )
        .reset_index()
    )

    summary["purchase_rate"] *= 100
    summary = summary.sort_values(
        "purchase_rate",
        ascending=False,
    ).reset_index(drop=True)

    tables_dir = PROJECT_ROOT / "results" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary.to_csv(
        tables_dir / "eda_purchase_rate_by_visitor_type.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    sns.barplot(
        data=summary,
        x="VisitorType",
        y="purchase_rate",
        color="steelblue",
        errorbar=None,
        ax=ax,
    )

    overall_rate = df["Revenue"].mean() * 100

    ax.axhline(
        overall_rate,
        color="darkred",
        linestyle="--",
        label=f"Celková miera nákupu: {overall_rate:.2f} %",
    )

    labels = [
        f"{row.purchase_rate:.2f} %\n(n={row.sessions:,})"
        for row in summary.itertuples()
    ]

    ax.bar_label(ax.containers[0], labels=labels, padding=5)

    ax.set_title("Miera dokončenia nákupu podľa typu návštevníka")
    ax.set_xlabel("Typ návštevníka")
    ax.set_ylabel("Podiel sessions s nákupom (%)")
    ax.set_ylim(0, max(summary["purchase_rate"].max(), overall_rate) * 1.3)
    ax.legend(loc="upper right")

    save_figure("eda_purchase_rate_by_visitor_type.png")

    return summary