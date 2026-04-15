"""
STEP 4 — VISUALIZATION
=======================
Produces 3 visualizations from rules.csv:

1. Rule Network Graph  — nodes=tokens, edges=rules, colored by outcome
2. Lift Heatmap        — single-token actions vs outcomes, cell=max lift
3. Top Rules Bar Chart — top rules per outcome, annotated with confidence

Usage:
    python 4_visualization.py
    python 4_visualization.py --outcome score --top_n 40
"""

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import networkx as nx

OUTPUT_DIR = "data/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTCOME_COLORS = {
    "score":      "#2ecc71",
    "miss":       "#e74c3c",
    "turnover":   "#e67e22",
    "foul_score": "#3498db",
    "foul_miss":  "#9b59b6",
}


def load_rules(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rules.")
    return df


# ── Plot 1: Network Graph ──────────────────────────────────────────────────────
def plot_network(rules, top_n=50, outcome_filter=None):
    subset = rules.copy()
    if outcome_filter:
        subset = subset[subset["consequent_str"] == outcome_filter]
    subset = subset.head(top_n)

    G = nx.DiGraph()
    for _, row in subset.iterrows():
        ants    = row["antecedent_str"].split(" + ")
        outcome = row["consequent_str"]
        color   = OUTCOME_COLORS.get(outcome, "#95a5a6")
        for ant in ants:
            G.add_node(ant,     node_type="action",  )
            G.add_node(outcome, node_type="outcome")
            G.add_edge(ant, outcome,
                       weight=row["confidence"],
                       color=color)

    if not G.nodes:
        print("  No nodes to plot — skipping network.")
        return

    fig, ax = plt.subplots(figsize=(18, 12))
    fig.patch.set_facecolor("#0f0f23")
    ax.set_facecolor("#0f0f23")

    pos = nx.spring_layout(G, k=2.8, seed=42)

    node_colors, node_sizes = [], []
    for node in G.nodes():
        if G.nodes[node].get("node_type") == "outcome":
            node_colors.append(OUTCOME_COLORS.get(node, "#95a5a6"))
            node_sizes.append(3000)
        else:
            node_colors.append("#4a90d9")
            node_sizes.append(1000)

    nx.draw_networkx_edges(G, pos,
        edge_color=[G[u][v]["color"] for u, v in G.edges()],
        width=[G[u][v]["weight"] * 4 for u, v in G.edges()],
        alpha=0.65, arrows=True, arrowsize=18,
        connectionstyle="arc3,rad=0.1", ax=ax)

    nx.draw_networkx_nodes(G, pos,
        node_color=node_colors, node_size=node_sizes, alpha=0.95, ax=ax)

    labels = {n: n.replace("_", "\n") for n in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels,
        font_color="white", font_size=7, font_weight="bold", ax=ax)

    legend_patches = [
        mpatches.Patch(color=c, label=o.replace("_", " ").title())
        for o, c in OUTCOME_COLORS.items()
    ]
    ax.legend(handles=legend_patches, loc="upper left",
              facecolor="#1a1a2e", labelcolor="white", fontsize=9)

    title = "NBA Possession ARM — Rule Network"
    if outcome_filter:
        title += f"  ({outcome_filter.upper()} only)"
    ax.set_title(title, color="white", fontsize=14, pad=15)
    ax.axis("off")

    fname = os.path.join(OUTPUT_DIR, f"network{'_'+outcome_filter if outcome_filter else ''}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {fname}")


# ── Plot 2: Lift Heatmap ───────────────────────────────────────────────────────
def plot_lift_heatmap(rules, top_actions=20):
    single = rules[~rules["antecedent_str"].str.contains(r"\+")].copy()
    if single.empty:
        print("  No single-token rules for heatmap — skipping.")
        return

    pivot = single.pivot_table(
        index="antecedent_str", columns="consequent_str",
        values="lift", aggfunc="max"
    ).fillna(1.0)

    pivot.index   = [i.replace("_", " ") for i in pivot.index]
    pivot.columns = [c.replace("_", " ").title() for c in pivot.columns]
    pivot = pivot.loc[pivot.max(axis=1).nlargest(top_actions).index]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns)*2.5), max(7, top_actions*0.5)))
    sns.heatmap(pivot, cmap="RdYlGn", center=1.0, annot=True, fmt=".2f",
                linewidths=0.5, linecolor="#ccc", ax=ax,
                cbar_kws={"label": "Lift  (>1 = positive association)"})
    ax.set_title("Lift Heatmap: Basketball Action → Outcome", fontsize=13, pad=12)
    ax.set_xlabel("Outcome", fontsize=11)
    ax.set_ylabel("Action Token", fontsize=11)
    ax.tick_params(axis="x", rotation=25)
    ax.tick_params(axis="y", rotation=0)

    fname = os.path.join(OUTPUT_DIR, "lift_heatmap.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ── Plot 3: Top Rules Bar Chart ────────────────────────────────────────────────
def plot_top_rules(rules, top_n=20):
    outcomes = sorted(rules["consequent_str"].unique())
    fig, axes = plt.subplots(1, len(outcomes),
                             figsize=(8*len(outcomes), 8), squeeze=False)

    for i, outcome in enumerate(outcomes):
        ax     = axes[0][i]
        subset = rules[rules["consequent_str"] == outcome].head(top_n)
        color  = OUTCOME_COLORS.get(outcome, "#95a5a6")

        if subset.empty:
            ax.set_title(f"No rules for {outcome}")
            continue

        labels = [r.replace(" + ", " +\n") for r in subset["antecedent_str"][::-1]]
        bars   = ax.barh(labels, subset["lift"][::-1].values,
                         color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax.axvline(x=1.0, color="black", linestyle="--", linewidth=1.2, alpha=0.5)
        ax.set_title(f"→ {outcome.upper()}", fontsize=12, color=color, fontweight="bold")
        ax.set_xlabel("Lift")
        ax.tick_params(axis="y", labelsize=8)

        for bar, conf in zip(bars, subset["confidence"][::-1].values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                    f"conf={conf:.2f}", va="center", ha="left", fontsize=7, color="#444")

    fig.suptitle(f"Top {top_n} Rules by Lift per Outcome", fontsize=14, y=1.01)
    plt.tight_layout()
    fname = os.path.join(OUTPUT_DIR, "top_rules_by_outcome.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {fname}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main(top_n, outcome_filter):
    rules = load_rules("data/rules.csv")

    print("\n[1/3] Network graph...")
    plot_network(rules, top_n=top_n, outcome_filter=outcome_filter or None)

    print("[2/3] Lift heatmap...")
    plot_lift_heatmap(rules)

    print("[3/3] Top rules bar chart...")
    plot_top_rules(rules, top_n=20)

    print(f"\nAll plots saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n",   type=int, default=50)
    parser.add_argument("--outcome", type=str, default="",
                        help="Filter network: score | miss | turnover")
    args = parser.parse_args()
    main(args.top_n, args.outcome)