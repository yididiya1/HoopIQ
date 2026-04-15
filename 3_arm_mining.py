"""
STEP 3 — ASSOCIATION RULE MINING
=================================
Loads transactions.csv, runs FP-Growth, mines rules where
the consequent is a single outcome token (score / miss / turnover / etc.)
Saves results to data/rules.csv

Usage:
    python 3_arm_mining.py
    python 3_arm_mining.py --min_support 0.01 --min_confidence 0.4 --min_lift 1.2
"""

import argparse
import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder

TXN_PATH   = "data/transactions.csv"
RULES_PATH = "data/rules.csv"

OUTCOME_TOKENS = {"score", "miss", "turnover", "foul_score", "foul_miss"}

# Tokens that are structural artifacts of outcomes — they perfectly predict
# an outcome by construction (e.g. "assist" only appears on made shots).
# We block them from appearing as antecedents in rules.
LEAKY_TOKENS = {"assist", "steal", "block", "foul_drawn", "rebound"}

DEFAULT_MIN_SUPPORT    = 0.01
DEFAULT_MIN_CONFIDENCE = 0.35
DEFAULT_MIN_LIFT       = 1.1


def load_transactions(path):
    df   = pd.read_csv(path)
    txns = [row.split(",") for row in df["transaction"].dropna()]
    print(f"Loaded {len(txns):,} transactions.")
    return txns


def encode(txns):
    te  = TransactionEncoder()
    arr = te.fit(txns).transform(txns)
    df  = pd.DataFrame(arr, columns=te.columns_)
    print(f"Encoded: {df.shape[0]:,} transactions x {df.shape[1]} unique tokens")
    return df


def mine(encoded_df, min_support):
    print(f"\nRunning FP-Growth (min_support={min_support})...")
    freq = fpgrowth(encoded_df, min_support=min_support, use_colnames=True)
    print(f"  Frequent itemsets: {len(freq):,}")
    return freq


def generate_rules(freq, min_confidence):
    print(f"Generating rules (min_confidence={min_confidence})...")
    rules = association_rules(freq, metric="confidence", min_threshold=min_confidence)
    print(f"  Raw rules: {len(rules):,}")
    return rules


def filter_outcome_rules(rules, min_lift):
    def single_outcome_consequent(cons):
        return len(cons) == 1 and list(cons)[0] in OUTCOME_TOKENS

    def no_outcome_in_antecedent(ants):
        # Block both outcome tokens AND leaky proxy tokens from antecedents
        blocked = OUTCOME_TOKENS | LEAKY_TOKENS
        return len(set(ants) & blocked) == 0

    mask = (
        rules["consequents"].apply(single_outcome_consequent) &
        rules["antecedents"].apply(no_outcome_in_antecedent) &
        (rules["lift"] >= min_lift)
    )
    filtered = rules[mask].copy()
    print(f"  Outcome rules (lift >= {min_lift}): {len(filtered):,}")
    return filtered


def format_rules(rules):
    rules = rules.copy()
    rules["antecedent_str"] = rules["antecedents"].apply(lambda x: " + ".join(sorted(x)))
    rules["consequent_str"] = rules["consequents"].apply(lambda x: list(x)[0])
    rules["n_antecedents"]  = rules["antecedents"].apply(len)
    for col in ["support", "confidence", "lift", "conviction", "leverage"]:
        if col in rules.columns:
            rules[col] = rules[col].round(4)
    rules = rules.sort_values(
        ["lift", "confidence", "support"], ascending=[False, False, False]
    ).reset_index(drop=True)
    cols = ["antecedent_str", "consequent_str", "n_antecedents",
            "support", "confidence", "lift", "conviction", "leverage"]
    return rules[[c for c in cols if c in rules.columns]]


def print_summary(rules, top_n=20):
    print("\n" + "="*70)
    print(f"TOP {top_n} RULES BY LIFT")
    print("="*70)
    display = rules.head(top_n)[["antecedent_str","consequent_str","support","confidence","lift"]]
    display.columns = ["Antecedent", "Outcome", "Support", "Confidence", "Lift"]
    print(display.to_string(index=False))

    print("\n" + "="*70)
    print("BREAKDOWN BY OUTCOME")
    print("="*70)
    for outcome in sorted(OUTCOME_TOKENS):
        subset = rules[rules["consequent_str"] == outcome]
        if subset.empty:
            continue
        print(f"\n  -> {outcome.upper()}  ({len(subset)} rules, top 5):")
        top = subset.head(5)[["antecedent_str", "confidence", "lift"]]
        top.columns = ["Antecedent", "Conf", "Lift"]
        print(top.to_string(index=False))

    print("\n" + "="*70)
    print("MULTI-TOKEN RULES (lift >= 1.3)")
    print("="*70)
    rich = rules[(rules["n_antecedents"] >= 2) & (rules["lift"] >= 1.3)].head(15)
    if rich.empty:
        print("  None found — try lowering --min_lift")
    else:
        display = rich[["antecedent_str","consequent_str","confidence","lift"]]
        display.columns = ["Antecedent","Outcome","Conf","Lift"]
        print(display.to_string(index=False))


def main(min_support, min_confidence, min_lift):
    txns     = load_transactions(TXN_PATH)
    encoded  = encode(txns)
    freq     = mine(encoded, min_support)

    if freq.empty:
        print("\nERROR: No frequent itemsets. Try lowering --min_support.")
        return

    rules    = generate_rules(freq, min_confidence)
    filtered = filter_outcome_rules(rules, min_lift)

    if filtered.empty:
        print("\nERROR: No outcome rules after filtering.")
        print("  Try: --min_lift 1.0  or  --min_confidence 0.2")
        return

    formatted = format_rules(filtered)
    formatted.to_csv(RULES_PATH, index=False)
    print(f"\nRules saved to: {RULES_PATH}")
    print_summary(formatted)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min_support",    type=float, default=DEFAULT_MIN_SUPPORT)
    parser.add_argument("--min_confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--min_lift",       type=float, default=DEFAULT_MIN_LIFT)
    args = parser.parse_args()
    main(args.min_support, args.min_confidence, args.min_lift)