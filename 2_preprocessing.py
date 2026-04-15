"""
STEP 2 — PREPROCESSING
=======================
Reads raw PlayByPlayV3 CSVs, segments into possessions,
encodes events into clean tokens, labels each possession
with an outcome (score / miss / turnover / foul).

Output: data/transactions.csv — one row per possession

V3 column reference (confirmed from real data):
    actionType  : "Made Shot" | "Missed Shot" | "Turnover" |
                  "Free Throw" | "Rebound" | "Foul" | etc.
    subType     : "Driving Layup Shot" | "Pullup Jump shot" |
                  "Step Back Jump Shot" | "Cutting Dunk Shot" | etc.
    description : full text string
    shotResult  : "Made" | "Missed"

Usage:
    python 2_preprocessing.py
"""

import os
import re
import pandas as pd
from collections import Counter
from tqdm import tqdm

RAW_DIR  = "data/raw"
OUT_PATH = "data/transactions.csv"
os.makedirs("data", exist_ok=True)


# ── Event Taxonomy ─────────────────────────────────────────────────────────────
# Matched against subType first (structured), then description (free text).
# Order matters — first match wins per column.

EVENT_PATTERNS = [
    # ── V3 subType exact / near-exact values ─────────────────────────────────
    (r"pick.?and.?roll|pick-and-roll",            "pick_and_roll"),
    (r"cutting layup|cutting dunk|cutting finger", "cut"),
    (r"driving layup|driving dunk|driving finger", "drive"),
    (r"driving floating|floating jump",            "floater"),
    (r"pull.?up jump|pullup jump",                 "pull_up"),
    (r"step.?back jump",                           "step_back"),
    (r"fadeaway|turnaround fadeaway",              "fadeaway"),
    (r"turnaround jump",                           "fadeaway"),
    (r"hook shot|hook bank",                       "hook_shot"),
    (r"alley.?oop",                                "alley_oop"),
    (r"putback|tip layup|tip dunk",                "putback"),
    (r"running layup|running finger",              "drive"),
    (r"jump bank shot|bank shot",                  "bank_shot"),
    (r"jump shot",                                 "jump_shot"),
    (r"layup",                                     "layup"),
    (r"dunk",                                      "dunk"),
    (r"3pt|3-pt|3 pt|three point",                 "three_pointer"),

    # ── Description free-text patterns ───────────────────────────────────────
    (r"pick.?and.?roll|p&r|pick n roll",           "pick_and_roll"),
    (r"off.?ball.?screen|off ball screen",         "off_ball_screen"),
    (r"flare.?screen",                             "flare_screen"),
    (r"hand.?off|dribble.?hand.?off",              "dribble_handoff"),
    (r"driv",                                      "drive"),
    (r"isol",                                      "isolation"),
    (r"pull.?up",                                  "pull_up"),
    (r"step.?back",                                "step_back"),
    (r"pump.?fake",                                "pump_fake"),
    (r"spin.?move",                                "spin_move"),
    (r"euro.?step",                                "euro_step"),
    (r"post.?up|in the post",                      "post_up"),
    (r"fadeaway|fade away",                        "fadeaway"),
    (r"hook",                                      "hook_shot"),
    (r"alley.?oop",                                "alley_oop"),
    (r"cutting|cutter\b",                          "cut"),
    (r"transition|fast.?break",                    "transition"),
    (r"putback|put.?back|\btip\b",                 "putback"),
    (r"kick.?out",                                 "kick_out"),
    (r"no.?look",                                  "no_look_pass"),
    (r"\bast\b|assist",                            "assist"),
    (r"3pt|3-pt|three.?point",                     "three_pointer"),
    (r"layup|lay.?up",                             "layup"),
    (r"dunk",                                      "dunk"),
    (r"floater|teardrop",                          "floater"),
    (r"mid.?range|midrange",                       "mid_range"),
    (r"jump.?shot|jumper",                         "jump_shot"),
    (r"block|\bblk\b",                             "block"),
    (r"steal|\bstl\b",                             "steal"),
]

COMPILED = [(re.compile(p, re.IGNORECASE), t) for p, t in EVENT_PATTERNS]

OUTCOME_TOKENS = {"score", "miss", "turnover", "foul_score", "foul_miss"}


# ── Outcome Detection ──────────────────────────────────────────────────────────
def get_outcome(row):
    """
    Uses V3's actionType string (confirmed values from real data):
        "Made Shot"   → score
        "Missed Shot" → miss
        "Turnover"    → turnover
        "Free Throw"  → foul_score / foul_miss (last FT of trip only)
    """
    atype = str(row.get("actionType",  "") or "").strip()
    desc  = str(row.get("description", "") or "").strip()

    if atype == "Made Shot":
        return "score"
    if atype == "Missed Shot":
        return "miss"
    if atype == "Turnover":
        return "turnover"
    if atype == "Free Throw":
        m = re.search(r"free throw\s+(\d+)\s+of\s+(\d+)", desc, re.IGNORECASE)
        if m and m.group(1) == m.group(2):
            return "foul_miss" if "MISS" in desc.upper() else "foul_score"
        return None
    return None


# ── Token Extraction ───────────────────────────────────────────────────────────
def extract_tokens(row):
    """
    Match event patterns against subType (structured) + description (free text),
    then append shot-zone token for any field-goal attempt.
    Returns list of matched tokens for this single play row.
    """
    text = (
        str(row.get("subType",     "") or "") + " " +
        str(row.get("description", "") or "")
    ).strip()

    tokens = []
    if text:
        for pattern, token in COMPILED:
            if pattern.search(text):
                tokens.append(token)
    return tokens


# ── Possession Segmenter ───────────────────────────────────────────────────────
LEAKY_TOKENS = {
    "assist", "steal", "block",
    "turnover", "rebound", "foul_drawn",
}


def _get_team(row):
    """Return teamTricode as a string, or None for NaN / blank values."""
    raw = row.get("teamTricode")
    if raw is None:
        return None
    try:
        if pd.isna(raw):
            return None
    except (TypeError, ValueError):
        pass
    return str(raw).strip() or None


def _save_possession(tokens, outcome, possessions, team):
    clean = [t for t in tokens if t not in LEAKY_TOKENS]
    if clean and outcome:
        possessions.append({
            "tokens":  list(set(clean)),
            "outcome": outcome,
            "team":    team,
        })


def segment_possessions(df):
    """
    Tracks actual team possessions, merging through offensive rebounds so that
    multi-attempt possessions (miss → offensive board → putback) carry all their
    move tokens into one transaction.

    Key logic:
        - After a Missed Shot, defer finalising — wait for the rebound.
        - Offensive rebound (same team): possession continues, accumulate more tokens.
        - Defensive rebound (other team): finalise as "miss" and start fresh.
        - Made Shot / Turnover / Foul: finalise immediately.
    """
    possessions    = []
    current_tokens = []
    poss_team      = None   # team currently holding the ball
    pending_miss   = False  # True after a Missed Shot, awaiting rebound verdict

    for _, row in df.iterrows():
        team  = _get_team(row)
        atype = str(row.get("actionType") or "").strip()

        # ── Rebound: decide offensive vs. defensive ───────────────────────────
        if atype == "Rebound":
            if pending_miss:
                if team and poss_team and team == poss_team:
                    # Offensive rebound — possession continues, keep tokens
                    current_tokens.append("off_rebound")
                    pending_miss = False
                else:
                    # Defensive rebound (or team rebound with no tricode) — end it
                    _save_possession(current_tokens, "miss", possessions, poss_team)
                    current_tokens = []
                    poss_team      = team
                    pending_miss   = False
            # Rebound rows carry no move tokens regardless
            continue

        # ── Flush any pending miss that wasn't followed by a rebound ─────────
        if pending_miss:
            _save_possession(current_tokens, "miss", possessions, poss_team)
            current_tokens = []
            poss_team      = team
            pending_miss   = False

        # ── Track which team has the ball ─────────────────────────────────────
        if not poss_team and team:
            poss_team = team
        elif team and poss_team and team != poss_team:
            if atype in ("Made Shot", "Missed Shot", "Turnover"):
                # Different team now acting without a rebound signal — reset
                current_tokens = []
                poss_team      = team

        # ── Accumulate move tokens for this row ───────────────────────────────
        current_tokens.extend(extract_tokens(row))

        # ── Check for a possession-ending outcome ─────────────────────────────
        outcome = get_outcome(row)

        if outcome == "miss":
            pending_miss = True          # defer — need to see the rebound first
        elif outcome is not None:
            _save_possession(current_tokens, outcome, possessions, poss_team)
            current_tokens = []
            poss_team      = None
            pending_miss   = False

    # End of game: flush anything still pending
    if pending_miss and current_tokens:
        _save_possession(current_tokens, "miss", possessions, poss_team)

    return possessions


def build_transaction(poss):
    items = sorted(set(poss["tokens"] + [poss["outcome"]]))
    return ",".join(items)


# ── Main ───────────────────────────────────────────────────────────────────────
def main(team_filter=None):
    raw_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]
    print(f"Found {len(raw_files)} raw game files.\n")

    if team_filter:
        print(f"  Filtering to team: {team_filter.upper()}\n")

    all_possessions = []

    for fname in tqdm(raw_files, desc="Processing games"):
        try:
            df   = pd.read_csv(os.path.join(RAW_DIR, fname))
            poss = segment_possessions(df)
            all_possessions.extend(poss)
        except Exception as e:
            print(f"  WARNING {fname}: {e}")

    if team_filter:
        tf = team_filter.upper()
        all_possessions = [p for p in all_possessions if p.get("team", "").upper() == tf]
        print(f"  Possessions for {tf}: {len(all_possessions):,}")

    all_transactions = [build_transaction(p) for p in all_possessions]
    print(f"\nTotal possessions: {len(all_transactions):,}")

    out_df = pd.DataFrame({"transaction": all_transactions})
    out_df.to_csv(OUT_PATH, index=False)
    print(f"Saved to: {OUT_PATH}")

    # Summary
    all_tokens = []
    for txn in all_transactions:
        all_tokens.extend(txn.split(","))
    counts = Counter(all_tokens)

    print("\nTop 25 tokens:")
    for token, count in counts.most_common(25):
        print(f"  {token:<25} {count:>7,}")

    print("\nOutcome distribution:")
    total = len(all_transactions)
    for o in ["score", "miss", "turnover", "foul_score", "foul_miss"]:
        n = counts.get(o, 0)
        print(f"  {o:<15} {n:>7,}  ({100*n/total:.1f}%)" if total else f"  {o:<15} 0")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--team", default=None,
                        help="Only keep possessions for this team, e.g. GSW or SAC")
    args = parser.parse_args()
    main(team_filter=args.team)