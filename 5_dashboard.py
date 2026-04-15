"""
STEP 5 — HTML DASHBOARD  (HoopIQ style)
========================================
Multi-page SPA with sidebar navigation:
    Overview      — possession stats, scoring breakdown, top sequences
    Pattern Rules — full ARM table with token badges, sort/search/filter
    Play Suggester— filter rules by style / confidence / lift / length
    Benchmark     — FP-Growth sweep across support thresholds

Usage:
    python 5_dashboard.py
    python 5_dashboard.py --team SAC --season 2022-23 --no-benchmark
"""

import argparse, json, os, time
import pandas as pd
from collections import Counter

RULES_PATH = "data/rules.csv"
TXN_PATH   = "data/transactions.csv"
RAW_DIR    = "data/raw"
OUT_PATH   = "data/dashboard.html"

OUTCOME_TOKENS = {"score", "miss", "turnover", "foul_score", "foul_miss"}
LEAKY_TOKENS   = {"assist", "steal", "block", "turnover", "rebound", "foul_drawn"}

TOKEN_LABELS = {
    "jump_shot":"Jump Shot","layup":"Layup","three_pointer":"3-Pointer",
    "drive":"Drive","cut":"Cut","dunk":"Dunk","pick_and_roll":"Pick & Roll",
    "pull_up":"Pull-Up","step_back":"Step Back","fadeaway":"Fadeaway",
    "floater":"Floater","alley_oop":"Alley-Oop","putback":"Putback",
    "off_rebound":"Off. Rebound","isolation":"Isolation","post_up":"Post-Up",
    "hook_shot":"Hook Shot","bank_shot":"Bank Shot","transition":"Transition",
    "euro_step":"Euro Step","spin_move":"Spin Move","dribble_handoff":"DHO",
    "off_ball_screen":"Off-Ball Screen","flare_screen":"Flare Screen",
    "pump_fake":"Pump Fake","mid_range":"Mid-Range","no_look_pass":"No-Look",
}
TOKEN_COLORS = {
    "jump_shot":"#3b82f6","layup":"#22c55e","three_pointer":"#f97316",
    "drive":"#a855f7","cut":"#06b6d4","dunk":"#ef4444",
    "pick_and_roll":"#eab308","pull_up":"#ec4899","step_back":"#84cc16",
    "fadeaway":"#f59e0b","floater":"#14b8a6","alley_oop":"#8b5cf6",
    "putback":"#fb923c","off_rebound":"#94a3b8","isolation":"#dc2626",
    "post_up":"#7c3aed","hook_shot":"#0891b2","bank_shot":"#059669",
    "transition":"#2563eb","euro_step":"#db2777","spin_move":"#9333ea",
    "dribble_handoff":"#6366f1","off_ball_screen":"#0d9488",
    "flare_screen":"#0284c7","pump_fake":"#64748b","mid_range":"#78716c",
    "no_look_pass":"#475569",
}

# ── Data loading ──────────────────────────────────────────────────────────────
def load_data(rules_path, txn_path):
    rules_df     = pd.read_csv(rules_path)
    transactions = [r.split(",") for r in pd.read_csv(txn_path)["transaction"].dropna()]
    return rules_df, transactions

# ── Stats computation ─────────────────────────────────────────────────────────
def compute_stats(rules_df, transactions):
    total = len(transactions)

    oc = Counter()
    for txn in transactions:
        for t in txn:
            if t in OUTCOME_TOKENS: oc[t] += 1

    freq, score_cnt = Counter(), Counter()
    for txn in transactions:
        outcome = next((t for t in txn if t in OUTCOME_TOKENS), None)
        for t in txn:
            if t not in OUTCOME_TOKENS:
                freq[t] += 1
                if outcome == "score": score_cnt[t] += 1

    token_score_rate = {t: round(score_cnt[t]/freq[t], 3) for t in freq if freq[t]}

    SHOW = {"layup","dunk","three_pointer","jump_shot","drive","cut",
            "alley_oop","putback","floater","fadeaway","step_back",
            "pull_up","hook_shot","off_rebound"}
    scoring_bd = Counter()
    for txn in transactions:
        if "score" in txn:
            for t in txn:
                if t in SHOW: scoring_bd[t] += 1

    seq_count, seq_scored = Counter(), Counter()
    for txn in transactions:
        outcome = next((t for t in txn if t in OUTCOME_TOKENS), None)
        moves   = tuple(sorted(t for t in txn if t not in OUTCOME_TOKENS))
        if moves:
            seq_count[moves] += 1
            if outcome == "score": seq_scored[moves] += 1

    top_seqs = []
    for moves, cnt in seq_count.most_common(15):
        sc = seq_scored[moves]
        top_seqs.append({
            "moves":      list(moves),
            "count":      cnt,
            "support":    round(cnt / total, 4),
            "scored":     sc,
            "scores_pct": round(sc / cnt * 100) if cnt else 0,
            "scores":     sc > cnt * 0.5,
        })

    return {
        "total_possessions": total,
        "scoring_rate":      round(oc.get("score",0)/total*100, 1) if total else 0,
        "outcome_counts":    dict(oc),
        "token_freq":        dict(freq.most_common(20)),
        "token_score_rate":  token_score_rate,
        "scoring_breakdown": dict(scoring_bd),
        "top_sequences":     top_seqs,
    }

# ── Benchmark ─────────────────────────────────────────────────────────────────
def run_benchmark(transactions):
    from mlxtend.frequent_patterns import fpgrowth, association_rules
    from mlxtend.preprocessing import TransactionEncoder
    print("  Benchmark sweep:")
    te  = TransactionEncoder()
    enc = pd.DataFrame(te.fit(transactions).transform(transactions), columns=te.columns_)
    results = []
    for thresh in [0.02, 0.03, 0.05, 0.07, 0.10, 0.15]:
        t0   = time.perf_counter()
        freq = fpgrowth(enc, min_support=thresh, use_colnames=True)
        if not freq.empty:
            raw = association_rules(freq, metric="confidence", min_threshold=0.35)
            n   = len(raw[
                raw["consequents"].apply(lambda x: len(x)==1 and list(x)[0] in OUTCOME_TOKENS) &
                raw["antecedents"].apply(lambda x: not(set(x) & (OUTCOME_TOKENS|LEAKY_TOKENS))) &
                (raw["lift"] >= 1.1)
            ])
        else:
            n = 0
        elapsed = time.perf_counter() - t0
        results.append({"support_pct":f"{int(thresh*100)}%","support":thresh,
                         "rules":n,"time":round(elapsed,4),"itemsets":len(freq)})
        print(f"    {int(thresh*100):>3}%  {n:>3} rules  {elapsed:.4f}s  {len(freq)} itemsets")
    return results

# ── HTML builder ──────────────────────────────────────────────────────────────
def build_html(rules_df, stats, benchmark, game_count, team="ALL", season="2022-23"):
    bm_times = [b["time"] for b in benchmark]
    bm_rules = [b["rules"] for b in benchmark]
    meta = {
        "team":team,"season":season,"game_count":game_count,
        "rules_count":len(rules_df),
        "bm_fastest":  min(bm_times) if bm_times else 0,
        "bm_avg":      round(sum(bm_times)/len(bm_times),4) if bm_times else 0,
        "bm_max_rules":max(bm_rules) if bm_rules else 0,
    }
    html = HTML_TEMPLATE
    for k, v in {
        "__RULES_JSON__":       json.dumps(rules_df.to_dict(orient="records")),
        "__STATS_JSON__":       json.dumps(stats),
        "__BENCHMARK_JSON__":   json.dumps(benchmark),
        "__META_JSON__":        json.dumps(meta),
        "__TOKEN_LABELS_JSON__":json.dumps(TOKEN_LABELS),
        "__TOKEN_COLORS_JSON__":json.dumps(TOKEN_COLORS),
    }.items():
        html = html.replace(k, v)
    return html

# ══════════════════════════════════════════════════════════════════════════════
HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>HoopIQ — Play Analytics</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
<style>
/* ── Reset & vars ─────────────────────────────────────────── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#0d1117;--sb:#0a0e16;--card:#131929;--card2:#192033;
  --border:#1e2d42;--border2:#253347;
  --text:#e2e8f0;--muted:#4b6080;--muted2:#6b84a0;
  --orange:#f97316;--orange2:#ea6a0a;
  --green:#22c55e;--red:#ef4444;--blue:#3b82f6;
  --radius:8px;
}
body{background:var(--bg);color:var(--text);font-family:'Inter',system-ui,sans-serif;
     font-size:13px;line-height:1.5;display:flex;min-height:100vh;overflow:hidden}
button,input,select{font-family:inherit}
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border2);border-radius:3px}

/* ── Sidebar ──────────────────────────────────────────────── */
#sidebar{
  width:200px;min-width:200px;background:var(--sb);
  border-right:1px solid var(--border);
  display:flex;flex-direction:column;height:100vh;position:sticky;top:0;
  padding:0 0 16px;
}
.sb-logo{
  padding:22px 20px 18px;
  border-bottom:1px solid var(--border);
  margin-bottom:8px;
}
.sb-logo-name{
  font-size:18px;font-weight:800;letter-spacing:1px;
  color:#fff;
}
.sb-logo-name span{color:var(--orange)}
.sb-logo-sub{font-size:10px;color:var(--muted);letter-spacing:0.5px;margin-top:2px}
.nav-item{
  display:flex;align-items:center;gap:10px;
  padding:10px 20px;font-size:12px;font-weight:500;color:var(--muted2);
  cursor:pointer;transition:all .15s;border-left:3px solid transparent;
  text-decoration:none;
}
.nav-item:hover{color:var(--text);background:rgba(255,255,255,.03)}
.nav-item.active{
  color:var(--orange);background:rgba(249,115,22,.07);
  border-left-color:var(--orange);
}
.nav-icon{font-size:14px;width:16px;text-align:center;opacity:.7}
.nav-item.active .nav-icon{opacity:1}
.sb-footer{margin-top:auto;padding:12px 20px;font-size:10px;color:var(--muted)}
.engine-dot{
  display:inline-block;width:6px;height:6px;border-radius:50%;
  background:var(--green);margin-right:6px;
  box-shadow:0 0 6px var(--green);
}

/* ── Main ─────────────────────────────────────────────────── */
#main{flex:1;display:flex;flex-direction:column;height:100vh;overflow:hidden}
#topbar{
  height:52px;min-height:52px;
  display:flex;align-items:center;justify-content:space-between;
  padding:0 28px;border-bottom:1px solid var(--border);
  background:var(--sb);
}
.page-title{font-size:11px;font-weight:700;letter-spacing:2px;color:var(--muted2)}
.topbar-meta{display:flex;gap:6px;align-items:center}
.meta-pill{
  background:var(--card2);border:1px solid var(--border2);border-radius:6px;
  padding:4px 10px;font-size:11px;font-weight:600;color:var(--muted2);
}
.meta-pill.highlight{color:var(--orange);border-color:rgba(249,115,22,.3);
  background:rgba(249,115,22,.08)}
#content{flex:1;overflow-y:auto;padding:24px 28px 48px}

/* ── Pages ────────────────────────────────────────────────── */
.page{display:none}.page.active{display:block}

/* ── Stat cards ───────────────────────────────────────────── */
.stat-row{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:22px}
@media(max-width:900px){.stat-row{grid-template-columns:repeat(2,1fr)}}
.stat-card{
  background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
  padding:18px 20px;transition:border-color .2s;
}
.stat-card:hover{border-color:var(--border2)}
.stat-card-label{font-size:10px;font-weight:600;color:var(--muted);
  text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px}
.stat-card-value{font-size:28px;font-weight:800;color:var(--text);
  letter-spacing:-0.5px;line-height:1}
.stat-card-sub{font-size:11px;color:var(--muted);margin-top:5px}
.stat-card-value.orange{color:var(--orange)}
.stat-card-value.green{color:var(--green)}

/* ── Two-col grid ─────────────────────────────────────────── */
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:22px}
@media(max-width:960px){.two-col{grid-template-columns:1fr}}

/* ── Cards ────────────────────────────────────────────────── */
.card{
  background:var(--card);border:1px solid var(--border);
  border-radius:var(--radius);padding:18px 20px;
}
.card-header{
  display:flex;align-items:center;justify-content:space-between;
  margin-bottom:16px;
}
.card-title{font-size:11px;font-weight:700;letter-spacing:1.5px;
  color:var(--muted2);text-transform:uppercase}
.card-badge{
  font-size:10px;font-weight:600;padding:3px 8px;border-radius:4px;
  background:rgba(249,115,22,.12);color:var(--orange);
  border:1px solid rgba(249,115,22,.2);
}

/* ── Token badges ─────────────────────────────────────────── */
.tok{
  display:inline-flex;align-items:center;gap:4px;
  font-size:11px;font-weight:500;padding:2px 8px;border-radius:4px;
  white-space:nowrap;margin:1px;
}
.tok-dot{width:6px;height:6px;border-radius:50%;flex-shrink:0}
.tok-ind{font-size:10px;font-weight:700;margin-left:1px}

/* ── Donut chart ──────────────────────────────────────────── */
#donut-svg{display:block;margin:0 auto}
.donut-legend{
  display:flex;flex-wrap:wrap;gap:8px 14px;
  justify-content:center;margin-top:12px;
}
.legend-item{display:flex;align-items:center;gap:5px;font-size:11px;color:var(--muted2)}
.legend-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0}

/* ── Horizontal bars (overview freq) ─────────────────────── */
.hbar-row{
  display:flex;align-items:center;gap:8px;margin-bottom:6px;
}
.hbar-label{width:110px;font-size:11px;color:var(--muted2);text-align:right;
  flex-shrink:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.hbar-track{flex:1;height:10px;background:var(--card2);border-radius:4px;overflow:hidden}
.hbar-fill{height:100%;border-radius:4px;transition:width .5s}
.hbar-count{font-size:10px;color:var(--muted);min-width:36px;text-align:right}

/* ── Sequences table ──────────────────────────────────────── */
.seq-table{width:100%;table-layout:fixed;border-collapse:collapse;font-size:12px}
.seq-table th{
  padding:8px 12px;text-align:left;font-size:10px;font-weight:600;
  color:var(--muted);text-transform:uppercase;letter-spacing:.6px;
  border-bottom:1px solid var(--border2);background:var(--card2);
}
.seq-table td{padding:8px 12px;border-bottom:1px solid var(--border)}
.seq-table tr:last-child td{border-bottom:none}
.seq-table tr:hover td{background:rgba(255,255,255,.02)}
.seq-num{color:var(--muted);font-size:11px;font-weight:600}
.seq-tokens{display:flex;flex-wrap:wrap;gap:2px}
.sup-bar{
  display:flex;align-items:center;gap:6px;min-width:0;
}
.sup-track{width:60px;min-width:60px;max-width:60px;height:4px;background:var(--card2);border-radius:2px;flex-shrink:0}
.sup-fill{height:100%;background:var(--blue);border-radius:2px}
.sup-text{font-size:11px;color:var(--muted2);white-space:nowrap}
.scores-pill{
  font-size:10px;font-weight:700;padding:2px 8px;border-radius:4px;
}
.scores-yes{background:rgba(34,197,94,.12);color:var(--green);border:1px solid rgba(34,197,94,.2)}
.scores-no{background:rgba(239,68,68,.1);color:var(--red);border:1px solid rgba(239,68,68,.15)}

/* ── Rules table ──────────────────────────────────────────── */
.rules-toolbar{
  display:flex;align-items:center;gap:10px;margin-bottom:16px;flex-wrap:wrap;
}
.tb-select,.tb-search{
  background:var(--card2);border:1px solid var(--border2);border-radius:6px;
  padding:7px 12px;font-size:12px;color:var(--text);transition:border-color .15s;
}
.tb-select{cursor:pointer}
.tb-search{width:200px}
.tb-search::placeholder{color:var(--muted)}
.tb-select:focus,.tb-search:focus{outline:none;border-color:rgba(249,115,22,.4)}
.tb-count{font-size:11px;color:var(--muted);margin-left:auto}
.rules-section-label{
  font-size:10px;font-weight:700;letter-spacing:1.2px;color:var(--muted);
  text-transform:uppercase;margin-bottom:10px;
  display:flex;align-items:center;justify-content:space-between;
}
.rules-tbl{width:100%;border-collapse:collapse;font-size:12px}
.rules-tbl th{
  padding:9px 14px;text-align:left;font-size:10px;font-weight:600;
  color:var(--muted);text-transform:uppercase;letter-spacing:.5px;
  background:var(--card2);border-bottom:1px solid var(--border2);
  cursor:pointer;white-space:nowrap;user-select:none;
}
.rules-tbl th:hover{color:var(--text)}
.rules-tbl th.sorted{color:var(--orange)}
.sort-arrow{margin-left:3px;font-size:9px;opacity:.5}
.rules-tbl th.sorted .sort-arrow{opacity:1}
.rules-tbl td{padding:9px 14px;border-bottom:1px solid var(--border)}
.rules-tbl tr:hover td{background:rgba(255,255,255,.02)}
.rules-tbl tr:last-child td{border-bottom:none}
.rn{color:var(--muted);font-size:11px;font-weight:600;width:32px}
.toks-cell{display:flex;flex-wrap:wrap;gap:2px}
.conf-val{font-weight:600}
.lift-high{color:var(--green);font-weight:700}
.lift-med{color:#eab308;font-weight:600}
.lift-low{color:var(--muted2)}
.sup-inline{display:flex;align-items:center;gap:6px}
.sup-bar-sm{width:60px;height:3px;background:var(--card2);border-radius:2px;display:inline-block}
.sup-bar-fill{height:100%;background:var(--blue);border-radius:2px;display:block}
.num{text-align:right;font-variant-numeric:tabular-nums}
.pagination{display:flex;gap:5px;justify-content:center;margin-top:14px;flex-wrap:wrap}
.pg-btn{
  background:var(--card);border:1px solid var(--border2);border-radius:5px;
  padding:4px 10px;font-size:11px;color:var(--muted2);cursor:pointer;
  transition:all .15s;font-family:inherit;
}
.pg-btn:hover{color:var(--text);border-color:var(--border2)}
.pg-btn.active{background:var(--orange);border-color:var(--orange);color:#fff}

/* ── Suggester ────────────────────────────────────────────── */
.sug-layout{display:grid;grid-template-columns:260px 1fr;gap:18px;align-items:start}
@media(max-width:800px){.sug-layout{grid-template-columns:1fr}}
.sug-panel{
  background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
  padding:18px 18px;position:sticky;top:0;
}
.sug-panel-title{font-size:10px;font-weight:700;letter-spacing:1.5px;
  color:var(--muted);text-transform:uppercase;margin-bottom:16px}
.filter-group{margin-bottom:16px}
.filter-label{font-size:11px;font-weight:500;color:var(--muted2);margin-bottom:6px}
.filter-select{
  width:100%;background:var(--card2);border:1px solid var(--border2);border-radius:6px;
  padding:8px 10px;font-size:12px;color:var(--text);font-family:inherit;
}
.filter-select:focus{outline:none;border-color:rgba(249,115,22,.4)}
.slider-row{display:flex;justify-content:space-between;font-size:10px;
  color:var(--muted);margin-top:3px}
input[type=range]{
  width:100%;accent-color:var(--orange);cursor:pointer;
}
.sug-btn{
  width:100%;background:var(--orange);border:none;border-radius:6px;
  padding:10px;font-size:12px;font-weight:700;color:#fff;letter-spacing:.5px;
  cursor:pointer;transition:background .15s;font-family:inherit;margin-top:4px;
}
.sug-btn:hover{background:var(--orange2)}
.sug-results{display:flex;flex-direction:column;gap:10px}
.sug-card{
  background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
  padding:16px 18px;transition:border-color .15s;
}
.sug-card:hover{border-color:var(--border2)}
.sug-card-head{display:flex;align-items:center;gap:10px;margin-bottom:10px}
.sug-rank{
  font-size:10px;font-weight:700;color:var(--muted);letter-spacing:.5px;
}
.reliability{
  font-size:9px;font-weight:800;padding:2px 7px;border-radius:3px;
  letter-spacing:.8px;text-transform:uppercase;
}
.rel-elite{background:rgba(34,197,94,.12);color:#4ade80;border:1px solid rgba(34,197,94,.2)}
.rel-reliable{background:rgba(249,115,22,.12);color:var(--orange);border:1px solid rgba(249,115,22,.2)}
.rel-moderate{background:rgba(59,130,246,.1);color:#60a5fa;border:1px solid rgba(59,130,246,.2)}
.rel-developing{background:rgba(107,128,160,.1);color:var(--muted2);border:1px solid var(--border2)}
.sug-moves{display:flex;align-items:center;flex-wrap:wrap;gap:4px;margin-bottom:10px}
.sug-arrow{color:var(--muted);font-size:14px;font-weight:300}
.sug-outcome{
  font-size:11px;font-weight:700;padding:3px 10px;border-radius:4px;
}
.sug-stats{display:flex;gap:16px;font-size:11px;color:var(--muted2)}
.sug-stat strong{color:var(--text);font-weight:600}
.sug-stat .stat-label{margin-right:4px}
.sug-empty{text-align:center;padding:40px;color:var(--muted);font-size:13px}

/* ── Benchmark ────────────────────────────────────────────── */
.bm-stat-row{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:22px}
.bm-stat{
  background:var(--card);border:1px solid var(--border);border-radius:var(--radius);
  padding:16px 20px;
}
.bm-stat-label{font-size:10px;font-weight:600;color:var(--muted);
  letter-spacing:.8px;text-transform:uppercase;margin-bottom:6px}
.bm-stat-value{font-size:26px;font-weight:800;color:var(--orange);line-height:1;letter-spacing:-0.5px}
.bm-stat-unit{font-size:12px;font-weight:400;color:var(--muted2);margin-top:4px}
.sweep-row{
  display:flex;align-items:center;gap:10px;margin-bottom:10px;
}
.sweep-label{width:36px;font-size:11px;font-weight:600;color:var(--orange);text-align:right}
.sweep-bars{flex:1;display:flex;flex-direction:column;gap:3px}
.sweep-bar-wrap{display:flex;align-items:center;gap:6px}
.sweep-bar{height:9px;border-radius:3px;min-width:4px;transition:width .6s}
.sweep-bar.rules{background:var(--orange)}
.sweep-bar.time{background:var(--blue)}
.sweep-val{font-size:10px;color:var(--muted2);min-width:36px}
.sweep-itemsets{font-size:10px;color:var(--muted);margin-left:4px}
.bm-legend{display:flex;gap:14px;margin-bottom:12px;font-size:11px;color:var(--muted2)}
.bm-leg-dot{width:10px;height:10px;border-radius:2px;display:inline-block;margin-right:5px;vertical-align:middle}

/* ── Tooltip ──────────────────────────────────────────────── */
#tip{
  position:fixed;display:none;background:#1a2640;border:1px solid var(--border2);
  border-radius:7px;padding:8px 12px;font-size:11px;line-height:1.7;
  pointer-events:none;z-index:9999;max-width:260px;
  box-shadow:0 8px 24px rgba(0,0,0,.6);
}
</style>
</head>
<body>
<div id="tip"></div>

<!-- ── Sidebar ─────────────────────────────────────────────── -->
<aside id="sidebar">
  <div class="sb-logo">
    <div class="sb-logo-name">HOOP<span>IQ</span></div>
    <div class="sb-logo-sub">PLAY ANALYTICS ENGINE</div>
  </div>
  <nav>
    <a class="nav-item active" data-page="overview">
      <span class="nav-icon">⊞</span> Overview
    </a>
    <a class="nav-item" data-page="rules">
      <span class="nav-icon">≡</span> Pattern Rules
    </a>
    <a class="nav-item" data-page="suggester">
      <span class="nav-icon">◎</span> Play Suggester
    </a>
    <a class="nav-item" data-page="benchmark">
      <span class="nav-icon">↗</span> Benchmark
    </a>
    <a class="nav-item" data-page="load">
      <span class="nav-icon">↓</span> Load Data
    </a>
  </nav>
  <div class="sb-footer">
    <span class="engine-dot"></span>Engine ready<br>
    <span style="opacity:.5">v1.0 — FP-Growth</span>
  </div>
</aside>

<!-- ── Main ────────────────────────────────────────────────── -->
<div id="main">
  <div id="topbar">
    <span class="page-title" id="topbar-title">OVERVIEW</span>
    <div class="topbar-meta">
      <span class="meta-pill highlight" id="meta-team">—</span>
      <span class="meta-pill" id="meta-season">—</span>
      <span class="meta-pill" id="meta-games">— Games</span>
      <span class="meta-pill" id="meta-rules">— Rules</span>
    </div>
  </div>

  <div id="content">

    <!-- ── OVERVIEW ─────────────────────────────────────────── -->
    <div id="page-overview" class="page active">
      <div class="stat-row" id="ov-stat-row"></div>
      <div class="two-col">
        <div class="card">
          <div class="card-header">
            <span class="card-title">Scoring Event Breakdown</span>
            <span class="card-badge" id="ov-score-badge"></span>
          </div>
          <svg id="donut-svg" width="180" height="180"></svg>
          <div class="donut-legend" id="donut-legend"></div>
        </div>
        <div class="card">
          <div class="card-header">
            <span class="card-title">Top Action Frequency</span>
          </div>
          <div id="freq-bars"></div>
        </div>
      </div>
      <div class="card">
        <div class="card-header">
          <span class="card-title">Top 10 Raw Sequences</span>
          <span class="card-badge" id="ov-seq-sort">by frequency</span>
        </div>
        <table class="seq-table">
          <colgroup>
            <col style="width:32px">
            <col>
            <col style="width:80px">
            <col style="width:120px">
            <col style="width:70px">
          </colgroup>
          <thead>
            <tr>
              <th>#</th>
              <th>Sequence</th>
              <th>Count</th>
              <th>Support</th>
              <th>Scores?</th>
            </tr>
          </thead>
          <tbody id="seq-tbody"></tbody>
        </table>
      </div>
    </div>

    <!-- ── PATTERN RULES ────────────────────────────────────── -->
    <div id="page-rules" class="page">
      <div class="rules-toolbar">
        <select class="tb-select" id="rules-sort">
          <option value="lift">Sort: Lift ↓</option>
          <option value="confidence">Sort: Confidence ↓</option>
          <option value="support">Sort: Support ↓</option>
          <option value="n_antecedents">Sort: Items ↓</option>
        </select>
        <select class="tb-select" id="rules-outcome">
          <option value="all">All scoring types</option>
          <option value="score">Score only</option>
          <option value="miss">Miss only</option>
          <option value="turnover">Turnover only</option>
          <option value="foul_score">Foul (Made)</option>
          <option value="foul_miss">Foul (Miss)</option>
        </select>
        <input class="tb-search" id="rules-search" placeholder="Search antecedents…" type="text">
        <span class="tb-count" id="rules-count"></span>
      </div>
      <div class="rules-section-label">
        Scoring Association Rules
        <span class="card-badge">Lift-ranked</span>
      </div>
      <table class="rules-tbl">
        <thead>
          <tr>
            <th class="rn">#</th>
            <th data-col="antecedent_str" data-type="str">Antecedent(s) <span class="sort-arrow">↕</span></th>
            <th>→ Outcome</th>
            <th data-col="support" data-type="num" class="num">Support <span class="sort-arrow">↕</span></th>
            <th data-col="confidence" data-type="num" class="num">Confidence <span class="sort-arrow">↕</span></th>
            <th data-col="lift" data-type="num" class="num sorted">Lift <span class="sort-arrow">↓</span></th>
          </tr>
        </thead>
        <tbody id="rules-tbody"></tbody>
      </table>
      <div class="pagination" id="rules-pagination"></div>
    </div>

    <!-- ── PLAY SUGGESTER ───────────────────────────────────── -->
    <div id="page-suggester" class="page">
      <div class="sug-layout">
        <div class="sug-panel">
          <div class="sug-panel-title">Situation Filters</div>
          <div class="filter-group">
            <div class="filter-label">Play Style</div>
            <select class="filter-select" id="sug-style">
              <option value="all">All Plays</option>
              <option value="3pt">3-Point Focus</option>
              <option value="inside">Inside Game</option>
              <option value="midrange">Mid-Range</option>
              <option value="pr">Pick & Roll</option>
              <option value="transition">Transition</option>
            </select>
          </div>
          <div class="filter-group">
            <div class="filter-label">Minimum Confidence: <span id="sug-conf-val">30%</span></div>
            <input type="range" id="sug-conf" min="10" max="90" value="30" step="5">
            <div class="slider-row"><span>10%</span><span>90%</span></div>
          </div>
          <div class="filter-group">
            <div class="filter-label">Minimum Lift: <span id="sug-lift-val">1.0×</span></div>
            <input type="range" id="sug-lift" min="10" max="30" value="10" step="1">
            <div class="slider-row"><span>1.0×</span><span>3.0×</span></div>
          </div>
          <div class="filter-group">
            <div class="filter-label">Sequence Length</div>
            <select class="filter-select" id="sug-len">
              <option value="0">Any length</option>
              <option value="1">1 move</option>
              <option value="2">2 moves</option>
              <option value="3">3+ moves</option>
            </select>
          </div>
          <button class="sug-btn" id="sug-go">SUGGEST PLAYS</button>
        </div>
        <div class="sug-results" id="sug-results">
          <div class="sug-empty">Set filters and click <strong>Suggest Plays</strong></div>
        </div>
      </div>
    </div>

    <!-- ── BENCHMARK ────────────────────────────────────────── -->
    <div id="page-benchmark" class="page">
      <div class="bm-stat-row" id="bm-stats"></div>
      <div class="card" style="margin-bottom:16px">
        <div class="card-header">
          <span class="card-title">Support Threshold Sweep</span>
          <span class="card-badge">FP-Growth performance</span>
        </div>
        <div class="bm-legend">
          <span><span class="bm-leg-dot" style="background:var(--orange)"></span>Rules found</span>
          <span><span class="bm-leg-dot" style="background:var(--blue)"></span>Runtime (s)</span>
        </div>
        <div id="bm-sweep"></div>
      </div>
      <div class="two-col">
        <div class="card">
          <div class="card-header"><span class="card-title">Rules vs Support</span></div>
          <svg id="bm-bar-svg" style="width:100%;overflow:visible"></svg>
        </div>
        <div class="card">
          <div class="card-header"><span class="card-title">Runtime vs Support</span></div>
          <svg id="bm-line-svg" style="width:100%;overflow:visible"></svg>
        </div>
      </div>
      <div id="bm-empty" style="display:none" class="sug-empty">
        Benchmark was skipped. Re-run without <code>--no-benchmark</code>.
      </div>
    </div>

    <!-- ── LOAD DATA ─────────────────────────────────────────── -->
    <div id="page-load" class="page">
      <div class="card" style="max-width:600px">
        <div class="card-header"><span class="card-title">Pipeline Commands</span></div>
        <p style="color:var(--muted2);margin-bottom:16px;font-size:12px;line-height:1.7">
          This dashboard is generated from <code style="color:var(--orange)">data/rules.csv</code>.
          Run the full pipeline to refresh data:
        </p>
        <pre style="background:var(--card2);border:1px solid var(--border2);border-radius:6px;
          padding:14px 16px;font-size:11px;color:#a5f3fc;line-height:2;overflow-x:auto"
        ># Pull data for a specific team + season
python 1_data_pull.py --team GSW --season 2023-24 --max_games 0

# Preprocess — keep only that team's possessions
python 2_preprocessing.py --team GSW

# Mine association rules
python 3_arm_mining.py --min_support 0.01 --min_confidence 0.35

# Regenerate this dashboard
python 5_dashboard.py --team GSW --season 2023-24</pre>
      </div>
    </div>

  </div><!-- /content -->
</div><!-- /main -->

<script src="https://d3js.org/d3.v7.min.js"></script>
<script>
// ── Injected data ─────────────────────────────────────────────────────────────
const RULES         = __RULES_JSON__;
const STATS         = __STATS_JSON__;
const BENCHMARK     = __BENCHMARK_JSON__;
const META          = __META_JSON__;
const TOKEN_LABELS  = __TOKEN_LABELS_JSON__;
const TOKEN_COLORS  = __TOKEN_COLORS_JSON__;

const OUTCOME_META = {
  score:      {label:"Score",      color:"#22c55e"},
  miss:       {label:"Miss",       color:"#ef4444"},
  turnover:   {label:"Turnover",   color:"#f97316"},
  foul_score: {label:"Foul (Made)",color:"#3b82f6"},
  foul_miss:  {label:"Foul (Miss)",color:"#a855f7"},
};

const PLAY_STYLES = {
  all:        {label:"All Plays",      tokens:null},
  "3pt":      {label:"3-Point Focus",  tokens:["three_pointer","step_back","pull_up"]},
  inside:     {label:"Inside Game",    tokens:["dunk","cut","layup","drive","alley_oop","putback"]},
  midrange:   {label:"Mid-Range",      tokens:["fadeaway","hook_shot","floater","jump_shot","bank_shot"]},
  pr:         {label:"Pick & Roll",    tokens:["pick_and_roll"]},
  transition: {label:"Transition",     tokens:["transition"]},
};

// ── Utilities ─────────────────────────────────────────────────────────────────
const tip  = document.getElementById("tip");
const showTip = (ev, html) => {
  tip.style.display = "block"; tip.innerHTML = html;
  const x = Math.min(ev.clientX + 12, window.innerWidth - 270);
  tip.style.left = x + "px"; tip.style.top = (ev.clientY - 38) + "px";
};
const moveTip = ev => {
  const x = Math.min(ev.clientX + 12, window.innerWidth - 270);
  tip.style.left = x + "px"; tip.style.top = (ev.clientY - 38) + "px";
};
const hideTip = () => tip.style.display = "none";

function tokLabel(t)  { return TOKEN_LABELS[t] || t.replace(/_/g," "); }
function tokColor(t)  { return TOKEN_COLORS[t] || "#6b7280"; }

function tokBadge(name, indicator) {
  const c = tokColor(name);
  const lbl = tokLabel(name);
  const ind = indicator === "score" ? `<span class="tok-ind" style="color:#4ade80">✓</span>`
            : indicator === "miss"  ? `<span class="tok-ind" style="color:#f87171">×</span>`
            : "";
  return `<span class="tok" style="background:${c}18;border:1px solid ${c}35;color:${c}">
    <span class="tok-dot" style="background:${c}"></span>${lbl}${ind}
  </span>`;
}

function outcomeTag(o) {
  const m = OUTCOME_META[o] || {label:o, color:"#6b7280"};
  return `<span class="tok" style="background:${m.color}18;border:1px solid ${m.color}35;color:${m.color};font-weight:600">
    ${m.label}
  </span>`;
}

function liftClass(v) {
  return v >= 1.8 ? "lift-high" : v >= 1.3 ? "lift-med" : "lift-low";
}

function pct(v, d=1) { return (v*100).toFixed(d)+"%"; }
function fmt(v, d=3)  { return Number(v).toFixed(d); }

// ── Navigation ────────────────────────────────────────────────────────────────
const PAGE_TITLES = {
  overview:"OVERVIEW", rules:"PATTERN RULES",
  suggester:"PLAY SUGGESTER", benchmark:"BENCHMARK", load:"LOAD DATA"
};
const pageInited = {};

document.querySelectorAll(".nav-item[data-page]").forEach(el => {
  el.addEventListener("click", () => {
    const pid = el.dataset.page;
    document.querySelectorAll(".nav-item").forEach(n => n.classList.remove("active"));
    document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
    el.classList.add("active");
    document.getElementById(`page-${pid}`).classList.add("active");
    document.getElementById("topbar-title").textContent = PAGE_TITLES[pid] || pid.toUpperCase();
    if (!pageInited[pid]) { pageInited[pid] = true; initPage(pid); }
  });
});

function initPage(pid) {
  if (pid === "rules")     initRules();
  if (pid === "benchmark") initBenchmark();
}

// ── Init header meta ──────────────────────────────────────────────────────────
function initMeta() {
  document.getElementById("meta-team").textContent    = META.team;
  document.getElementById("meta-season").textContent  = META.season;
  document.getElementById("meta-games").textContent   = `${META.game_count} Games`;
  document.getElementById("meta-rules").textContent   = `${META.rules_count} Rules`;
}

// ── OVERVIEW ──────────────────────────────────────────────────────────────────
function initOverview() {
  // Stat cards
  const oc = STATS.outcome_counts;
  const cards = [
    {label:"GAMES LOADED",    value:META.game_count,    sub:"regular season",       cls:""},
    {label:"TOTAL SEQUENCES", value:STATS.total_possessions.toLocaleString(),
                                                         sub:"possessions",          cls:""},
    {label:"SCORING RATE",    value:STATS.scoring_rate+"%",
                                                         sub:"sequences ending in score", cls:"green"},
    {label:"RULES MINED",     value:META.rules_count,   sub:"association rules",    cls:"orange"},
  ];
  document.getElementById("ov-stat-row").innerHTML = cards.map(c => `
    <div class="stat-card">
      <div class="stat-card-label">${c.label}</div>
      <div class="stat-card-value ${c.cls}">${c.value}</div>
      <div class="stat-card-sub">${c.sub}</div>
    </div>`).join("");

  // Donut
  const scoreCount = oc["score"] || 0;
  document.getElementById("ov-score-badge").textContent = `${scoreCount.toLocaleString()} scores`;
  drawDonut();

  // Frequency bars
  drawFreqBars();

  // Sequences table
  drawSeqTable();
}

function drawDonut() {
  const bd = STATS.scoring_breakdown;
  const data = Object.entries(bd)
    .sort((a,b) => b[1]-a[1])
    .slice(0, 7)
    .map(([k,v]) => ({token:k, count:v, color:tokColor(k)}));
  const total = data.reduce((s,d)=>s+d.count,0);

  const W = 180, H = 180, R = 70, r = 42;
  const svg = d3.select("#donut-svg").attr("width",W).attr("height",H);
  svg.selectAll("*").remove();
  const g = svg.append("g").attr("transform",`translate(${W/2},${H/2})`);

  const pie  = d3.pie().value(d=>d.count).sort(null);
  const arc  = d3.arc().innerRadius(r).outerRadius(R).cornerRadius(2).padAngle(.02);
  const arc2 = d3.arc().innerRadius(r).outerRadius(R+6).cornerRadius(2);

  const slices = g.selectAll("path").data(pie(data)).join("path")
    .attr("d", arc)
    .attr("fill", d=>d.data.color)
    .attr("opacity", .85)
    .style("cursor","pointer")
    .on("mouseover", function(ev,d) {
      d3.select(this).transition().duration(120).attr("d", arc2);
      showTip(ev, `<strong style="color:${d.data.color}">${tokLabel(d.data.token)}</strong><br>
        ${d.data.count.toLocaleString()} scoring possessions (${Math.round(d.data.count/total*100)}%)`);
    })
    .on("mousemove", moveTip)
    .on("mouseout", function(ev,d) {
      d3.select(this).transition().duration(120).attr("d", arc);
      hideTip();
    });

  g.append("text").attr("text-anchor","middle").attr("dy","-0.1em")
    .attr("fill","#e2e8f0").attr("font-size","18px").attr("font-weight","700")
    .text(total.toLocaleString());
  g.append("text").attr("text-anchor","middle").attr("dy","1.2em")
    .attr("fill","#4b6080").attr("font-size","9px").text("scoring plays");

  const legend = document.getElementById("donut-legend");
  legend.innerHTML = data.map(d => `
    <div class="legend-item">
      <span class="legend-dot" style="background:${d.color}"></span>
      ${tokLabel(d.token)} <span style="color:#4b6080;margin-left:3px">${Math.round(d.count/total*100)}%</span>
    </div>`).join("");
}

function drawFreqBars() {
  const freq = STATS.token_freq;
  const sr   = STATS.token_score_rate;
  const entries = Object.entries(freq).slice(0, 14);
  const max  = entries[0]?.[1] || 1;
  const container = document.getElementById("freq-bars");
  container.innerHTML = entries.map(([t,v]) => {
    const c   = tokColor(t);
    const ind = (sr[t]||0) > 0.55 ? "✓" : (sr[t]||0) < 0.40 ? "×" : "";
    const indC= ind === "✓" ? "#4ade80" : "#f87171";
    return `<div class="hbar-row">
      <div class="hbar-label">
        <span style="color:${c}">${tokLabel(t)}</span>
        ${ind ? `<span style="color:${indC};font-size:10px;margin-left:3px">${ind}</span>` : ""}
      </div>
      <div class="hbar-track">
        <div class="hbar-fill" style="width:${Math.round(v/max*100)}%;background:${c}88"></div>
      </div>
      <div class="hbar-count" style="color:${c}99">${v.toLocaleString()}</div>
    </div>`;
  }).join("");
}

function drawSeqTable() {
  const tbody = document.getElementById("seq-tbody");
  const seqs  = STATS.top_sequences.slice(0,10);
  const maxCnt = seqs[0]?.count || 1;
  tbody.innerHTML = seqs.map((s,i) => {
    const toks = s.moves.map(t => {
      const sr = STATS.token_score_rate[t] || 0;
      const ind = sr > 0.55 ? "score" : sr < 0.40 ? "miss" : null;
      return tokBadge(t, ind);
    }).join("");
    const supW = Math.round(s.support / (seqs[0]?.support || 1) * 100);
    return `<tr>
      <td class="seq-num">${i+1}</td>
      <td><div class="seq-tokens">${toks}</div></td>
      <td style="color:#e2e8f0;font-weight:600;white-space:nowrap">${s.count.toLocaleString()}</td>
      <td>
        <div class="sup-bar">
          <div class="sup-track"><div class="sup-fill" style="width:${supW}%"></div></div>
          <span class="sup-text">${(s.support*100).toFixed(1)}%</span>
        </div>
      </td>
      <td><span class="scores-pill ${s.scores?"scores-yes":"scores-no"}">${s.scores?"YES":"NO"}</span></td>
    </tr>`;
  }).join("");
}

// ── PATTERN RULES ─────────────────────────────────────────────────────────────
let rSortCol = "lift", rSortDir = -1, rPage = 0;
const R_PAGE = 20;

function getRulesFiltered() {
  const outcome = document.getElementById("rules-outcome").value;
  const q       = document.getElementById("rules-search").value.trim().toLowerCase();
  let data = RULES;
  if (outcome !== "all") data = data.filter(r => r.consequent_str === outcome);
  if (q) data = data.filter(r => r.antecedent_str.toLowerCase().includes(q));
  return [...data].sort((a,b) => {
    const av=a[rSortCol], bv=b[rSortCol];
    if (typeof av==="string") return rSortDir*av.localeCompare(bv);
    return rSortDir*(av-bv);
  });
}

function renderRules() {
  const data  = getRulesFiltered();
  const pages = Math.ceil(data.length / R_PAGE);
  rPage = Math.max(0, Math.min(rPage, pages-1));
  const slice = data.slice(rPage*R_PAGE, (rPage+1)*R_PAGE);
  const maxSup = Math.max(...RULES.map(r=>r.support));

  document.getElementById("rules-count").textContent =
    `${data.length} rule${data.length!==1?"s":""}`;

  const sr = STATS.token_score_rate;
  const tbody = document.getElementById("rules-tbody");
  tbody.innerHTML = slice.map((r,i) => {
    const ants = r.antecedent_str.split(" + ").map(t => {
      const rate = sr[t] || 0;
      const ind  = rate > 0.55 ? "score" : rate < 0.40 ? "miss" : null;
      return tokBadge(t, ind);
    }).join("");
    const supW = Math.round(r.support/maxSup*100);
    const lc   = liftClass(r.lift);
    return `<tr>
      <td class="rn">${rPage*R_PAGE+i+1}</td>
      <td><div class="toks-cell">${ants}</div></td>
      <td>${outcomeTag(r.consequent_str)}</td>
      <td class="num">
        <div class="sup-inline">
          <span class="sup-bar-sm"><span class="sup-bar-fill" style="width:${supW}%"></span></span>
          <span style="color:#6b84a0;font-size:11px">${(r.support*100).toFixed(1)}%</span>
        </div>
      </td>
      <td class="num"><span class="conf-val">${Math.round(r.confidence*100)}%</span></td>
      <td class="num"><span class="${lc}">${fmt(r.lift,2)}×</span></td>
    </tr>`;
  }).join("");

  const pag = document.getElementById("rules-pagination");
  pag.innerHTML = "";
  if (pages > 1) {
    for (let i=0; i<pages; i++) {
      const b = document.createElement("button");
      b.className = "pg-btn" + (i===rPage?" active":"");
      b.textContent = i+1;
      b.onclick = () => { rPage=i; renderRules(); };
      pag.appendChild(b);
    }
  }
}

function initRules() {
  document.getElementById("rules-sort").addEventListener("change", e => {
    rSortCol = e.target.value; rPage=0; renderRules();
  });
  document.getElementById("rules-outcome").addEventListener("change", () => { rPage=0; renderRules(); });
  document.getElementById("rules-search").addEventListener("input",  () => { rPage=0; renderRules(); });

  document.querySelectorAll(".rules-tbl th[data-col]").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;
      if (rSortCol===col) rSortDir*=-1;
      else { rSortCol=col; rSortDir=th.dataset.type==="str"?1:-1; }
      document.querySelectorAll(".rules-tbl th[data-col]").forEach(h => {
        h.classList.remove("sorted");
        h.querySelector(".sort-arrow").textContent="↕";
      });
      th.classList.add("sorted");
      th.querySelector(".sort-arrow").textContent = rSortDir===-1?"↓":"↑";
      rPage=0; renderRules();
    });
  });
  renderRules();
}

// ── PLAY SUGGESTER ────────────────────────────────────────────────────────────
function getReliability(lift) {
  if (lift >= 2.0) return {label:"ELITE",      cls:"rel-elite"};
  if (lift >= 1.6) return {label:"RELIABLE",   cls:"rel-reliable"};
  if (lift >= 1.3) return {label:"MODERATE",   cls:"rel-moderate"};
  return               {label:"DEVELOPING",  cls:"rel-developing"};
}

document.getElementById("sug-conf").addEventListener("input", e => {
  document.getElementById("sug-conf-val").textContent = e.target.value + "%";
});
document.getElementById("sug-lift").addEventListener("input", e => {
  document.getElementById("sug-lift-val").textContent = (e.target.value/10).toFixed(1) + "×";
});
document.getElementById("sug-go").addEventListener("click", runSuggester);

function runSuggester() {
  const style   = document.getElementById("sug-style").value;
  const minConf = parseInt(document.getElementById("sug-conf").value) / 100;
  const minLift = parseInt(document.getElementById("sug-lift").value)  / 10;
  const lenVal  = parseInt(document.getElementById("sug-len").value);

  const styleToks = PLAY_STYLES[style]?.tokens || null;

  let filtered = RULES.filter(r =>
    r.confidence >= minConf &&
    r.lift       >= minLift &&
    (lenVal === 0 ||
     lenVal === 3 ? r.n_antecedents >= 3 : r.n_antecedents === lenVal)
  );

  if (styleToks) {
    filtered = filtered.filter(r =>
      r.antecedent_str.split(" + ").some(t => styleToks.includes(t))
    );
  }

  filtered.sort((a,b) => b.lift - a.lift);

  const container = document.getElementById("sug-results");
  if (!filtered.length) {
    container.innerHTML = `<div class="sug-empty">No plays match these filters. Try loosening confidence or lift.</div>`;
    return;
  }

  const sr = STATS.token_score_rate;
  container.innerHTML = filtered.slice(0,20).map((r,i) => {
    const rel  = getReliability(r.lift);
    const ants = r.antecedent_str.split(" + ");
    const moves = ants.map((t,j) => {
      const rate = sr[t] || 0;
      const ind  = rate > 0.55 ? "score" : rate < 0.40 ? "miss" : null;
      return tokBadge(t, ind) + (j < ants.length-1 ? `<span class="sug-arrow">→</span>` : "");
    }).join("");
    const om = OUTCOME_META[r.consequent_str] || {label:r.consequent_str,color:"#6b7280"};
    return `<div class="sug-card">
      <div class="sug-card-head">
        <span class="sug-rank">#${i+1}</span>
        <span class="reliability ${rel.cls}">${rel.label}</span>
      </div>
      <div class="sug-moves">
        ${moves}
        <span class="sug-arrow" style="margin:0 4px">→</span>
        <span class="sug-outcome" style="background:${om.color}18;color:${om.color};border:1px solid ${om.color}35">
          ${om.label.toUpperCase()}
        </span>
      </div>
      <div class="sug-stats">
        <span class="sug-stat"><span class="stat-label">Confidence</span>
          <strong>${Math.round(r.confidence*100)}%</strong></span>
        <span class="sug-stat"><span class="stat-label">Lift</span>
          <strong>${fmt(r.lift,2)}×</strong></span>
        <span class="sug-stat"><span class="stat-label">Support</span>
          <strong>${(r.support*100).toFixed(1)}%</strong></span>
      </div>
    </div>`;
  }).join("");
}

// ── BENCHMARK ─────────────────────────────────────────────────────────────────
function initBenchmark() {
  if (!BENCHMARK.length) {
    document.getElementById("bm-empty").style.display = "block";
    return;
  }

  // Stat cards
  document.getElementById("bm-stats").innerHTML = [
    {label:"FASTEST RUN",     value:META.bm_fastest,   unit:"seconds"},
    {label:"AVG RUNTIME",     value:META.bm_avg,       unit:"across thresholds"},
    {label:"MAX RULES FOUND", value:META.bm_max_rules, unit:"at lowest support"},
  ].map(s => `
    <div class="bm-stat">
      <div class="bm-stat-label">${s.label}</div>
      <div class="bm-stat-value">${s.value}</div>
      <div class="bm-stat-unit">${s.unit}</div>
    </div>`).join("");

  // Sweep bars
  const maxRules = Math.max(...BENCHMARK.map(b=>b.rules), 1);
  const maxTime  = Math.max(...BENCHMARK.map(b=>b.time),  0.001);
  const BAR_MAX  = 400;

  document.getElementById("bm-sweep").innerHTML = BENCHMARK.map(b => `
    <div class="sweep-row">
      <div class="sweep-label">${b.support_pct}</div>
      <div class="sweep-bars">
        <div class="sweep-bar-wrap">
          <div class="sweep-bar rules" style="width:${Math.round(b.rules/maxRules*BAR_MAX)}px"></div>
          <span class="sweep-val" style="color:var(--orange)">${b.rules}</span>
          <span style="color:var(--muted);font-size:10px">rules</span>
        </div>
        <div class="sweep-bar-wrap">
          <div class="sweep-bar time" style="width:${Math.round(b.time/maxTime*BAR_MAX)}px"></div>
          <span class="sweep-val" style="color:var(--blue)">${b.time}s</span>
        </div>
      </div>
      <span class="sweep-itemsets">${b.itemsets} itemsets</span>
    </div>`).join("");

  // Rules vs Support bar chart (D3)
  drawBmBar();

  // Runtime vs Support line chart (D3)
  drawBmLine();
}

function drawBmBar() {
  const el = document.getElementById("bm-bar-svg");
  const W  = el.clientWidth || 300;
  const m  = {top:10,right:20,bottom:30,left:36};
  const H  = 160;
  const svg = d3.select(el).attr("height", H + m.top + m.bottom);
  svg.selectAll("*").remove();
  const g = svg.append("g").attr("transform",`translate(${m.left},${m.top})`);
  const iW = W - m.left - m.right;

  const x = d3.scaleBand().domain(BENCHMARK.map(b=>b.support_pct)).range([0,iW]).padding(.25);
  const y = d3.scaleLinear().domain([0, d3.max(BENCHMARK,b=>b.rules)*1.15]).range([H,0]);

  g.selectAll("rect").data(BENCHMARK).join("rect")
    .attr("x", b=>x(b.support_pct)).attr("width", x.bandwidth())
    .attr("y", b=>y(b.rules)).attr("height", b=>H-y(b.rules))
    .attr("fill","var(--orange)").attr("opacity",.85).attr("rx",3)
    .on("mouseover",(ev,b)=>showTip(ev,
      `<strong>${b.support_pct}</strong><br>${b.rules} rules · ${b.time}s`))
    .on("mousemove",moveTip).on("mouseout",hideTip);

  g.append("g").attr("transform",`translate(0,${H})`)
    .call(d3.axisBottom(x)).selectAll("text")
    .attr("fill","#4b6080").attr("font-size","10px");
  g.append("g").call(d3.axisLeft(y).ticks(4))
    .selectAll("text").attr("fill","#4b6080").attr("font-size","10px");
  g.selectAll(".domain,.tick line").attr("stroke","#1e2d42");
}

function drawBmLine() {
  const el = document.getElementById("bm-line-svg");
  const W  = el.clientWidth || 300;
  const m  = {top:10,right:20,bottom:30,left:52};
  const H  = 160;
  const svg = d3.select(el).attr("height", H + m.top + m.bottom);
  svg.selectAll("*").remove();
  const g = svg.append("g").attr("transform",`translate(${m.left},${m.top})`);
  const iW = W - m.left - m.right;

  const x = d3.scalePoint().domain(BENCHMARK.map(b=>b.support_pct)).range([0,iW]).padding(.3);
  const y = d3.scaleLinear().domain([0, d3.max(BENCHMARK,b=>b.time)*1.2]).range([H,0]);

  const area = d3.area().x(b=>x(b.support_pct)).y0(H).y1(b=>y(b.time)).curve(d3.curveMonotoneX);
  const line = d3.line().x(b=>x(b.support_pct)).y(b=>y(b.time)).curve(d3.curveMonotoneX);

  g.append("path").datum(BENCHMARK).attr("d",area)
    .attr("fill","rgba(59,130,246,.12)");
  g.append("path").datum(BENCHMARK).attr("d",line)
    .attr("fill","none").attr("stroke","#3b82f6").attr("stroke-width",2);
  g.selectAll("circle").data(BENCHMARK).join("circle")
    .attr("cx",b=>x(b.support_pct)).attr("cy",b=>y(b.time))
    .attr("r",4).attr("fill","#3b82f6").attr("stroke","var(--bg)").attr("stroke-width",2)
    .on("mouseover",(ev,b)=>showTip(ev,
      `<strong>${b.support_pct}</strong><br>${b.time}s runtime · ${b.itemsets} itemsets`))
    .on("mousemove",moveTip).on("mouseout",hideTip);

  g.append("g").attr("transform",`translate(0,${H})`)
    .call(d3.axisBottom(x)).selectAll("text")
    .attr("fill","#4b6080").attr("font-size","10px");
  g.append("g").call(d3.axisLeft(y).ticks(4).tickFormat(d=>d+"s"))
    .selectAll("text").attr("fill","#4b6080").attr("font-size","10px");
  g.selectAll(".domain,.tick line").attr("stroke","#1e2d42");
}

// ── Boot ──────────────────────────────────────────────────────────────────────
function init() {
  initMeta();
  initOverview();
  pageInited["overview"] = true;

  window.addEventListener("resize", () => {
    if (pageInited["overview"]) drawDonut();
    if (pageInited["benchmark"] && BENCHMARK.length) { drawBmBar(); drawBmLine(); }
  });
}
if (document.readyState === "loading")
  document.addEventListener("DOMContentLoaded", init);
else init();
</script>
</body></html>
"""

# ── Entry point ───────────────────────────────────────────────────────────────
def main(rules_path, out_path, no_benchmark, team, season):
    for p, label in [(rules_path,"rules.csv"),(TXN_PATH,"transactions.csv")]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found — run the pipeline first."); return

    print("Loading data…")
    rules_df, transactions = load_data(rules_path, TXN_PATH)
    game_count = len([f for f in os.listdir(RAW_DIR) if f.endswith(".csv")]) \
                 if os.path.isdir(RAW_DIR) else 0
    print(f"  {len(rules_df)} rules · {len(transactions):,} possessions · {game_count} games")

    print("Computing stats…")
    stats = compute_stats(rules_df, transactions)

    benchmark = [] if no_benchmark else run_benchmark(transactions)

    print("Generating HTML…")
    html = build_html(rules_df, stats, benchmark, game_count, team=team, season=season)

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\nDashboard → {out_path}  ({os.path.getsize(out_path)//1024} KB)")
    print(f"  open {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--rules",        default=RULES_PATH)
    p.add_argument("--out",          default=OUT_PATH)
    p.add_argument("--team",         default="ALL")
    p.add_argument("--season",       default="2022-23")
    p.add_argument("--no-benchmark", action="store_true")
    a = p.parse_args()
    main(a.rules, a.out, a.no_benchmark, a.team, a.season)
