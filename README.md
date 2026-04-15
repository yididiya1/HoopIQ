# NBA Play-by-Play Association Rule Mining
### Unsupervised Machine Learning Project

> **Goal:** Discover which basketball movements, dribble patterns, and play creation types associate with scoring, misses, and efficiency outcomes — using Association Rule Mining on NBA tracking data.

---

## What Changed from v1 (Play-by-Play) → v2 (Tracking)

**v1 problem:** Raw play-by-play only logs *events* (shots, turnovers) — not *how* they happened. No dribbles, no movements, no touch time.

**v2 solution:** Three richer NBA tracking data sources:

| Source | What it captures | API Endpoint |
|---|---|---|
| **Synergy Play Types** | HOW the possession was created (isolation, P&R, cut, etc.) | `SynergyPlayTypes` |
| **Shot Dribble Dashboard** | How many dribbles before the shot (0, 1, 2, 3-6, 7+) | `PlayerDashPtShots` |
| **Shot Touch Time Dashboard** | How long ball was held before shot (<2s, 2-6s, 6+s) | `PlayerDashPtShots` |

---

## Transaction Design

Each transaction = one player × situation profile + outcome:

```
{isolation, few_dribbles, long_touch, mid_range}      → miss
{pick_and_roll_bh, no_dribbles, quick_touch, three_pointer} → score
{cut, no_dribbles, quick_touch, at_rim}               → score
{post_up, many_dribbles, long_touch, at_rim}          → score
{spot_up, no_dribbles, quick_touch, three_pointer}    → miss
```

Four transaction types are built and pooled:
- **Synergy** — play type + efficiency outcome
- **Dribble** — dribble count + shot zone + FG outcome
- **Touch** — touch time + shot zone + FG outcome
- **Combined** — all three merged per player (richest)

---

## Token Vocabulary

| Category | Tokens |
|---|---|
| Play creation | `isolation`, `pick_and_roll_bh`, `pick_and_roll_roll`, `post_up`, `spot_up`, `cut`, `handoff`, `off_screen`, `transition`, `off_rebound` |
| Dribbles | `no_dribbles`, `one_dribble`, `two_dribbles`, `few_dribbles`, `many_dribbles` |
| Touch time | `quick_touch`, `medium_touch`, `long_touch` |
| Shot zone | `at_rim`, `short_mid_range`, `mid_range`, `three_pointer` |
| Frequency | `high_freq_play`, `low_freq_play` |
| Outcomes | `score`, `miss`, `turnover_prone`, `high_efficiency`, `low_efficiency` |

---

## Quickstart

```bash
pip install -r requirements.txt
bash run_all.sh 2022-23
```

Or step by step:
```bash
python 1_data_pull.py --season 2022-23
python 2_preprocessing.py --season 2022-23
python 3_arm_mining.py --min_support 0.02 --min_confidence 0.30 --min_lift 1.1
python 4_visualization.py
```

---

## Output Files

| File | Contents |
|---|---|
| `data/rules.csv` | All mined outcome rules |
| `data/rules_combined_only.csv` | Only rich multi-context rules |
| `data/plots/network.png` | Rule network graph |
| `data/plots/lift_heatmap.png` | Token × outcome lift heatmap |
| `data/plots/top_rules_by_outcome.png` | Top rules per outcome |
| `data/plots/play_type_outcome_map.png` | Play type efficiency comparison |

---

## Extensions

- **Sequential mining:** Use PrefixSpan on possession sequences
- **Team segmentation:** Compare rule sets per team
- **Season comparison:** Track how rules shift across seasons
- **Defender distance:** Add `open_shot` / `contested_shot` tokens from tracking
