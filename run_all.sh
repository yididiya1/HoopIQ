#!/bin/bash
# run_all.sh — Full NBA ARM pipeline (Synergy + Shot Dashboard version)
# Usage:
#   bash run_all.sh              # 2022-23 season
#   bash run_all.sh 2021-22      # specific season

SEASON=${1:-"2022-23"}

echo "========================================================"
echo "  NBA ARM Pipeline  |  Season: $SEASON"
echo "========================================================"

echo ""
echo "[1/4] Pulling data (Synergy + Dribbles + Touch Time)..."
python 1_data_pull.py --season $SEASON

echo ""
echo "[2/4] Building transactions..."
python 2_preprocessing.py --season $SEASON

echo ""
echo "[3/4] Mining association rules..."
python 3_arm_mining.py --min_support 0.02 --min_confidence 0.30 --min_lift 1.1

echo ""
echo "[4/4] Generating visualizations..."
python 4_visualization.py

echo ""
echo "========================================================"
echo "  Done!"
echo "  Rules       -> data/rules.csv"
echo "  Rich rules  -> data/rules_combined_only.csv"
echo "  Plots       -> data/plots/"
echo "========================================================"
