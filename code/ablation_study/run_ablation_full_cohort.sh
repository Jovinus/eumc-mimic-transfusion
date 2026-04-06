#!/bin/bash
# Ablation Study: Full Cohort Only (All patients)
# ================================================

set -e

echo "=============================================="
echo "ABLATION STUDY: Full Cohort (No Medical Scores)"
echo "=============================================="
echo ""

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Step 1: Training models..."
python sklearn_train_no_scores.py

echo ""
echo "Step 2: Generating ROC/PRC curves..."
python plot_roc_prc_curves_no_scores.py

echo ""
echo "Step 3: Generating SHAP plots..."
python plot_shap_beeswarm_no_scores.py

echo ""
echo "=============================================="
echo "FULL COHORT ABLATION COMPLETE!"
echo "=============================================="
echo ""
echo "Output files:"
echo "  - Results: result/model_performance_summary_ablation_no_scores.csv"
echo "  - Comparison: result/model_comparison_ablation_no_scores.csv"
echo "  - Models: result/models_ablation_no_scores/"
echo "  - ROC/PRC: result/figures_ablation_no_scores/curves/"
echo "  - SHAP: result/figures_ablation_no_scores/shap/"
