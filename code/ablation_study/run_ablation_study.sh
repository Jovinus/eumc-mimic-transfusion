#!/bin/bash
# Ablation Study: Model Training and Visualization without Medical Scores
# ========================================================================
# This script runs the complete ablation study pipeline for BOTH cohorts:
# 1. Full cohort (all patients with and without transfusion)
# 2. Transfused cohort (only patients who received transfusion)
#
# For each cohort, it:
# - Trains models without medical scores (SOFA, APACHE2)
# - Generates ROC/PRC curves
# - Generates SHAP beeswarm plots

set -e

echo "=============================================="
echo "ABLATION STUDY: No Medical Scores (SOFA, APACHE2)"
echo "=============================================="
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# ================================================
# FULL COHORT (All patients)
# ================================================
echo ""
echo "######################################################"
echo "# COHORT 1: FULL COHORT (All patients)"
echo "######################################################"
echo ""

echo "Step 1.1: Training models (Full cohort)..."
echo "==========================================="
python sklearn_train_no_scores.py

echo ""
echo "Step 1.2: Generating ROC/PRC curves (Full cohort)..."
echo "====================================================="
python plot_roc_prc_curves_no_scores.py

echo ""
echo "Step 1.3: Generating SHAP plots (Full cohort)..."
echo "================================================="
python plot_shap_beeswarm_no_scores.py

# ================================================
# TRANSFUSED COHORT (Patients with transfusion)
# ================================================
echo ""
echo "######################################################"
echo "# COHORT 2: TRANSFUSED COHORT (Transfusion patients)"
echo "######################################################"
echo ""

echo "Step 2.1: Training models (Transfused cohort)..."
echo "================================================="
python sklearn_train_no_scores_transfused.py

echo ""
echo "Step 2.2: Generating ROC/PRC curves (Transfused cohort)..."
echo "==========================================================="
python plot_roc_prc_curves_no_scores_transfused.py

echo ""
echo "Step 2.3: Generating SHAP plots (Transfused cohort)..."
echo "======================================================="
python plot_shap_beeswarm_no_scores_transfused.py

# ================================================
# SUMMARY
# ================================================
echo ""
echo "=============================================="
echo "ABLATION STUDY COMPLETE!"
echo "=============================================="
echo ""
echo "FULL COHORT Output:"
echo "  - Results: result/model_performance_summary_ablation_no_scores.csv"
echo "  - Comparison: result/model_comparison_ablation_no_scores.csv"
echo "  - Models: result/models_ablation_no_scores/"
echo "  - ROC/PRC: result/figures_ablation_no_scores/curves/"
echo "  - SHAP: result/figures_ablation_no_scores/shap/"
echo ""
echo "TRANSFUSED COHORT Output:"
echo "  - Results: result/model_performance_summary_ablation_no_scores_transfused.csv"
echo "  - Comparison: result/model_comparison_ablation_no_scores_transfused.csv"
echo "  - Models: result/models_ablation_no_scores_transfused/"
echo "  - ROC/PRC: result/figures_ablation_no_scores_transfused/curves/"
echo "  - SHAP: result/figures_ablation_no_scores_transfused/shap/"
