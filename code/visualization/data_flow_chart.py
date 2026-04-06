# %%
"""
Data Flow Chart (CONSORT Style)
================================
Creates a flow diagram showing the data selection process with
Inclusion/Exclusion criteria for the transfusion prediction study.

Generates a CONSORT-style flowchart showing:
1. Initial cohort from MIMIC-IV database
2. Inclusion/Exclusion criteria application
3. Final study cohort
4. Train/Test split
5. Transfused patient subgroup
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for script execution
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
from sklearn.model_selection import train_test_split

# %%
# Configuration
RANDOM_STATE = 1004
TRANSFUSION_TYPES = ['rbc_volume', 'ffp_volume', 'platelet_volume']
TIMEFRAMES = ['24h', '48h', '72h']

output_dir = Path("../../result/figures")
output_dir.mkdir(parents=True, exist_ok=True)

# %%
# Load data and calculate statistics
print("Loading data and calculating statistics...")

df_orig = pd.read_parquet("../../data/raw/analysis_data.parquet")
total_n = len(df_orig)
print(f"Total patients in analysis cohort: {total_n}")

# Recreate train/test split
stratify_cols = [
    'icu_mortality', 'hospital_mortality', 'mortality_28day',
    'blood_culture_positive_after_72h', 'new_antibiotic_after_72h'
]

df_orig['stratify_group'] = (
    df_orig[stratify_cols]
    .astype(str)
    .agg('_'.join, axis=1)
)

group_counts = df_orig['stratify_group'].value_counts()
rare_groups = group_counts[group_counts < 20].index.tolist()

if len(rare_groups) > 0:
    df_orig.loc[df_orig['stratify_group'].isin(rare_groups), 'stratify_group'] = 'rare'

df_train, df_test = train_test_split(
    df_orig,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=df_orig['stratify_group']
)

train_n = len(df_train)
test_n = len(df_test)

# Calculate transfused patients
def has_any_transfusion(df):
    """Check if patient received any transfusion at any timeframe."""
    transfusion_cols = []
    for tf in TIMEFRAMES:
        for t in TRANSFUSION_TYPES:
            col = f"{t}_{tf}"
            if col in df.columns:
                transfusion_cols.append(col)
    return (df[transfusion_cols].fillna(0) > 0).any(axis=1)

train_transfused_mask = has_any_transfusion(df_train)
test_transfused_mask = has_any_transfusion(df_test)

train_transfused_n = train_transfused_mask.sum()
test_transfused_n = test_transfused_mask.sum()
total_transfused_n = train_transfused_n + test_transfused_n

# Non-transfused
train_nontransfused_n = train_n - train_transfused_n
test_nontransfused_n = test_n - test_transfused_n
total_nontransfused_n = total_n - total_transfused_n

print(f"\n=== Data Statistics ===")
print(f"Total cohort: {total_n}")
print(f"  - Training: {train_n} (80%)")
print(f"  - Test: {test_n} (20%)")
print(f"\nTransfused patients: {total_transfused_n} ({total_transfused_n/total_n*100:.1f}%)")
print(f"  - Training: {train_transfused_n}")
print(f"  - Test: {test_transfused_n}")

# Calculate outcome prevalences in transfused cohort
transfused_mask = has_any_transfusion(df_orig)
df_transfused = df_orig[transfused_mask]

TARGET_COLS = [
    'icu_mortality', 'hospital_mortality', 'mortality_28day',
    'blood_culture_positive_after_72h', 'new_antibiotic_after_72h'
]

outcome_prevalences = {}
print(f"\n=== Outcome Prevalences (Transfused Cohort) ===")
for col in TARGET_COLS:
    prev = df_transfused[col].mean() * 100
    outcome_prevalences[col] = prev
    print(f"  {col}: {prev:.1f}%")


# %%
def draw_flow_chart(total_n, train_n, test_n, 
                    train_transfused_n, test_transfused_n,
                    train_nontransfused_n, test_nontransfused_n,
                    outcome_prevalences,
                    initial_icu_stays=94458,
                    figsize=(12, 18), dpi=150):
    """
    Draw CONSORT-style data flow chart.
    Train/Test split comes at the end after transfusion filtering.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Colors
    colors = {
        'database': '#2E86AB',      # Blue - data source
        'exclusion': '#DC3545',     # Red - exclusion
        'cohort': '#6C757D',        # Gray - main cohort
        'split': '#FFC107',         # Yellow - split
        'train': '#17A2B8',         # Cyan - training
        'test': '#6F42C1',          # Purple - test
        'subgroup': '#E83E8C',      # Pink - transfused subgroup
        'outcome': '#28A745',       # Green - outcomes
        'arrow': '#343A40'          # Dark gray - arrows
    }
    
    total_transfused = train_transfused_n + test_transfused_n
    transfusion_rate = total_transfused / total_n * 100
    excluded_n = initial_icu_stays - total_n
    
    # Box dimensions and positions
    box1 = {'y': 13.0, 'h': 0.8}   # MIMIC-IV
    box2 = {'y': 11.3, 'h': 0.9}   # First ICU stay filter
    box3 = {'y': 9.5, 'h': 0.8}    # Study Cohort
    box4 = {'y': 7.8, 'h': 1.0}    # Transfusion Filter
    box5 = {'y': 6.0, 'h': 0.8}    # Transfused
    box6 = {'y': 4.4, 'h': 0.7}    # Split
    box7 = {'y': 3.0, 'h': 0.8}    # Train/Test
    box8 = {'y': 1.2, 'h': 1.6}    # Outcomes
    
    gap = 0.15
    
    def add_box(ax, x, y, width, height, text, color='#6C757D', 
                fontsize=10, fontweight='normal', alpha=0.9, text_color='white'):
        """Add a rounded box with text."""
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                             boxstyle="round,pad=0.02,rounding_size=0.15",
                             facecolor=color, edgecolor='black', 
                             linewidth=1.5, alpha=alpha)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                fontweight=fontweight, color=text_color, wrap=True,
                multialignment='center')
        return box
    
    def add_vertical_arrow(ax, x, y_start, y_end, color='#343A40'):
        """Add a vertical arrow."""
        ax.annotate('', xy=(x, y_end), xytext=(x, y_start),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2,
                                  shrinkA=0, shrinkB=0))
    
    # ========================================
    # Level 1: Data Source
    # ========================================
    add_box(ax, 5, box1['y'], 5.5, box1['h'],
            f"MIMIC-IV Database\n(n = {initial_icu_stays:,} ICU stays)",
            color=colors['database'], fontsize=11, fontweight='bold')
    
    add_vertical_arrow(ax, 5, box1['y'] - box1['h']/2 - gap, 
                          box2['y'] + box2['h']/2 + gap)
    
    # ========================================
    # Level 2: First ICU Stay Selection
    # ========================================
    add_box(ax, 5, box2['y'], 7, box2['h'],
            f"First ICU stay per patient selected\n"
            f"(to ensure independence of observations)\n"
            f"Excluded: {excluded_n:,} subsequent ICU stays",
            color=colors['exclusion'], fontsize=9, fontweight='normal')
    
    add_vertical_arrow(ax, 5, box2['y'] - box2['h']/2 - gap, 
                          box3['y'] + box3['h']/2 + gap)
    
    # ========================================
    # Level 3: Study Cohort
    # ========================================
    add_box(ax, 5, box3['y'], 5.5, box3['h'],
            f"Study Cohort\n(n = {total_n:,} unique ICU admissions)",
            color=colors['cohort'], fontsize=12, fontweight='bold')
    
    add_vertical_arrow(ax, 5, box3['y'] - box3['h']/2 - gap, 
                          box4['y'] + box4['h']/2 + gap)
    
    # ========================================
    # Level 4: Transfusion Filter
    # ========================================
    add_box(ax, 5, box4['y'], 7, box4['h'],
            "Transfusion Subgroup Filter\n"
            "• Received any transfusion (RBC, FFP, or Platelet)\n"
            "• Within 72 hours of ICU admission",
            color=colors['subgroup'], fontsize=9, fontweight='normal')
    
    add_vertical_arrow(ax, 5, box4['y'] - box4['h']/2 - gap, 
                          box5['y'] + box5['h']/2 + gap)
    
    # ========================================
    # Level 5: Transfused Cohort
    # ========================================
    add_box(ax, 5, box5['y'], 5.5, box5['h'],
            f"Transfused Cohort\n(n = {total_transfused:,}, {transfusion_rate:.1f}%)",
            color=colors['subgroup'], fontsize=12, fontweight='bold')
    
    add_vertical_arrow(ax, 5, box5['y'] - box5['h']/2 - gap, 
                          box6['y'] + box6['h']/2 + gap)
    
    # ========================================
    # Level 6: Train/Test Split Box
    # ========================================
    add_box(ax, 5, box6['y'], 4.5, box6['h'],
            "Stratified Random Split\n(80% Training / 20% Test)",
            color=colors['split'], fontsize=10, fontweight='bold', text_color='black')
    
    # Branch to train/test
    split_bottom = box6['y'] - box6['h']/2 - gap
    branch_y = split_bottom - 0.2
    train_test_top = box7['y'] + box7['h']/2 + gap
    
    ax.plot([5, 5], [split_bottom, branch_y], color=colors['arrow'], lw=2)
    ax.plot([3, 7], [branch_y, branch_y], color=colors['arrow'], lw=2)
    ax.plot([3, 3], [branch_y, train_test_top + 0.05], color=colors['arrow'], lw=2)
    ax.plot([7, 7], [branch_y, train_test_top + 0.05], color=colors['arrow'], lw=2)
    add_vertical_arrow(ax, 3, train_test_top + 0.05, train_test_top)
    add_vertical_arrow(ax, 7, train_test_top + 0.05, train_test_top)
    
    # ========================================
    # Level 7: Training and Test Sets
    # ========================================
    add_box(ax, 3, box7['y'], 3.5, box7['h'],
            f"Training Set\n(n = {train_transfused_n:,})",
            color=colors['train'], fontsize=11, fontweight='bold')
    
    add_box(ax, 7, box7['y'], 3.5, box7['h'],
            f"Test Set\n(n = {test_transfused_n:,})",
            color=colors['test'], fontsize=11, fontweight='bold')
    
    # Arrow to outcomes
    train_test_bottom = box7['y'] - box7['h']/2 - gap
    outcome_branch_y = train_test_bottom - 0.2
    outcome_top = box8['y'] + box8['h']/2 + gap
    
    ax.plot([3, 3], [train_test_bottom, outcome_branch_y], color=colors['arrow'], lw=2)
    ax.plot([7, 7], [train_test_bottom, outcome_branch_y], color=colors['arrow'], lw=2)
    ax.plot([3, 7], [outcome_branch_y, outcome_branch_y], color=colors['arrow'], lw=2)
    ax.plot([5, 5], [outcome_branch_y, outcome_top + 0.05], color=colors['arrow'], lw=2)
    add_vertical_arrow(ax, 5, outcome_top + 0.05, outcome_top)
    
    # ========================================
    # Level 8: Outcomes
    # ========================================
    outcome_text = "Prediction Outcomes (Prevalence in Transfused Cohort)\n"
    outcome_text += "━" * 45 + "\n"
    outcome_text += f"• ICU Mortality: {outcome_prevalences['icu_mortality']:.1f}%\n"
    outcome_text += f"• Hospital Mortality: {outcome_prevalences['hospital_mortality']:.1f}%\n"
    outcome_text += f"• 28-Day Mortality: {outcome_prevalences['mortality_28day']:.1f}%\n"
    outcome_text += f"• Blood Culture Positive (>72h): {outcome_prevalences['blood_culture_positive_after_72h']:.1f}%\n"
    outcome_text += f"• New Antibiotic (>72h): {outcome_prevalences['new_antibiotic_after_72h']:.1f}%"
    
    add_box(ax, 5, box8['y'], 7.5, box8['h'],
            outcome_text,
            color=colors['outcome'], fontsize=9, fontweight='normal')
    
    # Title
    ax.text(5, 13.6, "Data Flow Chart: Transfusion Prediction Study",
            ha='center', va='center', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig, ax


# %%
# Generate the flow chart
fig, ax = draw_flow_chart(
    total_n=total_n,
    train_n=train_n,
    test_n=test_n,
    train_transfused_n=train_transfused_n,
    test_transfused_n=test_transfused_n,
    train_nontransfused_n=train_nontransfused_n,
    test_nontransfused_n=test_nontransfused_n,
    outcome_prevalences=outcome_prevalences
)

# Save figures
png_path = output_dir / "data_flow_chart.png"
pdf_path = output_dir / "data_flow_chart.pdf"
svg_path = output_dir / "data_flow_chart.svg"

fig.savefig(png_path, dpi=500, bbox_inches='tight', facecolor='white')
fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
fig.savefig(svg_path, bbox_inches='tight', facecolor='white')

print(f"\n=== Flow Chart Saved ===")
print(f"PNG: {png_path}")
print(f"PDF: {pdf_path}")
print(f"SVG: {svg_path}")

plt.close(fig)

# %%
# Also create a simplified version for publication
def draw_simple_flow_chart(total_n, train_n, test_n, 
                           train_transfused_n, test_transfused_n,
                           outcome_prevalences,
                           initial_icu_stays=94458,
                           figsize=(10, 16), dpi=150):
    """
    Draw simplified CONSORT-style flow chart for publication.
    Train/Test split comes at the end after transfusion filtering.
    """
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Simplified color scheme (grayscale friendly)
    box_color = '#E8E8E8'
    border_color = '#333333'
    text_color = '#000000'
    
    total_transfused = train_transfused_n + test_transfused_n
    transfusion_rate = total_transfused / total_n * 100
    excluded_n = initial_icu_stays - total_n
    
    # Box dimensions and positions
    box1 = {'y': 13.0, 'h': 0.7}   # MIMIC-IV
    box2 = {'y': 11.5, 'h': 0.8}   # First ICU stay
    box3 = {'y': 10.0, 'h': 0.7}   # Study Cohort
    box4 = {'y': 8.5, 'h': 0.8}    # Transfusion Filter
    box5 = {'y': 7.0, 'h': 0.7}    # Transfused
    box6 = {'y': 5.6, 'h': 0.6}    # Split
    box7 = {'y': 4.2, 'h': 0.7}    # Train/Test
    box8 = {'y': 1.8, 'h': 1.8}    # Outcomes
    
    gap = 0.15
    
    def add_box(ax, x, y, width, height, text, fontsize=10):
        """Add a simple box with text."""
        box = FancyBboxPatch((x - width/2, y - height/2), width, height,
                             boxstyle="round,pad=0.02,rounding_size=0.1",
                             facecolor=box_color, edgecolor=border_color, 
                             linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
                color=text_color, multialignment='center')
    
    def add_vertical_arrow(ax, x, y_start, y_end):
        """Add a vertical arrow."""
        ax.annotate('', xy=(x, y_end), xytext=(x, y_start),
                   arrowprops=dict(arrowstyle='->', color=border_color, lw=1.5,
                                  shrinkA=0, shrinkB=0))
    
    # Level 1: MIMIC-IV
    add_box(ax, 5, box1['y'], 5, box1['h'], 
            f"MIMIC-IV Database\n(n = {initial_icu_stays:,} ICU stays)", fontsize=11)
    add_vertical_arrow(ax, 5, box1['y'] - box1['h']/2 - gap, 
                          box2['y'] + box2['h']/2 + gap)
    
    # Level 2: First ICU stay selection
    add_box(ax, 5, box2['y'], 6.5, box2['h'],
            f"First ICU stay per patient\n"
            f"Excluded: {excluded_n:,} subsequent stays",
            fontsize=9)
    add_vertical_arrow(ax, 5, box2['y'] - box2['h']/2 - gap, 
                          box3['y'] + box3['h']/2 + gap)
    
    # Level 3: Study cohort
    add_box(ax, 5, box3['y'], 5, box3['h'], 
            f"Study Cohort\n(n = {total_n:,})", fontsize=11)
    add_vertical_arrow(ax, 5, box3['y'] - box3['h']/2 - gap, 
                          box4['y'] + box4['h']/2 + gap)
    
    # Level 4: Transfusion filter
    add_box(ax, 5, box4['y'], 6.5, box4['h'],
            f"Filter: Any Transfusion within 72h\n"
            f"(RBC, FFP, or Platelet)",
            fontsize=9)
    add_vertical_arrow(ax, 5, box4['y'] - box4['h']/2 - gap, 
                          box5['y'] + box5['h']/2 + gap)
    
    # Level 5: Transfused cohort
    add_box(ax, 5, box5['y'], 5, box5['h'],
            f"Transfused Cohort\n(n = {total_transfused:,}, {transfusion_rate:.1f}%)",
            fontsize=11)
    add_vertical_arrow(ax, 5, box5['y'] - box5['h']/2 - gap, 
                          box6['y'] + box6['h']/2 + gap)
    
    # Level 6: Split
    add_box(ax, 5, box6['y'], 4.5, box6['h'], "80/20 Stratified Split", fontsize=10)
    
    # Branch to train/test
    split_bottom = box6['y'] - box6['h']/2 - gap
    branch_y = split_bottom - 0.15
    train_test_top = box7['y'] + box7['h']/2 + gap
    
    ax.plot([5, 5], [split_bottom, branch_y], color=border_color, lw=1.5)
    ax.plot([3, 7], [branch_y, branch_y], color=border_color, lw=1.5)
    ax.plot([3, 3], [branch_y, train_test_top + 0.05], color=border_color, lw=1.5)
    ax.plot([7, 7], [branch_y, train_test_top + 0.05], color=border_color, lw=1.5)
    add_vertical_arrow(ax, 3, train_test_top + 0.05, train_test_top)
    add_vertical_arrow(ax, 7, train_test_top + 0.05, train_test_top)
    
    # Level 7: Train/Test sets
    add_box(ax, 3, box7['y'], 3.2, box7['h'],
            f"Training Set\n(n = {train_transfused_n:,})", fontsize=10)
    add_box(ax, 7, box7['y'], 3.2, box7['h'],
            f"Test Set\n(n = {test_transfused_n:,})", fontsize=10)
    
    # Arrow to outcomes
    train_test_bottom = box7['y'] - box7['h']/2 - gap
    outcome_branch_y = train_test_bottom - 0.15
    outcome_top = box8['y'] + box8['h']/2 + gap
    
    ax.plot([3, 3], [train_test_bottom, outcome_branch_y], color=border_color, lw=1.5)
    ax.plot([7, 7], [train_test_bottom, outcome_branch_y], color=border_color, lw=1.5)
    ax.plot([3, 7], [outcome_branch_y, outcome_branch_y], color=border_color, lw=1.5)
    ax.plot([5, 5], [outcome_branch_y, outcome_top + 0.05], color=border_color, lw=1.5)
    add_vertical_arrow(ax, 5, outcome_top + 0.05, outcome_top)
    
    # Level 8: Outcomes
    outcome_text = "Prediction Outcomes\n(Prevalence in Transfused Cohort)\n"
    outcome_text += "─" * 35 + "\n"
    outcome_text += f"ICU Mortality: {outcome_prevalences['icu_mortality']:.1f}%\n"
    outcome_text += f"Hospital Mortality: {outcome_prevalences['hospital_mortality']:.1f}%\n"
    outcome_text += f"28-Day Mortality: {outcome_prevalences['mortality_28day']:.1f}%\n"
    outcome_text += f"Blood Culture (+) >72h: {outcome_prevalences['blood_culture_positive_after_72h']:.1f}%\n"
    outcome_text += f"New Antibiotic >72h: {outcome_prevalences['new_antibiotic_after_72h']:.1f}%"
    
    add_box(ax, 5, box8['y'], 6.5, box8['h'], outcome_text, fontsize=9)
    
    plt.tight_layout()
    return fig, ax


# %%
# Generate simplified flow chart
fig_simple, ax_simple = draw_simple_flow_chart(
    total_n=total_n,
    train_n=train_n,
    test_n=test_n,
    train_transfused_n=train_transfused_n,
    test_transfused_n=test_transfused_n,
    outcome_prevalences=outcome_prevalences
)

# Save simplified version
simple_png_path = output_dir / "data_flow_chart_simple.png"
simple_pdf_path = output_dir / "data_flow_chart_simple.pdf"
simple_svg_path = output_dir / "data_flow_chart_simple.svg"

fig_simple.savefig(simple_png_path, dpi=500, bbox_inches='tight', facecolor='white')
fig_simple.savefig(simple_pdf_path, bbox_inches='tight', facecolor='white')
fig_simple.savefig(simple_svg_path, bbox_inches='tight', facecolor='white')

print(f"\n=== Simplified Flow Chart Saved ===")
print(f"PNG: {simple_png_path}")
print(f"PDF: {simple_pdf_path}")
print(f"SVG: {simple_svg_path}")

plt.close(fig_simple)

# %%
print("\n" + "="*60)
print("DATA FLOW CHART GENERATION COMPLETE")
print("="*60)
print(f"\nOutput directory: {output_dir}")
print(f"\nGenerated files:")
print(f"  - data_flow_chart.png/pdf/svg (detailed version)")
print(f"  - data_flow_chart_simple.png/pdf/svg (publication version)")
print(f"\nData Statistics:")
print(f"  Total cohort: {total_n:,}")
print(f"  Training set: {train_n:,}")
print(f"  Test set: {test_n:,}")
print(f"  Transfused (total): {total_transfused_n:,} ({total_transfused_n/total_n*100:.1f}%)")
print(f"  Transfused (train): {train_transfused_n:,}")
print(f"  Transfused (test): {test_transfused_n:,}")

# %%
