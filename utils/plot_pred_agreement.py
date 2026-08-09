import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects


###############################################################################
# CONFIGURATION
###############################################################################

analysis_dir = (
    r"output_v154/"
    r"Frequency-based-Generalization-and-Robustness-Enhancer/"
    r"output/AveragingModel/"
    r"Fused_CNN_ResNet50_CNN_ResNet50_Averaging/"
    r"analysis"
)

save_dir = "lf_hf_prediction_analysis"

os.makedirs(save_dir, exist_ok=True)


###############################################################################
# IMPORTANT
#
# In your saved CSV:
#
#     pred = output[:, 1]
#
# Therefore pred is the FAKE probability.
#
#     pred >= 0.5 -> Fake
#     pred <  0.5 -> Real
#
###############################################################################

THRESHOLD = 0.5


###############################################################################
# FIND ALL DATASETS
###############################################################################

files = glob.glob(
    os.path.join(
        analysis_dir,
        "*_test_0_meta.csv"
    )
)

datasets = sorted(
    os.path.basename(f).replace(
        "_test_0_meta.csv",
        ""
    )
    for f in files
)


print("\n" + "=" * 80)
print("Datasets found:")
print("=" * 80)

for dataset in datasets:
    print(f"  {dataset}")

print("=" * 80)


###############################################################################
# STORAGE
###############################################################################

agreement_results = []

detailed_results = []


###############################################################################
# HELPER
###############################################################################

def add_text_outline(text_object):
    """
    Add a subtle white outline around text.
    """

    text_object.set_path_effects([
        path_effects.Stroke(
            linewidth=2.5,
            foreground="white"
        ),
        path_effects.Normal()
    ])


###############################################################################
# PROCESS EACH DATASET
###############################################################################

for dataset in datasets:

    print("\n" + "=" * 80)
    print(f"Processing: {dataset}")
    print("=" * 80)


    ###########################################################################
    # PATHS
    ###########################################################################

    lf_path = os.path.join(
        analysis_dir,
        f"{dataset}_test_0_meta.csv"
    )

    hf_path = os.path.join(
        analysis_dir,
        f"{dataset}_test_1_meta.csv"
    )


    ###########################################################################
    # LOAD
    ###########################################################################

    lf_meta = pd.read_csv(lf_path)

    hf_meta = pd.read_csv(hf_path)


    ###########################################################################
    # SANITY CHECKS
    ###########################################################################

    if len(lf_meta) != len(hf_meta):

        raise ValueError(
            f"{dataset}: LF/HF sample count mismatch: "
            f"{len(lf_meta)} vs {len(hf_meta)}"
        )


    required_columns = {"label", "pred"}


    if not required_columns.issubset(lf_meta.columns):

        raise ValueError(
            f"{lf_path} must contain columns: {required_columns}"
        )


    if "pred" not in hf_meta.columns:

        raise ValueError(
            f"{hf_path} must contain column: pred"
        )


    ###########################################################################
    # GROUND TRUTH
    #
    # 0 = Real
    # 1 = Fake
    ###########################################################################

    gt = lf_meta["label"].to_numpy()


    ###########################################################################
    # FAKE PROBABILITY
    ###########################################################################

    lf_prob = lf_meta["pred"].to_numpy()

    hf_prob = hf_meta["pred"].to_numpy()


    ###########################################################################
    # CONVERT PROBABILITY -> PREDICTED LABEL
    #
    # >= 0.5 -> Fake
    # <  0.5 -> Real
    ###########################################################################

    lf_fake = lf_prob >= THRESHOLD

    hf_fake = hf_prob >= THRESHOLD


    ###########################################################################
    # TOTAL
    ###########################################################################

    total = len(gt)


    if total == 0:

        print(
            f"WARNING: {dataset} contains zero samples."
        )

        continue


    ###########################################################################
    # LF / HF CORRECTNESS
    ###########################################################################

    lf_correct = (
        lf_fake.astype(int) == gt
    )

    hf_correct = (
        hf_fake.astype(int) == gt
    )


    ###########################################################################
    # FOUR CORRECTNESS STATES
    ###########################################################################

    # LF correct, HF wrong
    lf_correct_only = (
        lf_correct
        & (~hf_correct)
    )

    # LF wrong, HF correct
    hf_correct_only = (
        (~lf_correct)
        & hf_correct
    )

    # Both correct
    both_correct = (
        lf_correct
        & hf_correct
    )

    # Both wrong
    both_wrong = (
        (~lf_correct)
        & (~hf_correct)
    )


    ###########################################################################
    # PREDICTION AGREEMENT / DISAGREEMENT
    ###########################################################################

    prediction_agreement = (
        lf_fake == hf_fake
    )

    prediction_disagreement = (
        lf_fake != hf_fake
    )


    agreement_pct = (
        100.0
        * np.sum(prediction_agreement)
        / total
    )


    disagreement_pct = (
        100.0
        * np.sum(prediction_disagreement)
        / total
    )


    ###########################################################################
    # CORRECTNESS PERCENTAGES
    ###########################################################################

    lf_correct_only_pct = (
        100.0
        * np.sum(lf_correct_only)
        / total
    )


    hf_correct_only_pct = (
        100.0
        * np.sum(hf_correct_only)
        / total
    )


    both_correct_pct = (
        100.0
        * np.sum(both_correct)
        / total
    )


    both_wrong_pct = (
        100.0
        * np.sum(both_wrong)
        / total
    )


    ###########################################################################
    # FOUR PREDICTION COMBINATIONS
    ###########################################################################

    # LF Fake / HF Real
    mask_lf_fake_hf_real = (
        lf_fake
        & (~hf_fake)
    )

    # LF Real / HF Fake
    mask_lf_real_hf_fake = (
        (~lf_fake)
        & hf_fake
    )

    # LF Fake / HF Fake
    mask_lf_fake_hf_fake = (
        lf_fake
        & hf_fake
    )

    # LF Real / HF Real
    mask_lf_real_hf_real = (
        (~lf_fake)
        & (~hf_fake)
    )


    combination_masks = [

        mask_lf_fake_hf_real,

        mask_lf_real_hf_fake,

        mask_lf_fake_hf_fake,

        mask_lf_real_hf_real,
    ]


    combination_names = [

        "LF Fake / HF Real",

        "LF Real / HF Fake",

        "LF Fake / HF Fake",

        "LF Real / HF Real",
    ]


    ###########################################################################
    # PERCENTAGES FOR EACH PREDICTION COMBINATION
    #
    # All percentages are relative to the ENTIRE DATASET.
    ###########################################################################

    combination_pct = []

    combination_both_wrong_pct = []


    for mask in combination_masks:

        combo_total = np.sum(mask)


        combo_pct = (
            100.0
            * combo_total
            / total
        )


        combination_pct.append(
            combo_pct
        )


        #######################################################################
        # Both wrong within this prediction combination
        #######################################################################

        combo_both_wrong = (
            mask
            & both_wrong
        )


        combo_both_wrong_percentage = (
            100.0
            * np.sum(combo_both_wrong)
            / total
        )


        combination_both_wrong_pct.append(
            combo_both_wrong_percentage
        )


    ###########################################################################
    # SAVE AGREEMENT RESULTS
    ###########################################################################

    agreement_results.append({

        "Dataset":
            dataset,

        "Agreement (%)":
            agreement_pct,

        "Disagreement (%)":
            disagreement_pct,
    })


    ###########################################################################
    # SAVE DETAILED RESULTS
    ###########################################################################

    detailed_results.append({

        "Dataset":
            dataset,


        # Prediction combinations
        "LF Fake / HF Real (%)":
            combination_pct[0],

        "LF Real / HF Fake (%)":
            combination_pct[1],

        "LF Fake / HF Fake (%)":
            combination_pct[2],

        "LF Real / HF Real (%)":
            combination_pct[3],


        # Correctness
        "LF correct only (%)":
            lf_correct_only_pct,

        "HF correct only (%)":
            hf_correct_only_pct,

        "Both correct (%)":
            both_correct_pct,

        "Both wrong (%)":
            both_wrong_pct,


        # Agreement
        "Agreement (%)":
            agreement_pct,

        "Disagreement (%)":
            disagreement_pct,


        "Total":
            total,
    })


    ###########################################################################
    # PRINT RESULTS
    ###########################################################################

    print(
        f"Total samples: {total:,}"
    )

    print(
        f"Agreement:       {agreement_pct:.2f}%"
    )

    print(
        f"Disagreement:    {disagreement_pct:.2f}%"
    )

    print(
        f"LF correct only: {lf_correct_only_pct:.2f}%"
    )

    print(
        f"HF correct only: {hf_correct_only_pct:.2f}%"
    )

    print(
        f"Both correct:    {both_correct_pct:.2f}%"
    )

    print(
        f"Both wrong:      {both_wrong_pct:.2f}%"
    )

    print("\nPrediction combinations:")

    for name, pct, wrong_pct in zip(
        combination_names,
        combination_pct,
        combination_both_wrong_pct
    ):

        print(
            f"  {name:<22} "
            f"{pct:6.2f}% "
            f"(both wrong: {wrong_pct:.2f}%)"
        )


###############################################################################
# CREATE DATAFRAMES
###############################################################################

agreement_df = pd.DataFrame(
    agreement_results
)

detailed_df = pd.DataFrame(
    detailed_results
)


###############################################################################
# SAVE CSV FILES
###############################################################################

agreement_csv = os.path.join(
    save_dir,
    "agreement_disagreement.csv"
)


detailed_csv = os.path.join(
    save_dir,
    "prediction_correctness.csv"
)


agreement_df.to_csv(
    agreement_csv,
    index=False
)


detailed_df.to_csv(
    detailed_csv,
    index=False
)


print("\n" + "=" * 80)
print("CSV files saved:")
print("=" * 80)

print(
    f"  {agreement_csv}"
)

print(
    f"  {detailed_csv}"
)


###############################################################################
# PLOT 1 — STACKED AGREEMENT / DISAGREEMENT
#
# ONE bar per dataset:
#
#   Blue   = Prediction Agreement
#   Orange = Prediction Disagreement
#
# Only the BLUE / AGREEMENT percentage is displayed.
#
# The legend is INSIDE the plot.
###############################################################################

print("\nCreating Plot 1: Prediction Agreement...")


###############################################################################
# CSV PATH
#
# prediction_correctness.csv is NOT inside analysis_dir.
#
# It is inside:
#
#     lf_hf_prediction_analysis/
#
###############################################################################

prediction_csv = os.path.join(
    save_dir,
    "prediction_correctness.csv"
)


###############################################################################
# CHECK FILE
###############################################################################

if not os.path.exists(prediction_csv):

    raise FileNotFoundError(
        f"\nCould not find:\n"
        f"{prediction_csv}\n\n"
        f"Make sure prediction_correctness.csv has already been generated."
    )


###############################################################################
# LOAD
###############################################################################

df = pd.read_csv(
    prediction_csv
)


print(
    f"Loaded:\n{prediction_csv}"
)

print(
    "\nCSV columns:"
)

print(
    df.columns.tolist()
)


###############################################################################
# COLUMN NAMES
#
# These match the CSV generated by the previous full script.
###############################################################################

datasets = df["Dataset"].to_numpy()

agreement = (
    df["Agreement (%)"].to_numpy()
)

disagreement = (
    df["Disagreement (%)"].to_numpy()
)


###############################################################################
# SANITY CHECK
###############################################################################

if not np.allclose(
    agreement + disagreement,
    100.0,
    atol=0.01
):

    print(
        "\nWARNING: Agreement + Disagreement is not exactly 100%."
    )

    print(
        "Normalizing them to 100% for visualization."
    )

    total = (
        agreement
        + disagreement
    )

    agreement = (
        agreement
        / total
        * 100.0
    )

    disagreement = (
        disagreement
        / total
        * 100.0
    )


###############################################################################
# FIGURE
###############################################################################

fig, ax = plt.subplots(
    figsize=(15, 8)
)


###############################################################################
# X POSITION
###############################################################################

x = np.arange(
    len(datasets)
)


###############################################################################
# BAR WIDTH
###############################################################################

width = 0.65


###############################################################################
# AGREEMENT — BOTTOM
###############################################################################

bars_agreement = ax.bar(
    x,
    agreement,
    width=width,
    label="Prediction Agreement"
)


###############################################################################
# DISAGREEMENT — STACKED ON TOP
###############################################################################

bars_disagreement = ax.bar(
    x,
    disagreement,
    width=width,
    bottom=agreement,
    label="Prediction Disagreement"
)


###############################################################################
# ONLY SHOW AGREEMENT %
#
# IMPORTANT:
#
# We do NOT display the disagreement percentage.
###############################################################################

for i, value in enumerate(agreement):

    if value <= 0:
        continue


    ###########################################################################
    # Center of blue agreement section
    ###########################################################################

    y = value / 2


    ###########################################################################
    # Text
    ###########################################################################

    text = ax.text(
        x[i],
        y,
        f"{value:.1f}%",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="black"
    )


    ###########################################################################
    # White outline
    #
    # Makes black text readable regardless of the bar appearance.
    ###########################################################################

    add_text_outline(
        text
    )


###############################################################################
# X AXIS
###############################################################################

ax.set_xticks(
    x
)

ax.set_xticklabels(
    datasets,
    rotation=45,
    ha="right",
    fontsize=11
)


###############################################################################
# Y AXIS
###############################################################################

ax.set_ylim(
    0,
    100
)

ax.set_ylabel(
    "Percentage of dataset (%)",
    fontsize=13
)


###############################################################################
# X LABEL
###############################################################################

ax.set_xlabel(
    "Dataset",
    fontsize=13
)


###############################################################################
# TITLE
###############################################################################

ax.set_title(
    "LF-HF Prediction Agreement and Disagreement",
    fontsize=17,
    pad=15
)


###############################################################################
# GRID
###############################################################################

ax.grid(
    axis="y",
    alpha=0.20
)

ax.set_axisbelow(True)


###############################################################################
# SPINES
###############################################################################

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)


###############################################################################
# LEGEND
#
# INSIDE the plot.
#
# No bbox_to_anchor.
# No external legend.
###############################################################################

ax.legend(
    loc="upper right",
    frameon=True,
    fontsize=11
)


###############################################################################
# LAYOUT
###############################################################################

plt.tight_layout()


###############################################################################
# SAVE
###############################################################################

plot1_path = os.path.join(
    save_dir,
    "agreement_disagreement.png"
)


plt.savefig(
    plot1_path,
    dpi=300,
    bbox_inches="tight"
)


plt.close()


###############################################################################
# DONE
###############################################################################

print(
    f"\nSaved Plot 1:\n{plot1_path}"
)


###############################################################################
# ============================================================================
# PLOT 2
#
# FOR EACH DATASET:
#
#     LF Fake / HF Real
#     LF Real / HF Fake
#     LF Fake / HF Fake
#     LF Real / HF Real
#
#
# STACK COLORS:
#
#     Blue   = LF correct only
#     Orange = HF correct only
#     Green  = Both correct
#     Red    = Both wrong
#
#
# IMPORTANT:
#
# ONLY "BOTH WRONG" HAS A PERCENTAGE LABEL.
#
# This avoids the clutter you had before.
# ============================================================================
###############################################################################

for dataset in datasets:

    print(
        f"\nCreating Plot 2: {dataset}"
    )


    ###########################################################################
    # LOAD
    ###########################################################################

    lf_path = os.path.join(
        analysis_dir,
        f"{dataset}_test_0_meta.csv"
    )


    hf_path = os.path.join(
        analysis_dir,
        f"{dataset}_test_1_meta.csv"
    )


    lf_meta = pd.read_csv(
        lf_path
    )


    hf_meta = pd.read_csv(
        hf_path
    )


    ###########################################################################
    # DATA
    ###########################################################################

    gt = lf_meta["label"].to_numpy()

    lf_prob = lf_meta["pred"].to_numpy()

    hf_prob = hf_meta["pred"].to_numpy()

    total = len(gt)


    ###########################################################################
    # PREDICTIONS
    ###########################################################################

    lf_fake = (
        lf_prob >= THRESHOLD
    )


    hf_fake = (
        hf_prob >= THRESHOLD
    )


    ###########################################################################
    # CORRECTNESS
    ###########################################################################

    lf_correct = (
        lf_fake.astype(int) == gt
    )


    hf_correct = (
        hf_fake.astype(int) == gt
    )


    ###########################################################################
    # FOUR CORRECTNESS STATES
    ###########################################################################

    lf_correct_only = (
        lf_correct
        & (~hf_correct)
    )


    hf_correct_only = (
        (~lf_correct)
        & hf_correct
    )


    both_correct = (
        lf_correct
        & hf_correct
    )


    both_wrong = (
        (~lf_correct)
        & (~hf_correct)
    )


    ###########################################################################
    # FOUR PREDICTION COMBINATIONS
    ###########################################################################

    combination_masks = [

        # LF Fake / HF Real
        (
            lf_fake
            & (~hf_fake)
        ),

        # LF Real / HF Fake
        (
            (~lf_fake)
            & hf_fake
        ),

        # LF Fake / HF Fake
        (
            lf_fake
            & hf_fake
        ),

        # LF Real / HF Real
        (
            (~lf_fake)
            & (~hf_fake)
        ),
    ]


    ###########################################################################
    # X AXIS LABELS
    ###########################################################################

    categories = [

        "LF Fake\nHF Real",

        "LF Real\nHF Fake",

        "LF Fake\nHF Fake",

        "LF Real\nHF Real",
    ]


    ###########################################################################
    # CALCULATE STACK VALUES
    #
    # All values are percentages of the ENTIRE DATASET.
    ###########################################################################

    lf_only_values = []

    hf_only_values = []

    both_correct_values = []

    both_wrong_values = []


    for mask in combination_masks:


        #######################################################################
        # LF CORRECT ONLY
        #######################################################################

        value = (
            100.0
            * np.sum(
                mask
                & lf_correct_only
            )
            / total
        )


        lf_only_values.append(
            value
        )


        #######################################################################
        # HF CORRECT ONLY
        #######################################################################

        value = (
            100.0
            * np.sum(
                mask
                & hf_correct_only
            )
            / total
        )


        hf_only_values.append(
            value
        )


        #######################################################################
        # BOTH CORRECT
        #######################################################################

        value = (
            100.0
            * np.sum(
                mask
                & both_correct
            )
            / total
        )


        both_correct_values.append(
            value
        )


        #######################################################################
        # BOTH WRONG
        #######################################################################

        value = (
            100.0
            * np.sum(
                mask
                & both_wrong
            )
            / total
        )


        both_wrong_values.append(
            value
        )


    ###########################################################################
    # NUMPY ARRAYS
    ###########################################################################

    lf_only_values = np.asarray(
        lf_only_values
    )


    hf_only_values = np.asarray(
        hf_only_values
    )


    both_correct_values = np.asarray(
        both_correct_values
    )


    both_wrong_values = np.asarray(
        both_wrong_values
    )


    ###########################################################################
    # FIGURE
    ###########################################################################

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )


    x = np.arange(
        len(categories)
    )


    width = 0.55


    ###########################################################################
    # STACK 1
    #
    # LF correct only
    ###########################################################################

    bars_lf = ax.bar(
        x,

        lf_only_values,

        width,

        label="LF correct only"
    )


    ###########################################################################
    # STACK 2
    #
    # HF correct only
    ###########################################################################

    bars_hf = ax.bar(
        x,

        hf_only_values,

        width,

        bottom=lf_only_values,

        label="HF correct only"
    )


    bottom_hf = (
        lf_only_values
        + hf_only_values
    )


    ###########################################################################
    # STACK 3
    #
    # Both correct
    ###########################################################################

    bars_both = ax.bar(
        x,

        both_correct_values,

        width,

        bottom=bottom_hf,

        label="Both correct"
    )


    bottom_both = (
        bottom_hf
        + both_correct_values
    )


    ###########################################################################
    # STACK 4
    #
    # Both wrong
    ###########################################################################

    bars_wrong = ax.bar(
        x,

        both_wrong_values,

        width,

        bottom=bottom_both,

        label="Both wrong"
    )


    ###########################################################################
    # ONLY LABEL BOTH WRONG
    #
    # This is deliberately the ONLY percentage printed on Plot 2.
    ###########################################################################

    for i in range(len(x)):

        value = both_wrong_values[i]


        if value <= 0:
            continue


        #######################################################################
        # RED SEGMENT BOTTOM
        #######################################################################

        red_bottom = bottom_both[i]


        #######################################################################
        # RED SEGMENT TOP
        #######################################################################

        red_top = (
            red_bottom
            + value
        )


        #######################################################################
        # LARGE RED SEGMENT
        #
        # Put percentage in the center.
        #######################################################################

        if value >= 3.0:

            y = (
                red_bottom
                + value / 2
            )


            text = ax.text(

                x[i],

                y,

                f"{value:.1f}%",

                ha="center",

                va="center",

                fontsize=11,

                fontweight="bold",

                color="black"
            )


        #######################################################################
        # SMALL RED SEGMENT
        #
        # Put the percentage immediately above the red segment.
        #
        # NO DASH.
        # NO ARROW.
        #######################################################################

        else:

            y = red_top + 1.0


            ###################################################################
            # Keep inside the plot if close to 100%.
            ###################################################################

            if y > 99:

                y = red_top - 1.0

                vertical_alignment = "top"

            else:

                vertical_alignment = "bottom"


            text = ax.text(

                x[i],

                y,

                f"{value:.1f}%",

                ha="center",

                va=vertical_alignment,

                fontsize=10,

                fontweight="bold",

                color="black"
            )


        #######################################################################
        # WHITE OUTLINE
        #######################################################################

        add_text_outline(
            text
        )


    ###########################################################################
    # FORMAT
    ###########################################################################

    ax.set_xticks(
        x
    )


    ax.set_xticklabels(
        categories,

        fontsize=11
    )


    ax.set_xlabel(
        "LF / HF prediction combination",

        fontsize=12
    )


    ax.set_ylabel(
        "Percentage of entire dataset (%)",

        fontsize=12
    )


    ax.set_title(
        f"{dataset}: LF-HF Prediction and Correctness",

        fontsize=16
    )


    ###########################################################################
    # Y LIMIT
    ###########################################################################

    ax.set_ylim(
        0,
        105
    )


    ###########################################################################
    # GRID
    ###########################################################################

    ax.grid(
        axis="y",

        alpha=0.20
    )


    ###########################################################################
    # SPINES
    ###########################################################################

    ax.spines["top"].set_visible(False)

    ax.spines["right"].set_visible(False)


    ###########################################################################
    # LEGEND
    #
    # Keep Plot 2 legend OUTSIDE the bars.
    #
    # This prevents it from covering the actual data.
    ###########################################################################

    ax.legend(
        frameon=False,

        loc="upper left",

        bbox_to_anchor=(1.02, 1.0),

        borderaxespad=0
    )


    ###########################################################################
    # RESERVE SPACE FOR OUTSIDE LEGEND
    ###########################################################################

    fig.subplots_adjust(
        right=0.78,

        bottom=0.16
    )


    ###########################################################################
    # SAVE
    ###########################################################################

    plot2_path = os.path.join(
        save_dir,

        f"{dataset}_prediction_correctness.png"
    )


    plt.savefig(
        plot2_path,

        dpi=300,

        bbox_inches="tight"
    )


    plt.close()


    print(
        f"Saved: {plot2_path}"
    )


###############################################################################
# DONE
###############################################################################

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)


print(
    "\nAll results saved to:\n"
    f"{os.path.abspath(save_dir)}"
)


print("\nFiles:")

print(
    "  agreement_disagreement.png"
)

print(
    "  agreement_disagreement.csv"
)

print(
    "  prediction_correctness.csv"
)

print(
    "  <dataset>_prediction_correctness.png"
)