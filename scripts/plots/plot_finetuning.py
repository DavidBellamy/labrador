from collections import OrderedDict
from pathlib import Path
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

results_dirs = [
    Path("data/results/drinks_per_day"),
    Path("data/results/cancer_diagnosis"),
    Path("data/results/covid_diagnosis"),
    Path("data/results/sepsis_mortality"),
]
time_string = time.strftime("%Y%m%d")

# Plot config
alcohol_methods = [
    "linear_regression",
    "random_forest",
    "xgboost",
    "bert68M",
    "labrador",
]

other_methods = [
    "logistic_regression",
    "random_forest",
    "xgboost",
    "bert68M",
    "labrador",
]

names_for_plot = [
    "Linear/Logistic Regression",
    "Random Forest",
    "XGBoost",
    "BERT",
    "Labrador",
]

plot_order = {
    "sepsis_mortality": "Sepsis Mortality Prediction",
    "cancer_diagnosis": "Cancer Diagnosis",
    "covid_diagnosis": "COVID-19 Diagnosis",
    "drinks_per_day": "Alcohol Consumption Prediction",
}

# Add custom y-axis limits for each subplot
custom_ylims = {
    "Sepsis Mortality Prediction": (0, 0.5),
    "Cancer Diagnosis": (0, 1.5),
    "COVID-19 Diagnosis": (0, 0.6),
    "Alcohol Consumption Prediction": (0, 15),
}


alcohol_names = dict(zip(alcohol_methods, names_for_plot))
other_names = dict(zip(other_methods, names_for_plot))

color_palette = {
    "Linear/Logistic Regression": (
        0.12156862745098039,
        0.4666666666666667,
        0.7058823529411765,
    ),
    "Random Forest": (1.0, 0.4980392156862745, 0.054901960784313725),
    "XGBoost": (0.17254901960784313, 0.6274509803921569, 0.17254901960784313),
    "BERT": "#F9D978",
    "Labrador": (0.8392156862745098, 0.15294117647058825, 0.1568627450980392),
}

# load and concatenate the results
df_list = dict()
for evaluation in results_dirs:
    all_files = evaluation.glob("*.csv")
    all_files = [f for f in all_files if "#" not in f.name]
    df_list[evaluation.name] = pd.concat(
        (pd.read_csv(f) for f in all_files), ignore_index=True
    )

# filter rows unless method in alcohol_methods or other_methods
for k, df in df_list.items():
    df_list[k] = df.query("method in @alcohol_methods or method in @other_methods")
    df_list[k] = df_list[k].dropna(subset=["method"])

    if "drinks" in k:
        df_list[k]["method"] = df_list[k]["method"].map(alcohol_names)
    else:
        df_list[k]["method"] = df_list[k]["method"].map(other_names)

fig, axes = plt.subplots(2, 2, figsize=(16, 16))
axes = axes.flatten()

# Iterate through the dataframes and create barplots
ordered_dict = OrderedDict((plot_order[key], df_list[key]) for key in plot_order)
int_to_letter = {0: "A", 1: "B", 2: "C", 3: "D"}
for i, (k, df) in enumerate(ordered_dict.items()):
    ax = axes[i]

    # Sort dataframe by method, so that it matches the order in the color palette
    df.sort_values(
        by="method",
        key=lambda column: column.map(lambda e: names_for_plot.index(e)),
        inplace=True,
    )

    if "alcohol" in k.lower():
        sns.barplot(data=df, x="method", y="test_mse", palette=color_palette, ax=ax)
        ax.set_ylabel("MSE", fontsize=18)
        ax.set_ylim(*custom_ylims[k])
    else:
        sns.barplot(data=df, x="method", y="test_ce", palette=color_palette, ax=ax)
        ax.set_ylabel("Cross Entropy", fontsize=18)
        ax.set_ylim(*custom_ylims[k])

    ax.set_xticklabels("")
    ax.tick_params(axis="y", which="major", labelsize=18)
    ax.set_xlabel(None)
    ax.set_title(f"{int_to_letter[i]}. {k}", loc="left", fontsize=18)
    ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
    ax.grid(axis="y", which="major", color="black", linewidth=0.01, alpha=0.5)
    ax.grid(axis="y", which="minor", color="black", linewidth=0.01, alpha=0.5)

lgd = fig.legend(
    title="Method",
    loc="upper center",
    ncol=2,
    fontsize=18,
    title_fontsize=18,
    bbox_to_anchor=(0.5, 0.2),
    handles=[
        mpl.patches.Patch(color=color_palette[method], label=method)
        for method in names_for_plot
    ],
)
plt.subplots_adjust(bottom=0.2)
plt.savefig(
    Path("figures/") / f"finetuning_{time_string}.png", dpi=300, bbox_inches="tight"
)
plt.savefig(
    Path("figures/") / f"finetuning_{time_string}.eps", dpi=300, bbox_inches="tight"
)
plt.close()
