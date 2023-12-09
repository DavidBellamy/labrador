import os.path as op
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


labrador_data_file = "labrador_embeddings.csv"
bert_data_file = "bert68M_embeddings.csv"
labrador_embeddings = pd.read_csv(op.join("data/embeddings/", labrador_data_file))
bert_embeddings = pd.read_csv(op.join("data/embeddings/", bert_data_file))
labrador_codebook = pd.read_csv("data/processed/labcode_codebook_labrador.csv")
bert_codebook = pd.read_csv("data/processed/labcode_codebook_bert.csv")
lab_panel_df = pd.read_csv("data/processed/labcode_codebook_panelized.csv")
mimic_patient_info = pd.read_csv("data/raw/patients.csv")

labrador_embeddings = labrador_embeddings.merge(
    labrador_codebook, how="left", left_on="lab_code", right_on="frequency_rank"
).merge(lab_panel_df, how="left", on="itemid")
labrador_embeddings = labrador_embeddings.merge(
    mimic_patient_info, how="left", on="subject_id"
)

labrador_embeddings["anchor_year"] = pd.to_datetime(
    labrador_embeddings["anchor_year"], format="%Y"
)
labrador_embeddings["patient_age"] = (
    labrador_embeddings["anchor_age"]
    + (
        pd.to_datetime(labrador_embeddings["charttime"])
        - labrador_embeddings["anchor_year"]
    )
    / pd.to_timedelta(1, unit="D")
    / 365.25
)
labrador_embeddings["mimic_panel"] = labrador_embeddings["mimic_panel"].fillna("Other")
labrador_embeddings = labrador_embeddings[
    [
        "x",
        "y",
        "mimic_panel",
        "frequency_rank_x",
        "lab_value",
        "label_x",
        "fluid_x",
        "category_x",
        "loinc_code_x",
        "itemid",
        "subject_id",
        "hadm_id",
        "charttime",
        "patient_age",
    ]
]
labrador_embeddings = labrador_embeddings[
    ~(
        (labrador_embeddings["mimic_panel"] == "Bilirubin")
        & (labrador_embeddings["patient_age"] > 1)
    )
]
labrador_embeddings["mimic_panel"] = labrador_embeddings["mimic_panel"].replace(
    "Bilirubin", "Bilirubin (neonates)"
)

bert_embeddings = bert_embeddings.merge(bert_codebook, how="left", on="token").merge(
    lab_panel_df, how="left", on="itemid"
)
bert_embeddings = bert_embeddings.merge(mimic_patient_info, how="left", on="subject_id")

bert_embeddings["anchor_year"] = pd.to_datetime(
    bert_embeddings["anchor_year"], format="%Y"
)
bert_embeddings["patient_age"] = (
    bert_embeddings["anchor_age"]
    + (pd.to_datetime(bert_embeddings["charttime"]) - bert_embeddings["anchor_year"])
    / pd.to_timedelta(1, unit="D")
    / 365.25
)

# Fill NaN's in column "mimic_panel" with "Other"
bert_embeddings["mimic_panel"] = bert_embeddings["mimic_panel"].fillna("Other")

# Reorder columns
bert_embeddings = bert_embeddings[
    [
        "x",
        "y",
        "mimic_panel",
        "frequency_rank_x",
        "valuenum",
        "label_x",
        "fluid_x",
        "category_x",
        "loinc_code_x",
        "itemid",
        "subject_id",
        "hadm_id",
        "charttime",
        "patient_age",
    ]
]

# Filter out rows with mimic_panel = Bilirubin and age > 1
bert_embeddings = bert_embeddings[
    ~(
        (bert_embeddings["mimic_panel"] == "Bilirubin")
        & (bert_embeddings["patient_age"] > 1)
    )
]
bert_embeddings["mimic_panel"] = bert_embeddings["mimic_panel"].replace(
    "Bilirubin", "Bilirubin (neonates)"
)

# Set the color palette for panel A
panel_list = [
    "CBC",
    "BMP",
    "Blood Gases",
    "Bilirubin (neonates)",
    "CBC differential",
    "Toxicology (blood)",
    "Urinalysis",
    "Liver enzymes",
    "Lipid panel",
    "Toxicology (urine)",
]

palette = {panel: color for panel, color in zip(panel_list, sns.color_palette())}

fig = plt.figure(figsize=(16, 16))
plt.rcParams.update({"font.size": 24})
plt.subplots_adjust(hspace=0.3)
subfigs = fig.subfigures(2, 1, height_ratios=[3, 2])
subfigs[0].suptitle("A", x=0.05, y=0.9, fontsize=36)

axes = subfigs[0].subplots(1, 2)
subfigs[0].subplots_adjust(bottom=0.35)
subfigs[1].subplots_adjust(bottom=0.1)

scatter = sns.scatterplot(
    data=labrador_embeddings[labrador_embeddings.mimic_panel != "Other"],
    x="x",
    y="y",
    size=1,
    hue="mimic_panel",
    alpha=0.75,
    ax=axes[0],
    palette=palette,
)
axes[0].title.set_text("Labrador")
axes[0].set_xlabel("UMAP dimension 1")
axes[0].set_ylabel("UMAP dimension 2")
axes[0].set_xticks([])
axes[0].set_yticks([])
handles, labels = axes[0].get_legend_handles_labels()

axes[0].legend(
    handles=handles[:-1],
    labels=labels[:-1],
    bbox_to_anchor=(0.1, -0.1),
    loc="upper left",
    ncol=2,
    markerscale=2.5,
)

sns.scatterplot(
    data=bert_embeddings[bert_embeddings.mimic_panel != "Other"],
    x="x",
    y="y",
    size=1,
    hue="mimic_panel",
    alpha=0.75,
    ax=axes[1],
    palette=palette,
)
axes[1].title.set_text("BERT")
axes[1].set_xlabel("UMAP dimension 1")
axes[1].set_ylabel("UMAP dimension 2")
axes[1].set_xticks([])
axes[1].set_yticks([])
axes[1].legend().remove()

quantitative_plots = subfigs[1].subfigures(1, 2)
quantitative_plots[0].suptitle("B", x=0.1, y=1, fontsize=36)
labrador_quantitative_plot_axes = quantitative_plots[0].subplots(2, 2)
bert_quantitative_plot_axes = quantitative_plots[1].subplots(2, 2)
quantitative_plots[1].subplots_adjust(right=0.8)
colorbar_ax = quantitative_plots[1].add_axes([0.85, 0.15, 0.025, 0.7])

lab_panel_df.sort_values(by="frequency_rank", inplace=True)
itemid_list = lab_panel_df["itemid"][:4].to_list()

for subfigure, df, value_column in zip(
    [labrador_quantitative_plot_axes, bert_quantitative_plot_axes],
    [labrador_embeddings, bert_embeddings],
    ["lab_value", "valuenum"],
):
    for ax, itemid in zip(subfigure.flatten(), itemid_list):
        temp_df = df[df[value_column] != 531]
        norm = plt.Normalize(
            temp_df[temp_df["itemid"] == itemid][value_column].min(),
            temp_df[temp_df["itemid"] == itemid][value_column].max(),
        )
        sm = plt.cm.ScalarMappable(cmap="plasma_r", norm=norm)
        sm.set_array([])
        scatter = sns.scatterplot(
            data=temp_df[temp_df["itemid"] == itemid],
            x="x",
            y="y",
            hue=value_column,
            size=1,
            alpha=0.75,
            palette="plasma_r",
            ax=ax,
        )
        ax.set_title(lab_panel_df[lab_panel_df["itemid"] == itemid].label.item())
        ax.get_legend().remove()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel("")
        ax.set_xlabel("")

quantitative_plots[1].colorbar(sm, cax=colorbar_ax)

time_string = time.strftime("%Y%m%d-%H%M%S")
plt.savefig(f"figures/umap_figure_{time_string}.png", dpi=300, bbox_inches="tight")
