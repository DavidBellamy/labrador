import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


def main(
    labrador_train_cat_loss: str,
    labrador_train_cont_loss: str,
    labrador_val_cat_loss: str,
    labrador_val_cont_loss: str,
    labrador_out_filepaths: str,
    bert_train_loss: str,
    bert_val_loss: str,
    bert_out_filepaths: str,
) -> None:
    # Plot Labrador pretraining loss

    # Read the data from CSV files
    train_loss_lab_codes = pd.read_csv(labrador_train_cat_loss)
    train_loss_lab_values = pd.read_csv(labrador_train_cont_loss)
    val_loss_lab_codes = pd.read_csv(labrador_val_cat_loss)
    val_loss_lab_values = pd.read_csv(labrador_val_cont_loss)

    # Create a 2x2 subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    font_size = 20

    # Plot the data in each panel
    axes[0, 0].plot(
        train_loss_lab_codes["step"],
        train_loss_lab_codes["loss"],
        label="Training loss (lab codes)",
    )
    axes[0, 1].plot(
        train_loss_lab_values["step"],
        train_loss_lab_values["loss"],
        label="Training loss (lab values)",
    )
    axes[1, 0].plot(
        val_loss_lab_codes["step"],
        val_loss_lab_codes["loss"],
        label="Validation loss (lab codes)",
    )
    axes[1, 1].plot(
        val_loss_lab_values["step"],
        val_loss_lab_values["loss"],
        label="Validation loss (lab values)",
    )

    # Set labels and titles for each panel
    axes[0, 0].set_xlabel("Step")
    axes[0, 0].set_ylabel("Cross-entropy")
    axes[0, 0].set_title("Training loss (lab codes)")

    axes[0, 1].set_xlabel("Step")
    axes[0, 1].set_ylabel("MSE")
    axes[0, 1].set_title("Training loss (lab values)")

    axes[1, 0].set_xlabel("Step")
    axes[1, 0].set_ylabel("Cross-entropy")
    axes[1, 0].set_title("Validation loss (lab codes)")

    axes[1, 1].set_xlabel("Step")
    axes[1, 1].set_ylabel("MSE")
    axes[1, 1].set_title("Validation loss (lab values)")

    def sci_notation(x, pos):
        return f"{x:.1e}"

    for ax in axes.flatten():
        ax.tick_params(axis="both", labelsize=font_size)
        ax.title.set_text(ax.title.get_text())
        ax.title.set_fontsize(font_size)
        ax.set_xlabel(ax.get_xlabel(), fontsize=font_size)
        ax.set_ylabel(ax.get_ylabel(), fontsize=font_size)
        ax.set_ylim(bottom=0)
        formatter = FuncFormatter(sci_notation)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(True)

    # Adjust the layout and display the plot
    plt.tight_layout()
    for path in labrador_out_filepaths:
        plt.savefig(path, dpi=300, bbox_inches="tight")

    # Plot Bert pretraining loss
    train_loss = pd.read_csv(bert_train_loss)
    val_loss = pd.read_csv(bert_val_loss)

    # Create a 1x2 subplot layout
    fig, axes = plt.subplots(2, 1, figsize=(16, 16))
    font_size = 20

    # Plot the data in each panel
    axes[0].plot(train_loss["step"], train_loss["loss"], label="Training loss")
    axes[1].plot(val_loss["step"], val_loss["loss"], label="Validation loss")

    # Set labels and titles for each panel
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].set_title("Training loss")

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Cross-entropy")
    axes[1].set_title("Validation loss")

    def sci_notation(x, pos):
        return f"{x:.1e}"

    for ax in axes.flatten():
        ax.tick_params(axis="both", labelsize=font_size)
        ax.title.set_text(ax.title.get_text())
        ax.title.set_fontsize(font_size)
        ax.set_xlabel(ax.get_xlabel(), fontsize=font_size)
        ax.set_ylabel(ax.get_ylabel(), fontsize=font_size)
        ax.set_ylim(bottom=0)
        formatter = FuncFormatter(sci_notation)
        ax.xaxis.set_major_formatter(formatter)
        ax.grid(True)

    # Adjust the layout and display the plot
    plt.tight_layout()
    for path in bert_out_filepaths:
        plt.savefig(path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    # Labrador pre-training loss filepaths
    labrador_train_cat_loss = (
        "data/results/pretraining_loss/labrador_train_loss_lab_codes.csv"
    )
    labrador_train_cont_loss = (
        "data/results/pretraining_loss/labrador_train_loss_lab_values.csv"
    )
    labrador_val_cat_loss = (
        "data/results/pretraining_loss/labrador_val_loss_lab_codes.csv"
    )
    labrador_val_cont_loss = (
        "data/results/pretraining_loss/labrador_val_loss_lab_values.csv"
    )
    labrador_out_filepath = [
        "figures/labrador_pretraining_loss.png",
        "figures/labrador_pretraining_loss.eps",
    ]

    # Bert pre-training loss filepaths
    bert_train_loss = "data/results/pretraining_loss/smallbert_trainloss_1.5M.csv"
    bert_val_loss = "data/results/pretraining_loss/smallbert_valloss_1.5M.csv"
    bert_out_filepath = [
        "figures/bert_pretraining_loss_1.5M.png",
        "figures/bert_pretraining_loss_1.5M.eps",
    ]

    main(
        labrador_train_cat_loss,
        labrador_train_cont_loss,
        labrador_val_cat_loss,
        labrador_val_cont_loss,
        labrador_out_filepath,
        bert_train_loss,
        bert_val_loss,
        bert_out_filepath,
    )
