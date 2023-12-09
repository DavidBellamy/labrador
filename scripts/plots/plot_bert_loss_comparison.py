import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd


def main(
    smallbert_train_loss: str,
    smallbert_val_loss: str,
    bigbert_train_loss: str,
    bigbert_val_loss: str,
    bert_out_filepaths: str,
) -> None:
    # Load data from CSV files
    small_train_loss = pd.read_csv(smallbert_train_loss)
    small_val_loss = pd.read_csv(smallbert_val_loss)
    big_train_loss = pd.read_csv(bigbert_train_loss)
    big_val_loss = pd.read_csv(bigbert_val_loss)

    # Create a 1x2 subplot layout
    fig, axes = plt.subplots(2, 1, figsize=(16, 16))
    font_size = 20
    legend_font_size = 20  # Adjust the font size of the legends

    # Sci notation function
    def sci_notation(x, pos):
        return f"{x:.1e}"

    # Plotting
    axes[0].plot(small_train_loss["step"], small_train_loss["loss"], label="BERT 68.5M")
    axes[0].plot(big_train_loss["step"], big_train_loss["loss"], label="BERT 194M")
    axes[1].plot(small_val_loss["step"], small_val_loss["loss"], label="BERT 68.5M")
    axes[1].plot(big_val_loss["step"], big_val_loss["loss"], label="BERT 194M")

    # Labels and titles
    for ax in axes.flatten():
        ax.set_xlabel("Step", fontsize=font_size)
        ax.set_ylabel("Cross-entropy", fontsize=font_size)
        ax.tick_params(axis="both", labelsize=font_size)
        ax.xaxis.set_major_formatter(FuncFormatter(sci_notation))
        ax.grid(True)

    axes[0].set_title("Training loss", fontsize=font_size)
    axes[1].set_title("Validation loss", fontsize=font_size)

    # Legends
    axes[0].legend(fontsize=legend_font_size)
    axes[1].legend(fontsize=legend_font_size)

    # Display
    plt.tight_layout()
    for path in bert_out_filepaths:
        plt.savefig(path, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    smallbert_train_loss = "data/results/pretraining_loss/smallbert_trainloss_1.5M.csv"
    smallbert_val_loss = "data/results/pretraining_loss/smallbert_valloss_1.5M.csv"
    bigbert_train_loss = "data/results/pretraining_loss/bigbert_trainloss.csv"
    bigbert_val_loss = "data/results/pretraining_loss/bigbert_valloss.csv"
    bert_out_filepaths = [
        "figures/bert_loss_comparison.png",
        "figures/bert_loss_comparison.eps",
    ]

    main(
        smallbert_train_loss,
        smallbert_val_loss,
        bigbert_train_loss,
        bigbert_val_loss,
        bert_out_filepaths,
    )
