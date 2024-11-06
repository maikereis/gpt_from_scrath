import matplotlib.pyplot as plt
import seaborn as sns

def plot_values(epochs_seen, examples_seen, train_values, val_values, label='loss'):
    fig, ax1 = plt.subplots(figsize=(5,3))

    sns.lineplot(x=epochs_seen, y=train_values, label=f"Training {label}", ax=ax1)
    sns.lineplot(x=epochs_seen, y=val_values, label=f"Validation {label}", linestyle="-.", ax=ax1)

    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    sns.lineplot(x=examples_seen, y=train_values, alpha=0, ax=ax2)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.show()
