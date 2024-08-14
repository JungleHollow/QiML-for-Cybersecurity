import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from glob import glob

all_files = glob("*.csv", root_dir=r"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\models_binary\histories")

for file in all_files:
    filename = file.split(".csv")[0]
    df = pd.read_csv(rf"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\models_binary\histories\{file}")

    loss, val_loss = df["loss"], df["val_loss"]
    acc, val_acc = df["accuracy"], df["val_accuracy"]

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(df.index, loss, label="Training Loss")
    ax.plot(df.index, val_loss, label="Validation Loss")

    plt.tight_layout()
    plt.legend()
    plt.title(f"Loss History for {filename}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Function")
    plt.savefig(rf"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\history_graphs\entanglement_experiment\{filename}_loss.png", dpi=300, bbox_inches="tight")

    del fig, ax

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(df.index, acc, label="Training Accuracy")
    ax.plot(df.index, val_acc, label="Validation Accuracy")

    plt.tight_layout()
    plt.legend()
    plt.title(f"Accuracy History for {filename}")
    plt.xlabel("Epochs")
    plt.ylabel("Classification Accuracy")
    plt.savefig(rf"C:\Users\WornA\OneDrive\Desktop\Uni Work\Honours Thesis\Code\history_graphs\entanglement_experiment\{filename}_accuracy.png", dpi=300, bbox_inches="tight")
