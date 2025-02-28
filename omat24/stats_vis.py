import json
import numpy as np
import matplotlib.pyplot as plt

with open("dataset_stats.json", "r") as f:
    data = json.load(f)

splits = ["val"]
force_labels = ["Fx", "Fy", "Fz"]

for split in splits:
    if not data.get(split):
        print(f"No data for {split}")
        continue

    # Separate sub-datasets by name
    subdatasets = list(data[split].keys())
    rattled_ds = [ds for ds in subdatasets if "rattled" in ds]
    aimd_ds = [ds for ds in subdatasets if "aimd" in ds]

    # Plot "rattled" sub-datasets
    if rattled_ds:
        rattled_forces = np.array(
            [data[split][ds]["means"]["forces"] for ds in rattled_ds]
        )
        rattled_forces_std = np.array(
            [data[split][ds]["std"]["forces"] for ds in rattled_ds]
        )

        x = np.arange(len(rattled_ds))
        width = 0.25

        plt.figure(figsize=(8, 6))
        for i in range(3):
            plt.bar(
                x + (i - 1) * width,
                rattled_forces[:, i],
                width,
                label=force_labels[i],
                yerr=rattled_forces_std[:, i],
                capsize=5,
            )

        plt.xticks(x, rattled_ds, rotation=45, ha="right")
        plt.ylabel("Force (per atom)")
        plt.title(f"{split.capitalize()} Force Components (Rattled Only)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"stats/{split}_forces_rattled.png")
        plt.close()

    # Plot "aimd" sub-datasets
    if aimd_ds:
        aimd_forces = np.array([data[split][ds]["means"]["forces"] for ds in aimd_ds])
        aimd_forces_std = np.array([data[split][ds]["std"]["forces"] for ds in aimd_ds])

        x = np.arange(len(aimd_ds))
        width = 0.25

        plt.figure(figsize=(8, 6))
        for i in range(3):
            plt.bar(
                x + (i - 1) * width,
                aimd_forces[:, i],
                width,
                label=force_labels[i],
                yerr=aimd_forces_std[:, i],
                capsize=5,
            )

        plt.xticks(x, aimd_ds, rotation=45, ha="right")
        plt.ylabel("Force (per atom)")
        plt.title(f"{split.capitalize()} Force Components (AIMD Only)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"stats/{split}_forces_aimd.png")
        plt.close()
