import pandas as pd
import wandb
import matplotlib.pyplot as plt

api = wandb.Api()
runs = api.runs("material-scaling/wikitext-scaling")

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains output keys/values for
    # metrics such as accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

runs_df = pd.DataFrame(
    {"summary": summary_list, "config": config_list, "name": name_list}
)

# print(summary_list[0])
# # {'_runtime': 4.2593889236450195, '_step': 5, '_timestamp': 1725991199.82478, '_wandb': {'runtime': 4}, 'loss': 3.1876030762990317, 'perplexity': 21.68320655822754}
# print(config_list[0])
# # {'fraction': '1%', 'batch_size': 64, 'epochs': 5, 'learning_rate': 0.001}


# Extract perplexity and run names
perplexity_values = [summary.get("perplexity", None) for summary in summary_list]
run_names = name_list

# Separate the data based on run names
wikitext_2_v1_perplexity = [
    summary.get("perplexity", None)
    for summary, name in zip(summary_list, run_names)
    if name.startswith("wikitext-2-v1")
]
wikitext_2_v1_names = [name for name in run_names if name.startswith("wikitext-2-v1")]

wikitext_103_v1_perplexity = [
    summary.get("perplexity", None)
    for summary, name in zip(summary_list, run_names)
    if name.startswith("wikitext-103-v1")
]
wikitext_103_v1_names = [
    name for name in run_names if name.startswith("wikitext-103-v1")
]

# Extract the percentage part of the run names and convert to numbers
wikitext_2_v1_labels = [
    int(name.split("_")[-1].replace("%", "")) for name in wikitext_2_v1_names
]
wikitext_103_v1_labels = [
    int(name.split("_")[-1].replace("%", "")) for name in wikitext_103_v1_names
]

# Create a figure
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the line chart for wikitext-2-v1
ax.plot(
    wikitext_2_v1_labels,
    wikitext_2_v1_perplexity,
    marker="o",
    linestyle="-",
    color="b",
    label="wikitext-2-v1",
)

# Plot the line chart for wikitext-103-v1
ax.plot(
    wikitext_103_v1_labels,
    wikitext_103_v1_perplexity,
    marker="o",
    linestyle="-",
    color="g",
    label="wikitext-103-v1",
)

# Set labels and title
ax.set_xlabel("Percentage")
ax.set_ylabel("Perplexity")
ax.set_title("Perplexity vs Percentage")
ax.tick_params(axis="x", rotation=45)

# Add legend
ax.legend()

plt.tight_layout()

# Save the plot as an image file
plt.savefig("perplexity_vs_percentage.png")


def create_log_log_perplexity(entity_name, project_name, batch_name):
    api = wandb.Api()
    group_filter = {"group": batch_name}
    runs = api.runs(f"{entity_name}/{project_name}", filters=group_filter)
    data_proportions = []
    train_perplexities = []
    validation_perplexities = []
    for run in runs:
        data_proportion = float(run.config.get("fraction"))
        train_perplexity = float(run.config.get("train_perplexity"))
        validation_perplexity = float(run.config.get("validation_perplexity"))
        data_proportions.append(data_proportion)
        train_perplexities.append(train_perplexity)
        validation_perplexities.append(validation_perplexity)
    plt.figure(figsize=(8, 6))
    plt.loglog(
        data_proportions, train_perplexities, marker="o", linestyle="-", color="blue"
    )
    plt.loglog(
        data_proportions,
        validation_perplexities,
        marker="o",
        linestyle="-",
        color="green",
    )
    plt.legend()
    plt.xlabel("Data Set Size")
    plt.ylabel("Validation Loss")
    plt.title(batch_name)
    plt.grid(True, which="both", ls="--")
    plt.savefig(batch_name)


create_log_log_perplexity("material-scaling", "wikitext-scaling", "")
