sweep_config = {
    "method": "grid",
    "name": "transformer-10k",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"values": [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]},
        "batch_size": {"values": [8, 16, 32, 64, 128, 1024, 2048]},
        "architecture": {"value": "Transformer"},
        "epochs": {"value": 30},
        "dataset": {"value": "all"},  # wandb needs a "dataset" key
        "datasets": {"value": "all"},
        "split_name": {"value": "val"},
        "data_fractions": {"value": 0.01},
        "val_data_fraction": {"value": 0.01},
        "vis_every": {"value": 100},
        "val_every": {"value": 5},
        "train_workers": {"value": 6},
        "val_workers": {"value": 6},
        "mixed_precision": {"value": True},
        "no_log": {"value": True},
    },
}
