sweep_config = {
    "method": "grid",
    "name": "ADiT-df=0.01",
    "metric": {"name": "val_loss", "goal": "minimize"},
    "parameters": {
        "learning_rate": {"values": [1e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]},
        "batch_size": {"values": [8, 16, 32, 64, 128]},
        "architecture": {"value": "ADiT"},
        "epochs": {"value": 500},
        "dataset": {"value": "all"},  # wandb needs a "dataset" key
        "datasets": {"value": "all"},
        "split_name": {"value": "val"},
        "data_fractions": {"value": 0.01},
        "val_data_fraction": {"value": 0.01},
        "vis_every": {"value": 1000},
        "val_every": {"value": 5},
        "train_workers": {"value": 0},
        "val_workers": {"value": 0},
        "mixed_precision": {"value": True},
        "no_log": {"value": True},
    },
}
