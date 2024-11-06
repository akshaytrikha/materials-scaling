# fcn_train_utils.py

import torch.optim as optim
from loss import compute_mse_loss

def get_optimizer(model, learning_rate=1e-3):
    return optim.Adam(model.parameters(), lr=learning_rate)

def compute_loss(pred_forces, pred_energy, pred_stress, true_forces, true_energy, true_stress, mask):
    return compute_mse_loss(
        pred_forces, pred_energy, pred_stress,
        true_forces, true_energy, true_stress, mask
    )
