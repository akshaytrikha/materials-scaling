import torch

# Hyperparameters
BATCH_SIZE = 64
NUM_EPOCHS = 5
LR = 0.001
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")
