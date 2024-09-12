import torch.nn as nn


class FullyConnectedModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512):
        super(FullyConnectedModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  # x needs to be long here
        x = x.mean(dim=1)  # Sum or average embeddings
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
