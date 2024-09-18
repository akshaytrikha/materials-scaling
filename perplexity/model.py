import torch
import torch.nn as nn


class MetaFullyConnectedModels:
    def __init__(self, vocab_size):
        # Parameter Scaling Constants
        self.embedding_dims = [128, 256, 512, 1024]
        self.hidden_dims = [128, 256, 512, 1024]
        self.vocab_size = vocab_size

        # Generate all combinations of embedding_dims and hidden_dims
        self.configurations = list(
            zip(
                self.embedding_dims,
                self.hidden_dims,
            )
        )

    def __iter__(self):
        for emb_dim, hid_dim in self.configurations:
            yield FullyConnectedModel(
                self.vocab_size, embedding_dim=emb_dim, hidden_dim=hid_dim
            )

    def __len__(self):
        return len(self.configurations)


class FullyConnectedModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=512, hidden_dim=512):
        super(FullyConnectedModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x):
        x = self.embedding(x)  # x needs to be long here
        x = x.mean(dim=1)  # Sum or average embeddings
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class VanillaTransformer(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model=64,
        nhead=2,
        num_encoder_layers=2,
        dim_feedforward=128,
        dropout=0.1,
    ):
        super(VanillaTransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(
            torch.zeros(1, 1000, d_model)
        )  # Simple positional encoding

        # Use Transformer Encoder instead of full Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.fc_out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, src, src_mask=None):
        # src shape: (batch_size, seq_len)
        src = self.embedding(src) * (
            self.d_model**0.5
        )  # Embed and scale by sqrt(d_model)
        src = src + self.pos_encoder[:, : src.size(1), :]  # Add positional encoding

        # src shape: (seq_len, batch_size, d_model)
        src = src.transpose(0, 1)

        # Transformer Encoder (no tgt needed)
        output = self.transformer_encoder(src, src_mask)
        output = self.fc_out(output)  # Project back to vocabulary size

        # output shape: (seq_len, batch_size, vocab_size)
        return output.transpose(
            0, 1
        )  # Transpose back to (batch_size, seq_len, vocab_size)
