# models.py

import torch
import torch.nn as nn
import math
from typing import Tuple

class FullyConnectedModel(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 512, hidden_dim: int = 512):
        """
        Initializes the Fully Connected Network model.

        Args:
            vocab_size (int): Size of the vocabulary.
            embedding_dim (int, optional): Dimension of the embedding vectors. Defaults to 512.
            hidden_dim (int, optional): Dimension of the hidden layer. Defaults to 512.
        """
        super(FullyConnectedModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, vocab_size)

        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the FCN model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape [batch_size, seq_len].
            labels (torch.Tensor, optional): Labels tensor of shape [batch_size, seq_len]. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - If labels are provided: (loss, logits)
                - Else: logits tensor of shape [batch_size, seq_len, vocab_size]
        """
        # Embed the input_ids: [batch_size, seq_len, embedding_dim]
        x = self.embedding(input_ids)

        # Pass through the first fully connected layer: [batch_size, seq_len, hidden_dim]
        x = self.fc1(x)
        x = self.relu(x)

        # Pass through the second fully connected layer: [batch_size, seq_len, vocab_size]
        logits = self.fc2(x)

        if labels is not None:
            # # Debugging: Print shapes before reshaping
            # print(f"Logits shape before reshaping: {logits.shape}")    # Expected: [batch_size, seq_len, vocab_size]
            # print(f"Labels shape before reshaping: {labels.shape}")    # Expected: [batch_size, seq_len]

            # Reshape logits and labels for loss computation
            # Logits: [batch_size * seq_len, vocab_size]
            # Labels: [batch_size * seq_len]
            logits_reshaped = logits.reshape(-1, logits.size(-1))
            labels_reshaped = labels.reshape(-1)

            # Initialize the loss function
            loss_fn = nn.CrossEntropyLoss()

            # Compute the loss
            loss = loss_fn(logits_reshaped, labels_reshaped)

            return loss, logits
        else:
            return logits

class MetaFullyConnectedModels:
    def __init__(self, vocab_size: int):
        """
        Initializes the meta class for generating multiple FCN configurations.

        Args:
            vocab_size (int): Size of the vocabulary.
        """
        # Parameter Scaling Constants
        self.embedding_dims = [16, 32, 64, 128, 256, 256, 256]
        self.hidden_dims = [16, 32, 64, 128, 256, 512, 1024]
        self.vocab_size = vocab_size

        # Generate all combinations of embedding_dims and hidden_dims
        self.configurations = list(
            zip(
                self.embedding_dims,
                self.hidden_dims,
            )
        )

    def __iter__(self):
        """
        Yields FullyConnectedModel instances for each configuration.
        """
        for emb_dim, hid_dim in self.configurations:
            yield FullyConnectedModel(
                vocab_size=self.vocab_size,
                embedding_dim=emb_dim,
                hidden_dim=hid_dim
            )

    def __len__(self):
        """
        Returns the number of configurations.

        Returns:
            int: Number of model configurations.
        """
        return len(self.configurations)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # [d_model/2]
        pe = torch.zeros(max_len, 1, d_model)  # [max_len, 1, d_model]
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.2):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        self.embedding = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

        self.num_params = sum(p.numel() for p in self.parameters())

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            input_ids: Tensor, shape [batch_size, seq_len]
            labels: Tensor, shape [batch_size, seq_len] (optional)
            src_mask: Tensor, shape [seq_len, seq_len] (optional)

        Returns:
            If labels are provided:
                Tuple containing (loss, logits)
            Else:
                Logits tensor of shape [batch_size, seq_len, ntoken]
        """
        # Transpose to [seq_len, batch_size]
        input_ids = input_ids.transpose(0, 1)  # [seq_len, batch_size]
        src = self.embedding(input_ids) * math.sqrt(self.d_model)  # [seq_len, batch_size, d_model]
        src = self.pos_encoder(src)  # [seq_len, batch_size, d_model]

        output = self.transformer_encoder(src, src_mask)  # [seq_len, batch_size, d_model]
        output = self.linear(output)  # [seq_len, batch_size, ntoken]
        output = output.transpose(0, 1)  # [batch_size, seq_len, ntoken]

        if labels is not None:
            # # Debugging: Print shapes before reshaping
            # print(f"Output shape before reshaping: {output.shape}")  # Should be [batch_size, seq_len, ntoken]
            # print(f"Labels shape before reshaping: {labels.shape}")  # Should be [batch_size, seq_len]

            # Use .reshape() instead of .view()
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(output.reshape(-1, output.size(-1)), labels.reshape(-1))  # [batch_size*seq_len, ntoken], [batch_size*seq_len]

            return loss, output
        else:
            return output


class MetaVanillaTransformers:
    def __init__(self, vocab_size, d_model: int = 64, d_hid: int = 128, nhead: int = 2, nlayers: int = 2, dropout: float = 0.2):
        # You can modify these default values or make them configurable via arguments
        self.d_models = [d_model]  # Single configuration or multiple as needed
        self.d_hids = [d_hid]
        self.nheads = [nhead]
        self.nlayers = [nlayers]
        self.dropout = dropout

        # Generate all combinations (you can limit this to avoid combinatorial explosion)
        self.configurations = []
        for d_model in self.d_models:
            for d_hid in self.d_hids:
                for nhead in self.nheads:
                    for nlayers in self.nlayers:
                        if d_model % nhead == 0:  # Ensure d_model is divisible by nhead
                            self.configurations.append((d_model, d_hid, nhead, nlayers))

        self.vocab_size = vocab_size

    def __iter__(self):
        for d_model, d_hid, nhead, nlayers in self.configurations:
            yield TransformerModel(
                ntoken=self.vocab_size,
                d_model=d_model,
                nhead=nhead,
                d_hid=d_hid,
                nlayers=nlayers,
                dropout=self.dropout
            )

    def __len__(self):
        return len(self.configurations)

