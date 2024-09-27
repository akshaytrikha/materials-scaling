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
        """
        Implements the sinusoidal positional encoding.

        Args:
            d_model (int): Embedding dimension.
            dropout (float): Dropout rate.
            max_len (int): Maximum length of the input sequences.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encodings once in log space
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]

        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices

        pe = pe.unsqueeze(1)  # [max_len, 1, d_model]
        self.register_buffer('pe', pe)  # Not a parameter, but should be saved with the model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [seq_len, batch_size, embedding_dim]

        Returns:
            torch.Tensor: Positional encoded tensor of the same shape as input.
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class DecoderOnlyLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: str = "relu"):
        """
        Initializes a single decoder-only transformer layer.

        Args:
            d_model (int): Embedding dimension.
            nhead (int): Number of attention heads.
            dim_feedforward (int, optional): Dimension of the feedforward network. Defaults to 2048.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
            activation (str, optional): Activation function. Defaults to "relu".
        """
        super(DecoderOnlyLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation == "relu":
            self.activation = nn.functional.relu
        elif activation == "gelu":
            self.activation = nn.functional.gelu
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, tgt: torch.Tensor, tgt_mask: torch.Tensor = None, tgt_key_padding_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the decoder-only layer.

        Args:
            tgt (torch.Tensor): Target tensor of shape [seq_len, batch_size, d_model].
            tgt_mask (torch.Tensor, optional): Mask for the target sequence. Defaults to None.
            tgt_key_padding_mask (torch.Tensor, optional): Mask for the target keys per batch. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape [seq_len, batch_size, d_model].
        """
        # Self-attention
        attn_output, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(attn_output)
        tgt = self.norm1(tgt)

        # Feedforward
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(ff_output)
        tgt = self.norm2(tgt)

        return tgt


class TransformerModel(nn.Module):
    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.2, max_len: int = 5000):
        """
        Initializes the decoder-only Transformer model.

        Args:
            ntoken (int): Size of the vocabulary.
            d_model (int): Embedding dimension.
            nhead (int): Number of attention heads.
            d_hid (int): Dimension of the feedforward network.
            nlayers (int): Number of Transformer decoder layers.
            dropout (float): Dropout rate.
            max_len (int): Maximum length of the input sequences.
        """
        super(TransformerModel, self).__init__()
        self.model_type = 'Decoder-Only Transformer'
        self.d_model = d_model

        self.embedding = nn.Embedding(ntoken, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)

        self.layers = nn.ModuleList([DecoderOnlyLayer(d_model, nhead, d_hid, dropout) for _ in range(nlayers)])

        self.linear = nn.Linear(d_model, ntoken)

        self.init_weights()

        self.num_params = sum(p.numel() for p in self.parameters())

    def init_weights(self) -> None:
        """
        Initializes weights of the model.
        """
        initrange = 0.1
        nn.init.uniform_(self.embedding.weight, -initrange, initrange)
        nn.init.zeros_(self.linear.bias)
        nn.init.uniform_(self.linear.weight, -initrange, initrange)

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """
        Generates a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).

        Args:
            sz (int): Size of the mask (sequence length).

        Returns:
            torch.Tensor: Mask tensor of shape [sz, sz].
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Transformer model.

        Args:
            input_ids (torch.Tensor): Input tensor of shape [batch_size, seq_len].
            labels (torch.Tensor, optional): Labels tensor of shape [batch_size, seq_len]. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - If labels are provided: (loss, logits)
                - Else: logits tensor of shape [batch_size, seq_len, ntoken]
        """
        # Transpose to [seq_len, batch_size]
        input_ids = input_ids.transpose(0, 1)  # [seq_len, batch_size]

        # Embed the input tokens
        embedded = self.embedding(input_ids) * math.sqrt(self.d_model)  # [seq_len, batch_size, d_model]
        embedded = self.pos_encoder(embedded)  # [seq_len, batch_size, d_model]

        # Generate causal mask
        seq_len = embedded.size(0)
        src_mask = self.generate_square_subsequent_mask(seq_len).to(embedded.device)  # [seq_len, seq_len]

        # Pass through the decoder layers
        for layer in self.layers:
            embedded = layer(embedded, tgt_mask=src_mask)

        # Project to vocabulary size
        logits = self.linear(embedded)  # [seq_len, batch_size, ntoken]

        # Transpose back to [batch_size, seq_len, ntoken]
        logits = logits.transpose(0, 1)  # [batch_size, seq_len, ntoken]

        if labels is not None:
            # Debugging: Print shapes before reshaping
            print(f"Logits shape before reshaping: {logits.shape}")    # Expected: [batch_size, seq_len, ntoken]
            print(f"Labels shape before reshaping: {labels.shape}")    # Expected: [batch_size, seq_len]

            # Reshape logits and labels for loss computation
            logits_reshaped = logits.reshape(-1, logits.size(-1))  # [batch_size * seq_len, ntoken]
            labels_reshaped = labels.reshape(-1)                  # [batch_size * seq_len]

            # Initialize the loss function
            loss_fn = nn.CrossEntropyLoss()

            # Compute the loss
            loss = loss_fn(logits_reshaped, labels_reshaped)

            return loss, logits
        else:
            return logits


class MetaVanillaTransformers:
    def __init__(self, vocab_size, d_model: int = 64, d_hid: int = 128, nhead: int = 2, nlayers: int = 2, dropout: float = 0.2):
        """
        Meta class to generate multiple TransformerModel configurations.

        Args:
            vocab_size (int): Size of the vocabulary.
            d_model (int, optional): Embedding dimension. Defaults to 64.
            d_hid (int, optional): Hidden dimension. Defaults to 128.
            nhead (int, optional): Number of attention heads. Defaults to 2.
            nlayers (int, optional): Number of Transformer decoder layers. Defaults to 2.
            dropout (float, optional): Dropout rate. Defaults to 0.2.
        """
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
        """
        Yields TransformerModel instances for each configuration.
        """
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
        """
        Returns the number of configurations.

        Returns:
            int: Number of model configurations.
        """
        return len(self.configurations)


