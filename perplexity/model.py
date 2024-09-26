import torch
import torch.nn as nn
import math


# class MetaFullyConnectedModels:
#     def __init__(self, vocab_size):
#         # Parameter Scaling Constants
#         self.embedding_dims = [16, 32, 64, 128, 256, 512, 1024]
#         self.hidden_dims = [16, 32, 64, 128, 256, 512, 1024]
#         self.depths = [1, 2, 3, 4, 5, 6, 7]

#         self.vocab_size = vocab_size

#         # Generate all combinations of embedding_dims, hidden_dims, and depths
#         self.configurations = []
#         for emb_dim, hid_dim, depth in zip(
#             self.embedding_dims, self.hidden_dims, self.depths
#         ):
#             self.configurations.append((emb_dim, hid_dim, depth))

#     def __iter__(self):
#         for emb_dim, hid_dim, depth in self.configurations:
#             yield FullyConnectedModel(
#                 self.vocab_size, embedding_dim=emb_dim, hidden_dim=hid_dim, depth=depth
#             )

#     def __len__(self):
#         return len(self.configurations)


# class FullyConnectedModel(nn.Module):
#     def __init__(self, vocab_size, context_length, embedding_dim, hidden_dim, depth):
#         """
#         Initializes the FullyConnectedModel with variable depth.

#         Args:
#             vocab_size (int): Size of the vocabulary.
#             context_length (int): Number of previous tokens to consider.
#             embedding_dim (int): Dimension of the token embeddings.
#             hidden_dim (int): Dimension of the hidden layers.
#             depth (int): Number of hidden layers.
#         """
#         super(FullyConnectedModel, self).__init__()
#         self.context_length = context_length
#         self.embedding = nn.Embedding(vocab_size, embedding_dim)

#         layers = []
#         input_size = embedding_dim * context_length

#         # Create the first hidden layer
#         layers.append(nn.Linear(input_size, hidden_dim))
#         layers.append(nn.ReLU())

#         # Create (depth - 1) additional hidden layers
#         for _ in range(depth - 1):
#             layers.append(nn.Linear(hidden_dim, hidden_dim))
#             layers.append(nn.ReLU())

#         # Output layer
#         layers.append(nn.Linear(hidden_dim, vocab_size))

#         # Use ModuleList to hold the layers
#         self.layers = nn.ModuleList(layers)

#         # Calculate total number of parameters
#         self.num_params = sum(p.numel() for p in self.parameters())
#         print(
#             f"Total parameters: {self.num_params} (Embedding: {embedding_dim * vocab_size}, "
#             f"Layers: {hidden_dim * (embedding_dim * context_length + (depth -1)* hidden_dim)}, "
#             f"Output: {hidden_dim * vocab_size})"
#         )

import torch
import torch.nn as nn
import math


class MetaFullyConnectedModels:
    def __init__(self, vocab_size, context_length, num_configs=10):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.num_configs = num_configs

        # Base configuration
        self.base_embedding_dim = 16
        self.base_hidden_dim = 16
        self.base_depth = 1

        # Initialize configurations list
        self.configurations = []

        # Current configuration parameters
        current_E = self.base_embedding_dim
        current_H = self.base_hidden_dim
        current_D = self.base_depth
        current_params = self.compute_params(current_E, current_H, current_D)
        self.configurations.append((current_E, current_H, current_D))

        for _ in range(1, self.num_configs):
            # Target parameters: double the previous
            target_params = 2 * current_params

            # Scaling factors
            scale_factor = 2 ** (1 / 3)  # ~1.26
            new_E = int(math.ceil(current_E * scale_factor))
            new_H = int(math.ceil(current_H * scale_factor))
            new_D = current_D  # Start by keeping depth the same

            # Calculate new parameters with scaled E and H
            new_params = self.compute_params(new_E, new_H, new_D)

            # Check if new_params is close to target
            if new_params < target_params * 0.9:
                # If too low, consider increasing depth by 1
                new_D += 1
                new_params = self.compute_params(new_E, new_H, new_D)

            elif new_params > target_params * 1.1:
                # If too high, reduce scaling factors slightly
                new_E = int(math.floor(current_E * scale_factor * 0.95))
                new_H = int(math.floor(current_H * scale_factor * 0.95))
                new_D = current_D
                new_params = self.compute_params(new_E, new_H, new_D)
                if new_params < target_params * 0.9:
                    new_D += 1
                    new_params = self.compute_params(new_E, new_H, new_D)

            # Append the new configuration
            self.configurations.append((new_E, new_H, new_D))
            current_E, current_H, current_D, current_params = (
                new_E,
                new_H,
                new_D,
                new_params,
            )

    def compute_params(self, E, H, D):
        embedding_params = self.vocab_size * E
        first_hidden_params = E * self.context_length * H
        additional_hidden_params = H * H * (D - 1) if D > 1 else 0
        output_params = H * self.vocab_size
        total_params = (
            embedding_params
            + first_hidden_params
            + additional_hidden_params
            + output_params
        )
        return total_params

    def __iter__(self):
        for emb_dim, hid_dim, depth in self.configurations:
            yield FullyConnectedModel(
                vocab_size=self.vocab_size,
                context_length=self.context_length,
                embedding_dim=emb_dim,
                hidden_dim=hid_dim,
                depth=depth,
            )

    def __len__(self):
        return len(self.configurations)


class FullyConnectedModel(nn.Module):
    def __init__(self, vocab_size, context_length, embedding_dim, hidden_dim, depth):
        super(FullyConnectedModel, self).__init__()
        self.context_length = context_length
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        layers = []
        input_size = embedding_dim * context_length

        # Create the first hidden layer
        layers.append(nn.Linear(input_size, hidden_dim))
        layers.append(nn.ReLU())

        # Create (depth - 1) additional hidden layers
        for _ in range(depth - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dim, vocab_size))

        # Use Sequential to hold the layers
        self.layers = nn.Sequential(*layers)

        # Calculate total number of parameters
        self.num_params = sum(p.numel() for p in self.parameters())
        print(
            f"Total parameters: {self.num_params} "
            f"(Embedding: {embedding_dim * vocab_size}, "
            f"Layers: {embedding_dim * context_length * hidden_dim + hidden_dim**2 * (depth -1)}, "
            f"Output: {hidden_dim * vocab_size})"
        )

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, context_length, embedding_dim)
        x = x.view(x.size(0), -1)  # (batch_size, embedding_dim * context_length)
        x = self.layers(x)
        return x

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, context_length).

        Returns:
            Tensor: Output logits of shape (batch_size, vocab_size).
        """
        # x shape: (batch_size, context_length)
        x = self.embedding(x)  # (batch_size, context_length, embedding_dim)

        # Flatten the embeddings to create a single vector per example
        x = x.view(x.size(0), -1)  # (batch_size, embedding_dim * context_length)

        # Pass through all layers
        for layer in self.layers:
            x = layer(x)

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

def generate(model_save_path, tokenizer, input_text, max_length, device='cpu'):
    """
    Generates text from the model given an input prompt.
    
    input_text: str, input seed text for generating new text
    device: torch device (cpu or cuda)
    
    Returns:
    - Generated text as a string
    """
    # Step 1: Encode the input text to token indices
    if device == 'cpu':
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    input_ids = tokenizer.encode(input_text)
    input_ids = torch.tensor([input_ids], device=device)  # Make it a batch of 1
    # Load the model and set it to evaluation mode
    model = torch.load(model_save_path)
    model = model.to(device)
    model.eval()
    # Initialize the generated sequence with the input ids
    generated_ids = input_ids
    for _ in range(max_length):
        # Step 2: Pass the input through the model
        with torch.no_grad():
            logits = model(generated_ids)
        # Step 3: Sample the next token (using greedy sampling for simplicity)
        next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)
        # Step 4: Append the generated token to the sequence
        generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
        # If end-of-sequence token is generated, stop
        if next_token_id.item() == tokenizer.eos_token_id:
            break
    # Step 5: Decode the generated sequence back to text
    generated_text = tokenizer.decode(generated_ids.squeeze().tolist())
    return generated_text
