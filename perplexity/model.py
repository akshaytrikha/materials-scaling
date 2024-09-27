import torch
import torch.nn as nn
import math


class MetaFullyConnectedModels:
    def __init__(self, vocab_size, context_length):
        self.vocab_size = vocab_size
        self.context_length = context_length

        # Parameter Scaling Constants
        self.hidden_dims = [2, 4, 8, 16, 32, 64, 128, 256, 512]
        self.depths = [1, 2, 3, 4, 5, 6, 7]
        self.embedding_dim = 128

    def __iter__(self):
        for hid_dim in self.hidden_dims:
            for depth in self.depths:
                yield FullyConnectedModel(
                    vocab_size=self.vocab_size,
                    context_length=self.context_length,
                    embedding_dim=self.embedding_dim,
                    hidden_dim=hid_dim,
                    depth=depth,
                )

    def __len__(self):
        return len(self.configurations)


class FullyConnectedModel(nn.Module):
    def __init__(self, vocab_size, context_length, embedding_dim, hidden_dim, depth):
        """
        Initializes the FullyConnectedModel with variable depth.

        Args:
            vocab_size (int): Size of the vocabulary.
            context_length (int): Number of previous tokens to consider.
            embedding_dim (int): Dimension of the token embeddings.
            hidden_dim (int): Dimension of the hidden layers.
            depth (int): Number of hidden layers.
        """
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

        # Use ModuleList to hold the layers
        self.layers = nn.ModuleList(layers)

        # Calculate total number of parameters
        self.width = hidden_dim
        self.depth = depth
        self.num_params = sum(p.numel() for p in self.parameters())
        print(
            f"Total parameters: {self.num_params} (Embedding: {embedding_dim * vocab_size}, "
            f"Layers: {hidden_dim * (embedding_dim * context_length + (depth -1)* hidden_dim)}, "
            # f"Output: {hidden_dim * vocab_size})"
            f"Wide: {self.width}, Depth: {self.depth}"
        )


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


def generate(model_save_path, tokenizer, input_text, max_length, device="cpu"):
    """
    Generates text from the model given an input prompt.

    input_text: str, input seed text for generating new text
    device: torch device (cpu or cuda)

    Returns:
    - Generated text as a string
    """
    # Step 1: Encode the input text to token indices
    if device == "cpu":
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
