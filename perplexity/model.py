# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from x_transformers import TransformerWrapper, Decoder
import matplotlib.pyplot as plt


class MetaFullyConnectedModels:
    def __init__(self, vocab_size):
        # Parameter Scaling Constants
        self.vocab_size = vocab_size
        self.embedding_dims_and_hidden_dims = [[16, 32], [32, 64], [64, 128], [128, 256], [256, 512], [512, 1024]]
        self.depths = [i for i in range(1, 13, 1)]

    def __iter__(self):
        for item in self.embedding_dims_and_hidden_dims:
            for current_depth in self.depths:
                yield FullyConnectedModel(
                    self.vocab_size, embedding_dim=item[0], hidden_dim=item[1], depth=current_depth
                )

    def __len__(self):
        return len(self.configurations)


class FullyConnectedModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, depth=8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, self.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.inner_layers = nn.ModuleList()
        for _ in range(self.depth):
            self.inner_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.inner_layers.append(nn.ReLU())
            self.inner_layers.append(nn.Dropout(0.2))
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x, src_key_padding_mask=None):
        x = self.embedding(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        for layer in self.inner_layers:
            x = layer(x)
        x = self.fc2(x)
        return x

class XTransformerModel(nn.Module):
    def __init__(self, vocab_size, max_seq_len=512, d_model=512, n_layers=8, n_heads=8, d_ff=2048):
        super().__init__()
        # Transformer model setup
        self.model = TransformerWrapper(
            num_tokens=vocab_size,           # Size of the vocabulary
            max_seq_len=max_seq_len,         # Max sequence length
            attn_layers=Decoder(             # Decoder layers (no encoder for autoregressive generation)
                dim=d_model,                 # Model dimension (embedding size)
                depth=n_layers,              # Number of transformer layers
                heads=n_heads,               # Number of attention heads
                ff_mult=d_ff // d_model,     # Feed-forward multiplier
            )
        )
        self.num_params = sum(p.numel() for p in self.parameters())  # Count parameters

    def forward(self, src, src_key_padding_mask=None):
        """
        Args:
            src: Tensor of shape [batch_size, seq_length]
            src_key_padding_mask: Tensor of shape [batch_size, seq_length] (optional)
        
        Returns:
            logits: Tensor of shape [batch_size, seq_length, vocab_size]
        """
        if src_key_padding_mask is not None:
            # x-transformers expects mask of shape [batch_size, seq_length]
            # where True indicates tokens to be masked (ignored)
            attn_mask = src_key_padding_mask.bool()  # Ensure it's boolean
        else:
            attn_mask = None
        
        # x-transformers expects input of shape [batch_size, seq_length]
        return self.model(src, mask=attn_mask)  # Pass the 2D attention mask directly

class MetaXTransformers:
    def __init__(
        self,
        vocab_size
    ):
        # Predefined configurations to match target parameter counts, including varying d_ff
        self.configurations = [
            # {"d_model": 2, "n_layers": 1, "n_heads": 1, "d_ff": 8},    # ~200k params
            # {"d_model": 4, "n_layers": 1, "n_heads": 1, "d_ff": 16},    # ~400k params
            # {"d_model": 8, "n_layers": 1, "n_heads": 1, "d_ff": 32},    # ~800k params
            {"d_model": 16, "n_layers": 1, "n_heads": 1, "d_ff": 64},    # ~1.6M params
            # {"d_model": 32, "n_layers": 1, "n_heads": 1, "d_ff": 128},    # ~3.2M params
            # {"d_model": 64, "n_layers": 2, "n_heads": 2, "d_ff": 256},    # ~6.5M params
            # {"d_model": 128, "n_layers": 4, "n_heads": 4, "d_ff": 512},   # ~14M params
            # {"d_model": 256, "n_layers": 8, "n_heads": 8, "d_ff": 1024},  # ~34M params
            # {"d_model": 512, "n_layers": 10, "n_heads": 10, "d_ff": 2048},  # ~86M params
        ]

        self.vocab_size = vocab_size

    def __getitem__(self, idx):
        if idx >= len(self.configurations):
            raise IndexError("Configuration index out of range.")
        config = self.configurations[idx]
        return XTransformerModel(
            vocab_size=self.vocab_size,
            d_model=config["d_model"],
            n_layers=config["n_layers"],
            n_heads=config["n_heads"],
            d_ff=config["d_ff"]
        )

    def __len__(self):
        return len(self.configurations)

    def __iter__(self):
        for idx in range(len(self.configurations)):
            yield self[idx]


def plot_top_k_probs(probabilities, k, tokenizer):
    """
    Plot the top-k probabilities as a bar chart.

    Args:
        probabilities: Tensor of shape [vocab_size], the probabilities for the current step.
        k: int, the number of top probabilities to display.
        tokenizer: The tokenizer instance to convert token IDs to readable tokens.
    """
    # Get the top-k probabilities and their indices
    top_k_probs, top_k_indices = torch.topk(probabilities, k)
    
    # Convert token indices to actual tokens
    top_k_tokens = [tokenizer.decode([idx]) for idx in top_k_indices.tolist()]

    # Plot the probabilities as a bar chart
    plt.figure(figsize=(10, 5))
    plt.bar(top_k_tokens, top_k_probs.cpu().numpy())
    plt.title(f"Top {k} Probability Distribution")
    plt.xlabel("Tokens")
    plt.ylabel("Probabilities")
    plt.show()

def generate(meta_model, model_save_path, tokenizer, input_text, max_length, device, temperature=1.0, top_k=10):
    """
    Generates text from the model given an input prompt using temperature-based sampling.

    Args:
        meta_model: A meta model instance like MetaVanillaTransformers or MetaFullyConnectedModels.
        model_save_path: str, path to the saved model's state dict.
        tokenizer: A tokenizer instance (e.g., GPT2Tokenizer).
        input_text: str, input seed text for generating new text.
        max_length: int, the maximum number of tokens to generate.
        device: torch.device, the device to run computations on.
        temperature: float, controls the randomness of predictions by scaling the logits.
        top_k: int, the number of top probabilities to display and plot.

    Returns:
        str: The generated text.
    """
    # Step 1: Encode the input text into token indices
    input_ids = tokenizer.encode(input_text)
    input_ids = torch.tensor([input_ids], device=device)  # Make it a batch of 1

    # Step 2: Initialize the model using the provided meta_model
    model = next(iter(meta_model))  # Get the first model configuration (you can modify this as needed)
    
    # Step 3: Load the model state from the saved path
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Check if the model is a Transformer or FCN
    is_transformer = isinstance(model, XTransformerModel)

    if is_transformer:
        # Transformer Model: Autoregressive Generation (token by token)
        generated_ids = input_ids

        for _ in range(max_length):
            # Pass the input through the model
            with torch.no_grad():
                logits = model(generated_ids)  # Model forward pass
                logits = logits[:, -1, :]  # Only consider the last token's logits

            # Apply temperature to the logits and convert to probabilities
            logits = logits / temperature
            probabilities = F.softmax(logits, dim=-1)

            # Plot top-k probability distribution
            plot_top_k_probs(probabilities.squeeze(), top_k, tokenizer)

            # Sample the next token based on the probability distribution
            next_token_id = torch.multinomial(probabilities, num_samples=1)

            # Append the generated token to the sequence
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

            # Decode and print the generated sequence so far
            print(f"Generated sequence so far: {tokenizer.decode(generated_ids.squeeze().tolist())}")

            # Stop if the EOS token is generated
            if next_token_id.item() == tokenizer.eos_token_id:
                break

        # Decode the generated sequence back to text
        generated_text = tokenizer.decode(generated_ids.squeeze().tolist())

    else:
        generated_ids = input_ids[0]
        for _ in range(max_length):
            # Step 2: Pass the input through the model
            with torch.no_grad():
                logits = model(generated_ids)
            # Step 3: Sample the next token (using greedy sampling for simplicity)
            logits = logits / temperature
            probabilities = torch.softmax(logits, dim=-1).squeeze()[-1, :]
            next_token_id = torch.multinomial(probabilities, num_samples=1)

            # Step 4: Append generated token to the sequence
            generated_ids = torch.cat((generated_ids, next_token_id), dim=0)
            # If end-of-sequence token is generated, stop
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    return generated_text

# Example Usage:
# Assuming you've initialized the model meta-class (MetaVanillaTransformers or MetaFullyConnectedModels)
# and that the model has been trained and saved.

# if __name__ == '__main__':
#     meta_model = MetaXTransformers(vocab_size=50257)
#     tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
#     generated_text = generate(
#         meta_model=meta_model, 
#         model_save_path="saved_models/VanillaTransformer_dv=small_df=1_p=6597696.pt", 
#         tokenizer=tokenizer, 
#         input_text="Once upon a time", 
#         max_length=50, 
#         device=torch.device("cpu"), 
#         temperature=0.3  # Default temperature (you can adjust this)
#     )
#     print(generated_text)


# def verify_model_sizes(vocab_size):
#     """
#     Instantiate models from MetaXTransformers, print the number of parameters for each.
#     """
#     # Instantiate MetaXTransformers with the desired vocab size
#     meta_transformers = MetaXTransformers(vocab_size=vocab_size)

#     # Iterate through the configurations and instantiate models
#     for idx, transformer_model in enumerate(meta_transformers):
#         # Count the parameters
#         param_count = transformer_model.num_params
#         # Print the configuration index and parameter count
#         print(f"Model {idx+1}: {param_count} parameters")

# if __name__ == "__main__":
#     # Set a sample vocabulary size (this should match your intended vocabulary size)
#     vocab_size = 50187  # Example vocab size
#     # Verify model sizes
#     verify_model_sizes(vocab_size)


# print(generate(
#     MetaXTransformers(len(GPT2Tokenizer.from_pretrained("gpt2"))),
#     "local/VanillaTransformer_dv=small_df=0.1_p=202612.pt",
#     GPT2Tokenizer.from_pretrained("gpt2"),
#     "we are trying to",
#     10,
#     torch.device("cpu"),
#     0.3
# ))
