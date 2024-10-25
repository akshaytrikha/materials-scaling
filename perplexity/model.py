# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
from x_transformers import TransformerWrapper, Decoder
import matplotlib.pyplot as plt


class MetaFullyConnectedModels:
    def __init__(self, vocab_size):
        self.configurations = [
            {"embedding_dim": 2, "hidden_dim": 2, "depth": 1},      # 251,297 params
            {"embedding_dim": 4, "hidden_dim": 4, "depth": 2},      # 452,373 params
            # {"embedding_dim": 8, "hidden_dim": 8, "depth": 4},      # 854,729 params
            # {"embedding_dim": 16, "hidden_dim": 16, "depth": 8},    # 1,660,929 params
            # {"embedding_dim": 32, "hidden_dim": 32, "depth": 12},   # 3,280,433 params
            # {"embedding_dim": 64, "hidden_dim": 64, "depth": 12},   # 6,537,233 params
            # {"embedding_dim": 128, "hidden_dim": 128, "depth": 12}, # 13,130,705 params
            # {"embedding_dim": 320, "hidden_dim": 320, "depth": 16}, # 33,960,977 params
            # {"embedding_dim": 640, "hidden_dim": 640, "depth": 54}  # 86,942,417 params
        ]
        self.vocab_size = vocab_size

    def __iter__(self):
        for item in self.embedding_dims_and_hidden_dims:
            for current_depth in self.depths:
                yield FullyConnectedModel(
                    self.vocab_size, embedding_dim=item[0], hidden_dim=item[1], depth=current_depth
                )
    
    def __getitem__(self, idx):
        if idx >= len(self.configurations):
            raise IndexError("Configuration index out of range.")
        config = self.configurations[idx]
        return FullyConnectedModel(
            vocab_size=self.vocab_size,
            embedding_dim=config["embedding_dim"],
            hidden_dim=config["hidden_dim"],
            depth=config["depth"]
        )

    def __len__(self):
        return len(self.configurations)
    
    def __iter__(self):
        for idx in range(len(self.configurations)):
            yield self[idx]


class FullyConnectedModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=128, depth=8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.embedding = nn.Embedding(vocab_size, self.embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, self.hidden_dim)
        self.layernorm = nn.LayerNorm(self.hidden_dim)
        self.leakyrelu = nn.LeakyReLU(negative_slope=0.01)
        self.inner_layers = nn.ModuleList()
        for _ in range(self.depth):
            self.inner_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.inner_layers.append(nn.LayerNorm(self.hidden_dim))
            self.inner_layers.append(nn.LeakyReLU(negative_slope=0.01))
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, x, src_key_padding_mask=None):
        x = self.embedding(x)
        # print(f"embedding is {x}")
        x = self.fc1(x)
        # print(f"fc1 is {x}")
        x = self.layernorm(x)
        # print(f"layernorm is {x}")
        x = self.leakyrelu(x)
        # print(f"leakyrelu is {x}")
        for layer in self.inner_layers:
            x = layer(x)
            # print(f"layer output is {x}")
        x = self.fc2(x)
        # print(f"fc2 is {x}")
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
            {"d_model": 2, "n_layers": 1, "n_heads": 1, "d_ff": 8},    # ~200k params
            # {"d_model": 4, "n_layers": 1, "n_heads": 1, "d_ff": 16},    # ~400k params
            # {"d_model": 8, "n_layers": 1, "n_heads": 1, "d_ff": 32},    # ~800k params
            # {"d_model": 16, "n_layers": 1, "n_heads": 1, "d_ff": 64},    # ~1.6M params
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

from tokenizers import Tokenizer
from tokenizers.models import BPE
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def load_bpe_tokenizer(tokenizer_path):
    """
    Load the saved ByteLevelBPE tokenizer
    
    Args:
        tokenizer_path: Path to the saved tokenizer.json file
    
    Returns:
        Tokenizer: Loaded tokenizer with added convenience methods
    """
    # Load the base tokenizer
    base_tokenizer = Tokenizer.from_file(tokenizer_path)
    
    # Create a wrapper class to add the required methods
    class TokenizerWrapper:
        def __init__(self, base_tokenizer):
            self.tokenizer = base_tokenizer
            self.pad_token_id = self.tokenizer.token_to_id("<pad>")
            self.eos_token_id = self.tokenizer.token_to_id("</s>")
        
        def encode(self, text):
            # Directly return the ids from the encoding
            encoded = self.tokenizer.encode(text)
            return encoded.ids
            
        def decode(self, ids):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            if isinstance(ids, int):
                ids = [ids]
            return self.tokenizer.decode(ids)
    
    # Return wrapped tokenizer
    return TokenizerWrapper(base_tokenizer)

def generate(meta_model, model_save_path, tokenizer_path, input_text, max_length, device, temperature=1.0, top_k=10):
    """
    Generates text from the model given an input prompt using temperature-based sampling.

    Args:
        meta_model: A meta model instance like MetaVanillaTransformers or MetaFullyConnectedModels.
        model_save_path: str, path to the saved model's state dict.
        tokenizer_path: str, path to the saved tokenizer.json file
        input_text: str, input seed text for generating new text.
        max_length: int, the maximum number of tokens to generate.
        device: torch.device, the device to run computations on.
        temperature: float, controls the randomness of predictions by scaling the logits.
        top_k: int, the number of top probabilities to display and plot.

    Returns:
        str: The generated text.
    """
    # Load the tokenizer
    tokenizer = load_bpe_tokenizer(tokenizer_path)
    
    # Step 1: Encode the input text into token indices
    print(f"Input text: {input_text}")
    input_ids = tokenizer.encode(input_text)
    input_ids = torch.tensor([input_ids], device=device)  # Make it a batch of 1

    # Step 2: Initialize the model using the provided meta_model
    model = next(iter(meta_model))  # Get the first model configuration
    
    # Step 3: Load the model state from the saved path
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Check if the model is a Transformer or FCN
    is_transformer = isinstance(model, XTransformerModel)
    
    # Create padding mask (all False since we're not padding)
    src_key_padding_mask = torch.zeros_like(input_ids, dtype=torch.bool, device=device)

    def plot_top_k_probs(probabilities, k, tokenizer):
        """Plot the top-k probabilities as a bar chart."""
        top_k_probs, top_k_indices = torch.topk(probabilities, k)
        top_k_tokens = [tokenizer.decode(idx.item()) for idx in top_k_indices]
        
        plt.figure(figsize=(10, 5))
        plt.bar(top_k_tokens, top_k_probs.cpu().numpy())
        plt.title(f"Top {k} Probability Distribution")
        plt.xlabel("Tokens")
        plt.ylabel("Probabilities")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    with torch.no_grad():
        if is_transformer:
            # Transformer Model: Autoregressive Generation (token by token)
            generated_ids = input_ids

            for _ in range(max_length):
                # Update padding mask for the current sequence
                current_mask = torch.zeros_like(generated_ids, dtype=torch.bool, device=device)
                
                # Pass the input through the model with padding mask
                logits = model(generated_ids, src_key_padding_mask=current_mask)
                logits = logits[:, -1, :]  # Only consider the last token's logits

                # Apply temperature and convert to probabilities
                logits = logits / temperature
                probabilities = F.softmax(logits, dim=-1)

                # Optional: Plot top-k probability distribution
                if top_k > 0:
                    plot_top_k_probs(probabilities.squeeze(), top_k, tokenizer)

                # Sample the next token
                next_token_id = torch.multinomial(probabilities, num_samples=1)
                generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

                # Print progress
                print(f"Generated sequence so far: {tokenizer.decode(generated_ids.squeeze().tolist())}")

                # Stop if EOS token is generated
                if next_token_id.item() == tokenizer.eos_token_id:
                    break

        else:
            # FCN Model: Single forward pass with attention to the last token
            generated_ids = input_ids.squeeze()
            for _ in range(max_length):
                # Pass the sequence through the model
                logits = model(generated_ids.unsqueeze(0), src_key_padding_mask=src_key_padding_mask)
                
                # Get predictions for the last token
                last_token_logits = logits[0, -1, :]
                
                # Apply temperature and get probabilities
                scaled_logits = last_token_logits / temperature
                probabilities = F.softmax(scaled_logits, dim=-1)

                # Optional: Plot top-k probability distribution
                if top_k > 0:
                    plot_top_k_probs(probabilities, top_k, tokenizer)

                # Sample next token
                next_token_id = torch.multinomial(probabilities, num_samples=1)
                generated_ids = torch.cat((generated_ids, next_token_id), dim=0)

                # Print progress
                print(f"Generated sequence so far: {tokenizer.decode(generated_ids.tolist())}")

                # Stop if EOS token is generated
                if next_token_id.item() == tokenizer.eos_token_id:
                    break

    # Decode the generated sequence back to text
    generated_text = tokenizer.decode(generated_ids.squeeze().tolist())
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

# generated_text = generate(
#     meta_model=MetaXTransformers(vocab_size=10000),  # Use your vocab_size from tokenizer training
#     model_save_path="VanillaTransformer_dv=small_df=1_p=41584.pt",
#     tokenizer_path="bpe_tokenizer/tokenizer.json",  # Path to your saved tokenizer
#     input_text="hello this is a test",
#     max_length=50,
#     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
#     temperature=0.7
# )
