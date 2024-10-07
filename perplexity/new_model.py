import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Decoder
from transformers import GPT2Tokenizer

class PredefinedTransformerModel(nn.Module):
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
        vocab_size,
        d_model: int = 64,  # Default values for the transformer size
        n_layers: int = 2,
        n_heads: int = 2,
        d_ff: int = 256,
    ):
        # You can add more configurations if needed
        self.d_models = [d_model]
        self.n_layers = [n_layers]
        self.n_heads = [n_heads]
        self.d_ff = d_ff

        self.configurations = []
        for d_model in self.d_models:
            for n_layers in self.n_layers:
                for n_heads in self.n_heads:
                    if d_model % n_heads == 0:  # Ensure model size is divisible by heads
                        self.configurations.append((d_model, n_layers, n_heads))

        self.vocab_size = vocab_size

    def __iter__(self):
        for d_model, n_layers, n_heads in self.configurations:
            yield PredefinedTransformerModel(
                vocab_size=self.vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                d_ff=self.d_ff,
            )

    def __len__(self):
        return len(self.configurations)

def generate(meta_model, model_save_path, tokenizer, input_text, max_length, device):
    """
    Generates text from the model given an input prompt.

    Args:
        meta_model: A meta model instance like MetaVanillaTransformers or MetaFullyConnectedModels.
        model_save_path: str, path to the saved model's state dict.
        tokenizer: A tokenizer instance (e.g., GPT2Tokenizer).
        input_text: str, input seed text for generating new text.
        max_length: int, the maximum number of tokens to generate.
        device: torch.device, the device to run computations on.

    Returns:
        str: The generated text.
    """
    # Step 1: Encode the input text into token indices
    input_ids = tokenizer.encode(input_text)
    input_ids = torch.tensor([input_ids], device=device)  # Make it a batch of 1
    print(f"input_ids.shape is {input_ids.shape}")

    # Step 2: Initialize the model using the provided meta_model
    model = next(iter(meta_model))  # Get the first model configuration (you can modify this as needed)
    
    # Step 3: Load the model state from the saved path
    model.load_state_dict(torch.load(model_save_path), map_location=device)
    model = model.to(device)
    model.eval()

    # Initialize the generated sequence with the input ids
    generated_ids = input_ids

    # Step 4: Generate tokens iteratively until max_length is reached or EOS token is encountered
    for _ in range(max_length):
        # Pass the input through the model
        with torch.no_grad():
            logits = model(generated_ids)  # Model forward pass
            logits = logits[:, -1, :]  # Only consider the last token's logits (for greedy decoding)

        # Sample the next token (greedy sampling)
        next_token_id = torch.argmax(logits, dim=-1).unsqueeze(0)

        # Append the generated token to the sequence
        generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

        # Decode and print the generated sequence so far
        print(f"Generated sequence so far: {tokenizer.decode(generated_ids.squeeze().tolist())}")

        # Stop if the EOS token is generated
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    # Step 5: Decode the generated sequence back to text
    generated_text = tokenizer.decode(generated_ids.squeeze().tolist())

    return generated_text

# Example Usage:
# Assuming you've initialized the model meta-class (MetaVanillaTransformers or MetaFullyConnectedModels)
# and that the model has been trained and saved.

meta_model = MetaXTransformers(vocab_size=50257)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
generated_text = generate(meta_model, "/kaggle/working/materials-scaling/perplexity/kaggle/working/saved_models/wikitext-2-v1_VanillaTransformer_ts=2024_10_07-21:00:08/VanillaTransformer_dv=small_df=1_p=6597696.pt", tokenizer, "Once upon a time", 50, torch.device("cuda"))
print(generated_text)
