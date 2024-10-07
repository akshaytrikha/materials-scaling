import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Decoder

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


# Example of using this in the text generation function
def generate(model_save_path, tokenizer, input_text, max_length, device):
    """
    Generates text from the model given an input prompt.

    input_text: str, input seed text for generating new text
    device: torch device (cpu or cuda)

    Returns:
    - Generated text as a string
    """
    # Step 1: Encode the input text to token indices
    input_ids = tokenizer.encode(input_text)
    input_ids = torch.tensor([input_ids], device=device)  # Make it a batch of 1
    print(f"input_ids.shape is {input_ids.shape}")
    
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
            print(logits.shape)
            
        # Step 3: Sample the next token (using greedy sampling for simplicity)
        next_token_id = torch.argmax(logits, dim=-1).unsqueeze(1)
        print(f"the next_token_id.shape is {next_token_id.shape}")
        
        # Step 4: Append the generated token to the sequence
        generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
        print(tokenizer.decode(generated_ids.squeeze().tolist()))
        
        # Stop if end-of-sequence token is generated
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    return ""