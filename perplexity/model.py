# model.py

import torch
import torch.nn as nn
import math
from typing import Tuple
from transformers import GPT2Tokenizer


class MetaFullyConnectedModels:
    def __init__(self, vocab_size):
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

    def forward(self, x, src_key_padding_mask=None):
        x = self.embedding(x)
        x = x.mean(dim=1)  # Average embeddings across sequence length
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # [d_model/2]
        pe = torch.zeros(max_len, 1, d_model)  # [max_len, 1, d_model]
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # Apply sin to even indices
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # Apply cos to odd indices
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        ntoken: int,
        d_model: int,
        nhead: int,
        d_hid: int,
        nlayers: int,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.model_type = "Transformer"
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

    def forward(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor = None,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Args:
            src: Tensor, shape [seq_len, batch_size]
            src_mask: Tensor, shape [seq_len, seq_len] (optional)

        Returns:
            output Tensor of shape [batch_size, seq_len, ntoken]
        """
        src = self.embedding(src) * math.sqrt(
            self.d_model
        )  # [seq_len, batch_size, d_model]
        src = self.pos_encoder(src)  # [seq_len, batch_size, d_model]

        output = self.transformer_encoder(
            src, src_key_padding_mask=src_key_padding_mask
        )  # [seq_len, batch_size, d_model]
        output = self.linear(output)  # [seq_len, batch_size, ntoken]
        output = output.transpose(0, 1)  # [batch_size, seq_len, ntoken]
        return output


class MetaVanillaTransformers:
    def __init__(
        self,
        vocab_size,
        d_model: int = 640,
        d_hid: int = 2560,
        nhead: int = 10,
        nlayers: int = 10,
        dropout: float = 0.0,
    ):
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
                dropout=self.dropout,
            )

    def __len__(self):
        return len(self.configurations)


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
        next_token_id = torch.argmax(logits, dim=-1).unsqueeze(
            1
        )  # [0][len(input_text.split(" ")) - 1].unsqueeze(0).unsqueeze(0)
        # Step 4: Append the generated token to the sequence
        print(f"the next_token_id.shape is {next_token_id.shape}")
        generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
        print(tokenizer.decode(generated_ids.squeeze().tolist()))
        # print(generated_ids.shape)
        # If end-of-sequence token is generated, stop

        if next_token_id.item() == tokenizer.eos_token_id:
            break

    # Step 5: Decode the generated sequence back to text
    # generated_text = tokenizer.decode(generated_ids.squeeze().tolist())
    return ""  # generated_text


# generate(
#     "saved_models/wikitext-2-v1_FCN_ts=20241002_191720/FCN_dv=small_df=0.01_p=1658753.pt",
#     GPT2Tokenizer.from_pretrained("gpt2"),
#     "we are trying to",
#     10,
#     torch.device("cpu"),
# )
