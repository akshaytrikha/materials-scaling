import torch
from x_transformers import TransformerWrapper, Decoder

# embedding

class TokenEmbedding(torch.nn.Module):
    def __init__(self, dim, num_tokens, l2norm_embed = False):
        super().__init__()
        self.l2norm_embed = l2norm_embed
        self.emb = torch.nn.Embedding(num_tokens, dim)

    def forward(self, x):
        token_emb = self.emb(x.long())
        return torch.l2norm(token_emb) if self.l2norm_embed else token_emb

class XTransformerModel(torch.nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, n_layers, n_heads, d_ff):
        super().__init__()
        self.position_embedding = torch.nn.Linear(3, d_model)  # Assuming positions are 3D
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Decoder(
                dim=d_model,
                depth=n_layers,
                heads=n_heads,
                ff_mult=d_ff // d_model,
            ),
            token_emb=TokenEmbedding(d_model, vocab_size)
        )
        self.energy_predictor = torch.nn.Linear(d_model, 1)  # Example for energy prediction

    def forward(self, atomic_numbers, positions, src_key_padding_mask=None):
        # atomic_emb = self.atomic_embedding(atomic_numbers)  # [batch, seq, d_model]
        positions_emb = self.position_embedding(positions)
        print(positions_emb.shape)
        logits = self.transformer(atomic_numbers, mask=src_key_padding_mask, pos=positions_emb)
        energy = self.energy_predictor(logits.mean(dim=1))  # Example aggregation
        return energy
