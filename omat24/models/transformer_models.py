import torch
from x_transformers import TransformerWrapper, Decoder


class XTransformerModel(torch.nn.Module):
    def __init__(self, vocab_size, max_seq_len, d_model, n_layers, n_heads, d_ff):
        super().__init__()
        self.atomic_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.position_embedding = torch.nn.Linear(3, d_model)  # Assuming positions are 3D
        self.transformer = TransformerWrapper(
            num_tokens=vocab_size,
            max_seq_len=max_seq_len,
            attn_layers=Decoder(
                dim=2 * d_model,
                depth=n_layers,
                heads=n_heads,
                ff_mult=d_ff // d_model,
            ),
        )
        self.energy_predictor = torch.nn.Linear(d_model, 1)  # Example for energy prediction

    def forward(self, atomic_numbers, positions, src_key_padding_mask=None):
        atomic_emb = self.atomic_embedding(atomic_numbers)  # [batch, seq, d_model]
        pos_emb = self.position_embedding(positions)  # [batch, seq, d_model]
        print(atomic_emb.shape)
        print(pos_emb.shape)
        combined_emb = torch.cat([atomic_emb, pos_emb], dim=-1)
        breakpoint()
        logits = self.transformer(combined_emb, mask=src_key_padding_mask)
        energy = self.energy_predictor(logits.mean(dim=1))  # Example aggregation
        return energy
