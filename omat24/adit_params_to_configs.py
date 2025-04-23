#!/usr/bin/env python3
"""
Transformer Hyperparameter Calculator

Given a target parameter count (in **millions**), predict a transformer
architecture (d_model, n_heads, n_layers, kv_size, ffw_size) based on
empirical scaling laws.

Usage:
    python3 transformer_hyperparams.py <param_count_in_millions>

Examples:
    python3 transformer_hyperparams.py 1000    # 1 billion params
    python3 transformer_hyperparams.py 0.01    # 10 000   params
"""

import sys
import math

def calculate_hyperparameters(param_count_millions):
    # 1) Layers:    n_layers ≈ A_layers * size^B_layers
    A_layers, B_layers = 3.667, 0.270
    n_layers = max(2, round(A_layers * param_count_millions**B_layers))

    # 2) Heads: piecewise power law
    bp_heads = 389.0   # in millions
    if param_count_millions <= bp_heads:
        A1, B1 = 1.96, 0.370
        n_heads = round(A1 * param_count_millions**B1)
    else:
        A2, B2 = 1.15, 0.359
        n_heads = round(A2 * param_count_millions**B2)
    n_heads = max(1, n_heads)  # at least 1 head

    # 3) KV size step at ~365 M params
    kv_size = 64 if param_count_millions < 365.0 else 128

    # 4) d_model from param budget: total ≈ 12 * n_layers * d_model^2
    total_params = param_count_millions * 1e6
    coeff = 12.0
    raw = math.sqrt(total_params / (coeff * n_layers))
    d_model = max(64, round(raw/64)*64)

    # 5) FFN size ≈ 4× d_model
    ffw_size = 4 * d_model

    # 6) Adjust heads↔d_model divisibility
    if d_model % n_heads != 0:
        # try small tweak in head count
        candidates = [h for h in range(max(1,n_heads-4), n_heads+5)
                      if d_model % h == 0]
        if candidates:
            n_heads = min(candidates, key=lambda h: abs(h-n_heads))
        else:
            # fallback: make d_model divisible by n_heads
            d_model = n_heads * round(d_model / n_heads)
            ffw_size = 4 * d_model

    # 7) head_dim and actual size calc (approx)
    head_dim = d_model // n_heads
    vocab = 50000
    qkv = 4 * (n_layers * d_model * d_model)
    ffn = 2 * (n_layers * d_model * ffw_size)
    embed = vocab * d_model
    actual_M = (qkv + ffn + embed) / 1e6

    return {
        "model_size_M":         param_count_millions,
        "actual_param_count_M": actual_M,
        "n_layers":             n_layers,
        "n_heads":              n_heads,
        "kv_size":              kv_size,
        "d_model":              d_model,
        "ffw_size":             ffw_size,
        "head_dim":             head_dim
    }


def print_hyperparameters(hp):
    print("\n===== Transformer Hyperparameters =====")
    print(f"Target:   {hp['model_size_M']:.3f} M params")
    print(f"Estimate: {hp['actual_param_count_M']:.3f} M params\n")
    print(f"  d_model   : {hp['d_model']}")
    print(f"  n_heads   : {hp['n_heads']}  (head_dim={hp['head_dim']})")
    print(f"  n_layers  : {hp['n_layers']}")
    print(f"  kv_size   : {hp['kv_size']}")
    print(f"  ffw_size  : {hp['ffw_size']}")
    print("\n(Estimates from empirical scaling fits.)")


def main():
    if len(sys.argv)!=2:
        print(__doc__)
        sys.exit(1)

    try:
        # **key change**: allow floats (fractions of a million)
        target = float(sys.argv[1])
        if target <= 0:
            raise ValueError()
    except:
        print("Error: please provide a positive number, e.g. 0.01 for 10 K params.")
        sys.exit(1)

    hp = calculate_hyperparameters(target)
    print_hyperparameters(hp)


if __name__=="__main__":
    main()
