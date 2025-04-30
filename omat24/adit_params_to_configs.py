# adit_params_to_configs.py

#!/usr/bin/env python3
"""
adit_params_to_configs.py

Given a target parameter count (in millions), predict transformer
hyperparameters using empirical power‐law fits for d_model, heads, layers,
and then instantiate a real ADiTS2EFSModel to report the exact parameter count.
"""

import sys
import types
import math
from models.adit import ADiTS2EFSModel

# ─── Stub torch_scatter & torch_geometric hooks ───
_sc_mod = types.ModuleType("torch_scatter")
_sc_mod.scatter = lambda src, idx, dim=0, reduce="sum": src
sys.modules["torch_scatter"] = _sc_mod

_tg = types.ModuleType("torch_geometric")
sys.modules["torch_geometric"] = _tg
_tg_utils = types.ModuleType("torch_geometric.utils")
_tg_utils.to_dense_batch = lambda x, batch: (x, batch)
sys.modules["torch_geometric.utils"] = _tg_utils
setattr(_tg, "utils", _tg_utils)
_tg_nn = types.ModuleType("torch_geometric.nn")
_tg_nn.radius_graph = lambda *a, **k: None
sys.modules["torch_geometric.nn"] = _tg_nn
setattr(_tg, "nn", _tg_nn)
sys.modules["torch_geometric.typing"]     = types.ModuleType("torch_geometric.typing")
sys.modules["torch_geometric.isinstance"] = types.ModuleType("torch_geometric.isinstance")
# ─────────────────────────────────────────────────


def calculate_hyperparameters(param_count_millions):
    # 1) Layers
    A_L, B_L = 3.6667, 0.270
    n_layers = max(1, round(A_L * param_count_millions**B_L))

    # 2) Heads (piecewise)
    bp_h = 346.0
    if param_count_millions <= bp_h:
        A1, B1 = 1.96, 0.370
        n_heads = max(1, round(A1 * param_count_millions**B1))
    else:
        A2, B2 = 1.15, 0.359
        n_heads = max(1, round(A2 * param_count_millions**B2))

    # 3) Hidden dim (d_model)
    A_D, B_D = 113.09, 0.394
    d_model_raw = A_D * param_count_millions**B_D
    d_model = max(64, round(d_model_raw / 64) * 64)

    # 4) FFN dim
    ffw_size = 4 * d_model

    # 5) Divisibility adjust (heads ↔ d_model)
    if d_model % n_heads != 0:
        cands = [h for h in range(max(1,n_heads-4), n_heads+5) if d_model%h==0]
        if cands:
            n_heads = min(cands, key=lambda h: abs(h-n_heads))

    # 6) Instantiate real model for exact count
    model = ADiTS2EFSModel(
        max_num_elements=119,
        d_model=d_model,
        nhead=n_heads,
        dim_feedforward=ffw_size,
        activation="gelu",
        dropout=0.1,
        norm_first=True,
        bias=True,
        num_layers=n_layers
    )
    exact_M = model.num_params / 1e6
    head_dim = d_model // n_heads

    return {
        "model_size_M":         param_count_millions,
        "exact_param_count_M":  exact_M,
        "n_layers":             n_layers,
        "n_heads":              n_heads,
        "d_model":              d_model,
        "ffw_size":             ffw_size,
        "head_dim":             head_dim
    }


def print_hyperparameters(hp):
    print("\n===== Transformer Hyperparameters =====")
    print(f"Target: {hp['model_size_M']:.3f} M params")
    print(f"Exact:  {hp['exact_param_count_M']:.3f} M params\n")
    print(f"  d_model  : {hp['d_model']}")
    print(f"  nhead  : {hp['n_heads']}  (head_dim={hp['head_dim']})")
    print(f"  dim_feedforward : {hp['ffw_size']}")
    print(f"  num_layers : {hp['n_layers']}")
    print()


def main():
    if len(sys.argv)!=2:
        print(__doc__); sys.exit(1)
    try:
        t = float(sys.argv[1]); assert t>0
    except:
        print("Error: give a positive number (e.g. 0.01 for 10K params).")
        sys.exit(1)

    hp = calculate_hyperparameters(t)
    print_hyperparameters(hp)


if __name__=="__main__":
    main()
