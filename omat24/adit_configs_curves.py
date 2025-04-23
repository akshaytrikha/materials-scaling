#!/usr/bin/env python3
"""
adit_configs_curves.py

Loads transformer‐architecture metadata and produces six plots:
  1) Piecewise power‐law fit: Model size → # heads
  2) Power‐law fit:           Model size → # layers
  3) Piecewise quadratic fit: d_model     → # heads
  4) Linear fit:              d_model     → FFW size
  5) Ratio plot:              Model size → (FFW size / d_model)

Each figure is shown interactively and saved to a PNG file.
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Data
# -----------------------------------------------------------------------------
# Format: (model_size_M, d_model, ffw_size, kv_size, n_heads, n_layers)
# kv_size column will now be ignored.
data = np.array([
    (44,   512,  2048,  64,  8,  8),
    (57,   576,  2304,  64,  9,  9),
    (74,   640,  2560,  64, 10, 10),
    (90,   640,  2560,  64, 10, 13),
    (106,  640,  2560,  64, 10, 16),
    (117,  768,  3072,  64, 12, 12),
    (140,  768,  3072,  64, 12, 15),
    (163,  768,  3072,  64, 12, 18),
    (175,  896,  3584,  64, 14, 14),
    (196,  896,  3584,  64, 14, 16),
    (217,  896,  3584,  64, 14, 18),
    (251, 1024,  4096,  64, 16, 16),
    (278, 1024,  4096,  64, 16, 18),
    (306, 1024,  4096,  64, 16, 20),
    (425, 1280,  5120, 128, 10, 18),
    (489, 1280,  5120, 128, 10, 21),
    (509, 1408,  5632, 128, 11, 18),
    (552, 1280,  5120, 128, 10, 24),
    (587, 1408,  5632, 128, 11, 21),
    (632, 1536,  6144, 128, 12, 19),
    (664, 1408,  5632, 128, 11, 24),
    (724, 1536,  6144, 128, 12, 22),
    (816, 1536,  6144, 128, 12, 25),
    (893, 1792,  7168, 128, 14, 20),
    (1018,1792,  7168, 128, 14, 23),
    (1143,1792,  7168, 128, 14, 26),
    (1266,2048,  8192, 128, 16, 22),
    (1424,2176,  8704, 128, 17, 22),
    (1429,2048,  8192, 128, 16, 25),
    (1593,2048,  8192, 128, 16, 28),
    (1609,2176,  8704, 128, 17, 25),
    (1731,2304,  9216, 128, 18, 24),
    (1794,2176,  8704, 128, 17, 28),
    (2007,2304,  9216, 128, 18, 28),
    (2283,2304,  9216, 128, 18, 32),
    (2298,2560, 10240, 128, 20, 26),
    (2639,2560, 10240, 128, 20, 30),
    (2980,2560, 10240, 128, 20, 34),
    (3530,2688, 10752, 128, 22, 36),
    (3802,2816, 11264, 128, 22, 36),
    (4084,2944, 11776, 128, 22, 36),
    (4516,3072, 12288, 128, 24, 36),
    (6796,3584, 14336, 128, 28, 40),
    (9293,4096, 16384, 128, 32, 42),
    (11452,4352,17408, 128, 32, 47),
    (12295,4608,18432,128, 36, 44),
    (12569,4608,18432,128, 32, 47),
    (13735,4864,19456,128, 32, 47),
    (14940,4992,19968,128, 32, 49),
    (16183,5120,20480,128, 40, 47)
])


model_sizes = data[:,0]
d_models    = data[:,1]
ffw_sizes   = data[:,2]
n_heads     = data[:,4]
n_layers    = data[:,5]

# -----------------------------------------------------------------------------
# 2. Piecewise helpers
# -----------------------------------------------------------------------------
def piecewise_power_law_grid(x, y, candidate_breaks, min_pts=4):
    logx, logy = np.log(x), np.log(y)
    best = {'sse': np.inf}
    for b in candidate_breaks:
        m1, m2 = x<=b, x>b
        if m1.sum()<min_pts or m2.sum()<min_pts: continue
        p1 = np.polyfit(logx[m1], logy[m1], 1)
        p2 = np.polyfit(logx[m2], logy[m2], 1)
        pred1 = np.polyval(p1, logx[m1])
        pred2 = np.polyval(p2, logx[m2])
        sse = ((logy[m1]-pred1)**2).sum() + ((logy[m2]-pred2)**2).sum()
        if sse < best['sse']:
            best.update(bp=b, p1=p1, p2=p2, sse=sse)
    b   = best['bp']
    a1, sl1 = np.exp(best['p1'][1]), best['p1'][0]
    a2, sl2 = np.exp(best['p2'][1]), best['p2'][0]
    return b, (a1, sl1), (a2, sl2)

def piecewise_quadratic_grid(x, y, breaks, min_pts=4):
    best = {'sse': np.inf}
    for b in breaks:
        m1, m2 = x<=b, x>b
        if m1.sum()<min_pts or m2.sum()<min_pts: continue
        p1, p2 = np.polyfit(x[m1], y[m1], 2), np.polyfit(x[m2], y[m2], 2)
        pred1, pred2 = np.polyval(p1, x[m1]), np.polyval(p2, x[m2])
        sse = ((y[m1]-pred1)**2).sum() + ((y[m2]-pred2)**2).sum()
        if sse<best['sse']:
            best.update(bp=b, p1=p1, p2=p2, sse=sse)
    return best['bp'], best['p1'], best['p2']

model_breaks  = np.percentile(model_sizes, np.linspace(10,90,80))
dmodel_breaks = np.percentile(d_models,    np.linspace(10,90,80))

# -----------------------------------------------------------------------------
# Plot 1: model_size → heads (piecewise power law)
# -----------------------------------------------------------------------------
bp_h, (A1_h,B1_h),(A2_h,B2_h) = piecewise_power_law_grid(model_sizes, n_heads, model_breaks)
xs1 = np.linspace(model_sizes.min(), bp_h, 300)
xs2 = np.linspace(bp_h, model_sizes.max(), 300)
ys1 = A1_h * xs1**B1_h
ys2 = A2_h * xs2**B2_h

plt.figure(figsize=(10,6))
plt.scatter(model_sizes, n_heads, color='C0', s=60, label='Data')
plt.plot(xs1, ys1, 'r-', label=f'≤{bp_h:.0f}M: {A1_h:.2f}·x^{B1_h:.3f}')
plt.plot(xs2, ys2, 'r-', label=f'>{bp_h:.0f}M: {A2_h:.2f}·x^{B2_h:.3f}')
plt.axvline(bp_h, linestyle='--', color='k', label=f'Break ≈{bp_h:.0f}M')
plt.title('Model Size vs # Heads')
plt.xlabel('Model size (M params)')
plt.ylabel('Heads')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('1_model_size_vs_heads.png')

# -----------------------------------------------------------------------------
# Plot 2: model_size → layers (power law)
# -----------------------------------------------------------------------------
slope_L, inter_L = np.polyfit(np.log(model_sizes), np.log(n_layers), 1)
A_L, B_L = np.exp(inter_L), slope_L
xs = np.linspace(model_sizes.min(), model_sizes.max(), 300)
ys = A_L * xs**B_L

plt.figure(figsize=(10,6))
plt.scatter(model_sizes, n_layers, color='C1', s=60, label='Data')
plt.plot(xs, ys, 'r-', label=f'y={A_L:.4f}·x^{B_L:.3f}')
plt.title('Model Size vs # Layers')
plt.xlabel('Model size (M params)')
plt.ylabel('Layers')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('2_model_size_vs_layers.png')

# -----------------------------------------------------------------------------
# Plot 3: d_model → heads (piecewise quadratic)
# -----------------------------------------------------------------------------
bp_q, q1, q2 = piecewise_quadratic_grid(d_models, n_heads, dmodel_breaks)
a1_q,b1_q,c1_q = q1; a2_q,b2_q,c2_q = q2
xs1 = np.linspace(d_models.min(), bp_q, 300)
xs2 = np.linspace(bp_q, d_models.max(), 300)
plt.figure(figsize=(10,6))
plt.scatter(d_models, n_heads, color='purple', s=60, label='Data')
plt.plot(xs1, np.polyval(q1, xs1), 'r-', label=f'≤{bp_q:.0f}: {a1_q:.2e}x²+{b1_q:.3f}x+{c1_q:.2f}')
plt.plot(xs2, np.polyval(q2, xs2), 'r-', label=f'>{bp_q:.0f}: {a2_q:.2e}x²+{b2_q:.3f}x+{c2_q:.2f}')
plt.axvline(bp_q, linestyle='--', color='k', label=f'Break ≈{bp_q:.0f}')
plt.title('d_model vs # Heads')
plt.xlabel('d_model'); plt.ylabel('Heads')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('3_dmodel_vs_heads.png')

# -----------------------------------------------------------------------------
# Plot 4: d_model → FFW size (linear)
# -----------------------------------------------------------------------------
m_ffw, c_ffw = np.polyfit(d_models, ffw_sizes, 1)
xs = np.linspace(d_models.min(), d_models.max(), 300)
plt.figure(figsize=(10,6))
plt.scatter(d_models, ffw_sizes, color='teal', s=60, label='Data')
plt.plot(xs, m_ffw*xs + c_ffw, 'r-', label=f'y={m_ffw:.4f}x+{c_ffw:.1f}')
plt.title('d_model vs FFW size')
plt.xlabel('d_model'); plt.ylabel('FFW')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('5_dmodel_vs_ffw.png')

# -----------------------------------------------------------------------------
# Plot 5: model_size → (FFW / d_model) ratio
# -----------------------------------------------------------------------------
ratio = ffw_sizes / d_models
plt.figure(figsize=(10,6))
plt.scatter(model_sizes, ratio, color='magenta', s=60, label='Data')
plt.axhline(4.0, linestyle='-', label='4×')
plt.title('FFW/d_model Ratio')
plt.xlabel('Model size (M params)'); plt.ylabel('FFW/d_model')
plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig('6_ratio.png')

# -----------------------------------------------------------------------------
# Show all
# -----------------------------------------------------------------------------
plt.show()