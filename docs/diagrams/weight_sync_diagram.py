"""
Weight Synchronization Architecture Diagram
EKS cluster: 2x p4d.24xlarge (GPU) + 2x c7i (CPU)
Shows RDMA-based weight transfer from Trainer -> Generator
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(24, 16))
ax.set_xlim(0, 24)
ax.set_ylim(0, 16)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('#0d1117')

# ─── Colors ──────────────────────────────────
C_EKS_BG      = '#161b22'
C_EKS_BORDER  = '#30363d'
C_P4D_BG      = '#1a1f2e'
C_P4D_BORDER  = '#58a6ff'
C_C7I_BG      = '#1a2e1f'
C_C7I_BORDER  = '#3fb950'
C_GPU_BG      = '#2d1f3d'
C_GPU_BORDER  = '#bc8cff'
C_CPU_BG      = '#1f2d3d'
C_CPU_BORDER  = '#58a6ff'
C_INTERNAL    = '#8b949e'
C_TEXT_MAIN   = '#e6edf3'
C_TEXT_DIM    = '#8b949e'
C_TEXT_BRIGHT = '#ffffff'
C_VLLM_BG    = '#2d3d1f'
C_VLLM_BORDER = '#d2a8ff'
C_FSDP_BG    = '#3d2d1f'
C_FSDP_BORDER = '#f0883e'
C_STEP_BG    = '#21262d'
C_EFA_BG     = '#2d1a1a'
C_EFA_BORDER = '#f97316'

def box(x, y, w, h, fc, ec, lw=1.5, alpha=0.85, zorder=1, r=0.15):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       alpha=alpha, zorder=zorder)
    ax.add_patch(p)

def txt(x, y, s, sz=10, c=C_TEXT_MAIN, w='normal', ha='center', va='center', z=10):
    ax.text(x, y, s, fontsize=sz, color=c, weight=w, ha=ha, va=va, zorder=z,
            fontfamily='sans-serif')

def arr(x1, y1, x2, y2, c, lw=2, cs='arc3,rad=0', ms=15):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', color=c,
                        linewidth=lw, zorder=5, connectionstyle=cs, mutation_scale=ms)
    ax.add_patch(a)


# ═══════════════════════════════════════════
# EKS CLUSTER BOUNDARY
# ═══════════════════════════════════════════
box(0.3, 0.3, 23.4, 15.4, C_EKS_BG, C_EKS_BORDER, lw=2.5, r=0.3)
txt(12, 15.45, 'Amazon EKS Cluster', sz=16, c=C_TEXT_BRIGHT, w='bold')

# ═══════════════════════════════════════════
# STEP SEQUENCE — top-right panel (no overlap)
# ═══════════════════════════════════════════
box(15.3, 13.1, 8.0, 2.0, C_STEP_BG, C_EKS_BORDER, lw=1.5, r=0.15)
txt(19.3, 14.8, 'Weight Sync Flow (every N steps)', sz=9.5, c=C_TEXT_BRIGHT, w='bold')

steps = [
    ('1.', 'Trainer updates weights via FSDP2 backward pass'),
    ('2.', 'Controller calls refresh_shards() on each rank'),
    ('3.', 'GPU weights copied to CPU RDMA buffers (~100ms)'),
    ('4.', 'Controller triggers sync_weights_rdma(step)'),
    ('5.', 'Generator RDMA-reads 8 shards in parallel via EFA'),
    ('6.', 'Shards concatenated -> vLLM reload_weights()'),
]
for i, (n, d) in enumerate(steps):
    y = 14.4 - i * 0.22
    txt(15.5, y, n, sz=8, c=C_EFA_BORDER, ha='left', w='bold')
    txt(15.95, y, d, sz=7.5, c=C_TEXT_DIM, ha='left')

# ═══════════════════════════════════════════
# LEGEND — top-left panel
# ═══════════════════════════════════════════
box(0.7, 13.1, 5.0, 2.0, C_STEP_BG, C_EKS_BORDER, lw=1.5, r=0.15)
txt(3.2, 14.8, 'Legend', sz=9.5, c=C_TEXT_BRIGHT, w='bold')

legend = [
    (C_P4D_BORDER, 'p4d.24xlarge (8x A100 80GB, EFA)'),
    (C_C7I_BORDER, 'c7i (CPU-only, control plane)'),
    (C_EFA_BORDER, 'EFA / RDMA data path (400 Gbps)'),
    (C_INTERNAL,   'Control / monitoring flow'),
]
for i, (clr, desc) in enumerate(legend):
    y = 14.35 - i * 0.32
    box(0.95, y - 0.1, 0.35, 0.2, clr, clr, lw=1, r=0.05, alpha=0.9)
    txt(1.55, y, desc, sz=7.5, c=C_TEXT_DIM, ha='left')

# ═══════════════════════════════════════════
# NODE 1: p4d.24xlarge — LEARNER
# ═══════════════════════════════════════════
LX, LY, LW, LH = 0.7, 4.2, 10.0, 8.6
box(LX, LY, LW, LH, C_P4D_BG, C_P4D_BORDER, lw=2, r=0.25)
txt(LX + LW/2, LY + LH - 0.35, 'p4d.24xlarge  --  Learner Node', sz=13, c=C_P4D_BORDER, w='bold')

# FSDP2 training zone
box(1.0, 7.8, 9.4, 4.6, C_FSDP_BG, C_FSDP_BORDER, lw=1.5, r=0.2)
txt(5.7, 12.0, 'FSDP2 Training (Composable)', sz=10, c=C_FSDP_BORDER, w='bold')

# 8 GPUs - 2 rows x 4
for i in range(4):
    gx = 1.3 + i * 2.25
    box(gx, 10.1, 1.9, 1.5, C_GPU_BG, C_GPU_BORDER, lw=1, r=0.12)
    txt(gx + 0.95, 10.85, f'GPU {i}\nShard {i}', sz=8, c=C_GPU_BORDER)

for i in range(4):
    gx = 1.3 + i * 2.25
    box(gx, 8.1, 1.9, 1.5, C_GPU_BG, C_GPU_BORDER, lw=1, r=0.12)
    txt(gx + 0.95, 8.85, f'GPU {i+4}\nShard {i+4}', sz=8, c=C_GPU_BORDER)

# CPU RDMA Buffer zone
box(1.0, 4.5, 9.4, 2.9, C_CPU_BG, C_CPU_BORDER, lw=1.5, r=0.2)
txt(5.7, 7.05, 'CPU RDMA Buffers (expose_shards)', sz=9.5, c=C_CPU_BORDER, w='bold')

for i in range(8):
    bx = 1.2 + i * 1.14
    box(bx, 4.8, 0.95, 1.85, '#1a2640', C_CPU_BORDER, lw=1, r=0.1, alpha=0.7)
    txt(bx + 0.475, 5.7, f'Buf {i}', sz=7, c=C_CPU_BORDER)

# GPU -> CPU arrows
for col in range(4):
    cx = 1.3 + col * 2.25 + 0.95
    arr(cx, 10.1, cx - 0.7, 6.65, C_INTERNAL, lw=1)
    arr(cx, 8.1, cx + 0.4, 6.65, C_INTERNAL, lw=1)

txt(10.1, 7.6, 'refresh_shards()', sz=7, c=C_TEXT_DIM, ha='right')

# ═══════════════════════════════════════════
# NODE 2: p4d.24xlarge — GENERATOR
# ═══════════════════════════════════════════
GX, GY, GW, GH = 13.3, 4.2, 10.0, 8.6
box(GX, GY, GW, GH, C_P4D_BG, C_P4D_BORDER, lw=2, r=0.25)
txt(GX + GW/2, GY + GH - 0.35, 'p4d.24xlarge  --  Generator Node', sz=13, c=C_P4D_BORDER, w='bold')

# vLLM inference zone
box(13.6, 7.8, 9.4, 4.6, C_VLLM_BG, C_VLLM_BORDER, lw=1.5, r=0.2)
txt(18.3, 12.0, 'vLLM Inference Engine', sz=10, c=C_VLLM_BORDER, w='bold')

# 8 GPUs - 2 rows x 4
for i in range(4):
    gx = 13.9 + i * 2.25
    box(gx, 10.1, 1.9, 1.5, C_GPU_BG, C_GPU_BORDER, lw=1, r=0.12)
    txt(gx + 0.95, 10.85, f'GPU {i}\nTP rank {i}', sz=8, c=C_GPU_BORDER)

for i in range(4):
    gx = 13.9 + i * 2.25
    box(gx, 8.1, 1.9, 1.5, C_GPU_BG, C_GPU_BORDER, lw=1, r=0.12)
    txt(gx + 0.95, 8.85, f'GPU {i+4}\nTP rank {i+4}', sz=8, c=C_GPU_BORDER)

# Weight Reconstruction zone
box(13.6, 4.5, 9.4, 2.9, C_CPU_BG, C_VLLM_BORDER, lw=1.5, r=0.2)
txt(18.3, 7.05, 'Weight Reconstruction', sz=9.5, c=C_VLLM_BORDER, w='bold')
txt(18.3, 5.9, 'RDMA read 8 shards in parallel\nConcatenate shards per parameter\ncollective_rpc("reload_weights")', sz=8, c=C_TEXT_DIM)

# CPU -> GPU arrows
for col in range(4):
    cx = 13.9 + col * 2.25 + 0.95
    arr(cx - 0.3, 6.65, cx, 8.1, C_VLLM_BORDER, lw=1)
    arr(cx + 0.3, 6.65, cx, 10.1, C_VLLM_BORDER, lw=1)

txt(13.9, 7.6, 'reload_weights()', sz=7, c=C_TEXT_DIM, ha='left')

# ═══════════════════════════════════════════
# EFA / RDMA BRIDGE (center)
# ═══════════════════════════════════════════
box(10.4, 4.8, 2.8, 2.6, C_EFA_BG, C_EFA_BORDER, lw=2.5, r=0.15, alpha=0.95)
txt(11.8, 6.5, 'EFA / RDMA', sz=11, c=C_EFA_BORDER, w='bold')
txt(11.8, 5.9, '400 Gbps', sz=9, c=C_TEXT_DIM)
txt(11.8, 5.4, '8 parallel reads', sz=8, c=C_TEXT_DIM)

# Thick arrows: Learner -> EFA -> Generator
arr(10.2, 6.1, 10.5, 6.1, C_EFA_BORDER, lw=3)
arr(13.1, 6.1, 13.4, 6.1, C_EFA_BORDER, lw=3)

# ═══════════════════════════════════════════
# c7i #1 — Controller Node
# ═══════════════════════════════════════════
box(2.0, 0.7, 7.0, 3.0, C_C7I_BG, C_C7I_BORDER, lw=2, r=0.25)
txt(5.5, 3.3, 'c7i  --  Controller Node', sz=12, c=C_C7I_BORDER, w='bold')

box(2.3, 1.0, 3.0, 1.9, '#1a3322', C_C7I_BORDER, lw=1, r=0.12)
txt(3.8, 2.3, 'Monarch\nController', sz=9, c=C_TEXT_MAIN, w='bold')
txt(3.8, 1.5, 'Orchestrates\ntrain loop', sz=7.5, c=C_TEXT_DIM)

box(5.7, 1.0, 3.0, 1.9, '#1a3322', C_C7I_BORDER, lw=1, r=0.12)
txt(7.2, 2.3, 'Mesh\nCoordinator', sz=9, c=C_TEXT_MAIN, w='bold')
txt(7.2, 1.5, 'proc_mesh\nmanagement', sz=7.5, c=C_TEXT_DIM)

# ═══════════════════════════════════════════
# c7i #2 — Monitoring Node
# ═══════════════════════════════════════════
box(15.0, 0.7, 7.0, 3.0, C_C7I_BG, C_C7I_BORDER, lw=2, r=0.25)
txt(18.5, 3.3, 'c7i  --  Monitoring Node', sz=12, c=C_C7I_BORDER, w='bold')

box(15.3, 1.0, 3.0, 1.9, '#1a3322', C_C7I_BORDER, lw=1, r=0.12)
txt(16.8, 2.3, 'W&B\nLogger', sz=9, c=C_TEXT_MAIN, w='bold')
txt(16.8, 1.5, 'Metrics &\nrewards', sz=7.5, c=C_TEXT_DIM)

box(18.7, 1.0, 3.0, 1.9, '#1a3322', C_C7I_BORDER, lw=1, r=0.12)
txt(20.2, 2.3, 'FSx Lustre\nClient', sz=9, c=C_TEXT_MAIN, w='bold')
txt(20.2, 1.5, 'Checkpoint\nfallback', sz=7.5, c=C_TEXT_DIM)

# ═══════════════════════════════════════════
# CONTROL FLOW ARROWS
# ═══════════════════════════════════════════
# Controller -> Learner
arr(3.8, 3.7, 3.8, 4.2, C_C7I_BORDER, lw=1.5)
txt(2.6, 3.95, 'refresh_shards()', sz=7, c=C_C7I_BORDER)

# Controller -> Generator
arr(7.2, 3.7, 15.5, 4.2, C_C7I_BORDER, lw=1.5, cs='arc3,rad=-0.12')
txt(11.5, 3.5, 'sync_weights_rdma(step)', sz=7.5, c=C_C7I_BORDER)

# Monitoring -> both nodes (thin)
arr(16.8, 2.9, 15.5, 4.2, C_INTERNAL, lw=0.7)
arr(16.8, 2.9, 8.5, 4.2, C_INTERNAL, lw=0.7, cs='arc3,rad=0.15')

# ═══════════════════════════════════════════
# SAVE
# ═══════════════════════════════════════════
fig.savefig('/home/ubuntu/rl_grpo_qwen_math/docs/diagrams/weight_sync_architecture.png',
            dpi=200, bbox_inches='tight', facecolor='#0d1117', edgecolor='none')
print("Saved to docs/diagrams/weight_sync_architecture.png")
