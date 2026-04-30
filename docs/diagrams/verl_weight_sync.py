"""
Diagram 1: veRL In-Place Weight Sync
Colocated generation + training on the same GPU pool.
No network transfer — weights are resharded in-place via DTensor.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(20, 13))
ax.set_xlim(0, 20)
ax.set_ylim(0, 13)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('#0d1117')

# ─── Colors ──────────────────────────────────
BG       = '#161b22'
BORDER   = '#30363d'
GPU_BG   = '#2d1f3d'
GPU_EC   = '#bc8cff'
FSDP_BG  = '#3d2d1f'
FSDP_EC  = '#f0883e'
VLLM_BG  = '#1f3d2d'
VLLM_EC  = '#3fb950'
DTENSOR  = '#58a6ff'
SLEEP_BG = '#1f2d3d'
SLEEP_EC = '#79c0ff'
PHASE_BG = '#21262d'
PHASE_EC = '#484f58'
TXT      = '#e6edf3'
DIM      = '#8b949e'
WHITE    = '#ffffff'
ORANGE   = '#f97316'
GREEN    = '#3fb950'
PURPLE   = '#d2a8ff'

def box(x, y, w, h, fc, ec, lw=1.5, alpha=0.85, z=1, r=0.15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=z))

def txt(x, y, s, sz=10, c=TXT, w='normal', ha='center', va='center', z=10):
    ax.text(x, y, s, fontsize=sz, color=c, weight=w, ha=ha, va=va, zorder=z,
            fontfamily='sans-serif')

def arr(x1, y1, x2, y2, c, lw=2, cs='arc3,rad=0', ms=15, z=5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', color=c,
        linewidth=lw, zorder=z, connectionstyle=cs, mutation_scale=ms))

def darr(x1, y1, x2, y2, c, lw=1.5, cs='arc3,rad=0', z=5):
    """Double-headed arrow"""
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='<->', color=c,
        linewidth=lw, zorder=z, connectionstyle=cs, mutation_scale=12))

# ═══════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════
txt(10, 12.6, 'veRL: In-Place Weight Sync (Colocated)', sz=16, c=WHITE, w='bold')
txt(10, 12.15, 'Same GPU pool for generation + training. Zero network transfer.', sz=9, c=DIM)

# ═══════════════════════════════════════════
# SHARED GPU POOL (the big box)
# ═══════════════════════════════════════════
box(0.5, 2.8, 19.0, 8.8, BG, DTENSOR, lw=2.5, r=0.3)
txt(10, 11.25, 'Shared GPU Pool  --  Same 16x A100 GPUs (2 pods x 8 GPUs)', sz=11, c=DTENSOR, w='bold')

# ═══════════════════════════════════════════
# PHASE 1: GENERATION (left side, top)
# ═══════════════════════════════════════════
box(0.9, 7.0, 8.5, 3.8, VLLM_BG, VLLM_EC, lw=2, r=0.2)
txt(5.15, 10.45, 'Phase 1: Generation (~19s)', sz=11, c=VLLM_EC, w='bold')
txt(5.15, 9.95, 'vLLM Inference Engine', sz=9, c=DIM)

# 8 GPUs in 2 rows, showing TP sharding
for i in range(4):
    gx = 1.2 + i * 2.0
    box(gx, 8.6, 1.7, 1.0, GPU_BG, GPU_EC, lw=1, r=0.1)
    txt(gx + 0.85, 9.1, f'GPU {i}\nTP shard', sz=7, c=GPU_EC)

for i in range(4):
    gx = 1.2 + i * 2.0
    box(gx, 7.3, 1.7, 1.0, GPU_BG, GPU_EC, lw=1, r=0.1)
    txt(gx + 0.85, 7.8, f'GPU {i+4}\nTP shard', sz=7, c=GPU_EC)

# ═══════════════════════════════════════════
# PHASE 3: TRAINING (right side, top)
# ═══════════════════════════════════════════
box(10.6, 7.0, 8.5, 3.8, FSDP_BG, FSDP_EC, lw=2, r=0.2)
txt(14.85, 10.45, 'Phase 3: Training (~39s)', sz=11, c=FSDP_EC, w='bold')
txt(14.85, 9.95, 'FSDP2 Policy Update', sz=9, c=DIM)

for i in range(4):
    gx = 10.9 + i * 2.0
    box(gx, 8.6, 1.7, 1.0, GPU_BG, GPU_EC, lw=1, r=0.1)
    txt(gx + 0.85, 9.1, f'GPU {i}\nFSDP shard', sz=7, c=GPU_EC)

for i in range(4):
    gx = 10.9 + i * 2.0
    box(gx, 7.3, 1.7, 1.0, GPU_BG, GPU_EC, lw=1, r=0.1)
    txt(gx + 0.85, 7.8, f'GPU {i+8}\nFSDP shard', sz=7, c=GPU_EC)

# ═══════════════════════════════════════════
# DTensor RESHARDING (center, between phases)
# ═══════════════════════════════════════════
box(5.5, 3.2, 9.0, 3.3, SLEEP_BG, DTENSOR, lw=2, r=0.2)
txt(10.0, 6.15, 'DTensor In-Place Resharding', sz=11, c=DTENSOR, w='bold')

# Three sub-steps
box(5.8, 4.7, 2.6, 1.1, PHASE_BG, SLEEP_EC, lw=1, r=0.1)
txt(7.1, 5.5, 'vLLM Sleep', sz=9, c=SLEEP_EC, w='bold')
txt(7.1, 5.0, 'KV cache\noffload to CPU', sz=7, c=DIM)

box(8.7, 4.7, 2.6, 1.1, PHASE_BG, DTENSOR, lw=1, r=0.1)
txt(10.0, 5.5, 'Reshard', sz=9, c=DTENSOR, w='bold')
txt(10.0, 5.0, 'TP placement\n-> FSDP placement', sz=7, c=DIM)

box(11.6, 4.7, 2.6, 1.1, PHASE_BG, GREEN, lw=1, r=0.1)
txt(12.9, 5.5, 'vLLM Wake', sz=9, c=GREEN, w='bold')
txt(12.9, 5.0, 'KV cache\nreload to GPU', sz=7, c=DIM)

# Timing label
txt(10.0, 3.65, 'Zero network transfer - weights stay on GPU, only placement changes', sz=8, c=ORANGE, w='bold')

# Arrows between sub-steps
arr(8.4, 5.25, 8.7, 5.25, DIM, lw=1.5)
arr(11.3, 5.25, 11.6, 5.25, DIM, lw=1.5)

# Arrows from phases to resharding
arr(5.15, 7.0, 7.1, 6.5, VLLM_EC, lw=1.5, cs='arc3,rad=0.1')
arr(14.85, 7.0, 12.9, 6.5, FSDP_EC, lw=1.5, cs='arc3,rad=-0.1')

# Circular flow arrows (showing the loop)
arr(7.1, 6.5, 5.15, 7.0, DTENSOR, lw=1.5, cs='arc3,rad=0.15')
arr(12.9, 6.5, 14.85, 7.0, DTENSOR, lw=1.5, cs='arc3,rad=-0.15')

# ═══════════════════════════════════════════
# PHASE SEQUENCE (bottom)
# ═══════════════════════════════════════════
box(0.5, 0.3, 19.0, 2.0, PHASE_BG, BORDER, lw=1.5, r=0.2)
txt(10, 2.0, 'Training Loop Timeline (per step ~70s)', sz=10, c=WHITE, w='bold')

# Phase boxes in sequence
phases = [
    (1.0, 'Generation\n~19s', VLLM_EC, 3.5),
    (5.0, 'Reshard\n(in-place)', DTENSOR, 2.0),
    (7.5, 'Old LogProbs\n~5.5s', '#79c0ff', 2.5),
    (10.5, 'Ref LogProbs\n~5s', '#79c0ff', 2.5),
    (13.5, 'Advantage\n~0.05s', PURPLE, 1.5),
    (15.5, 'Training\n~39s', FSDP_EC, 3.5),
]

for px, plabel, pcolor, pw in phases:
    box(px, 0.6, pw, 1.0, BG, pcolor, lw=1.5, r=0.1)
    txt(px + pw/2, 1.1, plabel, sz=7.5, c=pcolor, w='bold')

# Arrows between phases
phase_edges = [(4.5, 5.0), (7.0, 7.5), (10.0, 10.5), (13.0, 13.5), (15.0, 15.5)]
for x1, x2 in phase_edges:
    arr(x1, 1.1, x2, 1.1, DIM, lw=1)

# ═══════════════════════════════════════════
# KEY INSIGHT callout
# ═══════════════════════════════════════════
box(14.5, 3.2, 5.0, 1.6, '#1a1a2d', ORANGE, lw=1.5, r=0.12)
txt(17.0, 4.5, 'Key Insight', sz=9, c=ORANGE, w='bold')
txt(17.0, 3.85, 'Weight sync = 0 seconds\nSame GPUs, just reshard\nTP -> FSDP -> TP', sz=7.5, c=DIM)

box(0.9, 3.2, 4.2, 1.6, '#1a1a2d', GREEN, lw=1.5, r=0.12)
txt(3.0, 4.5, 'Sleep/Wake', sz=9, c=GREEN, w='bold')
txt(3.0, 3.85, 'vLLM offloads KV cache\nto CPU during training\nfrees GPU for FSDP', sz=7.5, c=DIM)

fig.savefig('/home/ubuntu/rl_grpo_qwen_math/docs/diagrams/verl_weight_sync.png',
            dpi=200, bbox_inches='tight', facecolor='#0d1117', edgecolor='none')
print("Saved verl_weight_sync.png")
