"""
Diagram 2: Monarch Inter-Node Weight Sync
Separate Learner and Generator nodes connected via EFA/RDMA.
8 parallel RDMA reads pull FSDP shards from Learner CPU buffers.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(22, 14))
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('#0d1117')

# ─── Colors ──────────────────────────────────
BG       = '#161b22'
BORDER   = '#30363d'
P4D_BG   = '#1a1f2e'
P4D_EC   = '#58a6ff'
GPU_BG   = '#2d1f3d'
GPU_EC   = '#bc8cff'
CPU_BG   = '#1f2d3d'
CPU_EC   = '#58a6ff'
FSDP_BG  = '#3d2d1f'
FSDP_EC  = '#f0883e'
VLLM_BG  = '#1f3d2d'
VLLM_EC  = '#3fb950'
EFA_BG   = '#2d1a1a'
EFA_EC   = '#f97316'
RECON_BG = '#1f2d3d'
RECON_EC = '#d2a8ff'
TXT      = '#e6edf3'
DIM      = '#8b949e'
WHITE    = '#ffffff'
STEP_BG  = '#21262d'

def box(x, y, w, h, fc, ec, lw=1.5, alpha=0.85, z=1, r=0.15):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha, zorder=z))

def txt(x, y, s, sz=10, c=TXT, w='normal', ha='center', va='center', z=10):
    ax.text(x, y, s, fontsize=sz, color=c, weight=w, ha=ha, va=va, zorder=z,
            fontfamily='sans-serif')

def arr(x1, y1, x2, y2, c, lw=2, cs='arc3,rad=0', ms=15, z=5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='->', color=c,
        linewidth=lw, zorder=z, connectionstyle=cs, mutation_scale=ms))

# ═══════════════════════════════════════════
# TITLE
# ═══════════════════════════════════════════
txt(11, 13.6, 'Monarch: Inter-Node RDMA Weight Sync', sz=16, c=WHITE, w='bold')
txt(11, 13.15, 'Separate Learner + Generator nodes. 8 parallel RDMA reads over 400 Gbps EFA.', sz=9, c=DIM)

# ═══════════════════════════════════════════
# NODE 1: LEARNER (left)
# ═══════════════════════════════════════════
box(0.5, 1.8, 8.5, 10.8, P4D_BG, P4D_EC, lw=2.5, r=0.25)
txt(4.75, 12.2, 'Node 0  --  Learner', sz=13, c=P4D_EC, w='bold')
txt(4.75, 11.7, 'p4d.24xlarge (8x A100)', sz=9, c=DIM)

# FSDP2 training zone
box(0.8, 7.0, 7.9, 4.3, FSDP_BG, FSDP_EC, lw=1.5, r=0.2)
txt(4.75, 10.9, 'FSDP2 Training', sz=10, c=FSDP_EC, w='bold')

# GPU grid 2x4
for i in range(4):
    gx = 1.1 + i * 1.9
    box(gx, 9.2, 1.6, 1.3, GPU_BG, GPU_EC, lw=1, r=0.1)
    txt(gx + 0.8, 9.85, f'GPU {i}\nRank {i}', sz=7.5, c=GPU_EC)

for i in range(4):
    gx = 1.1 + i * 1.9
    box(gx, 7.4, 1.6, 1.3, GPU_BG, GPU_EC, lw=1, r=0.1)
    txt(gx + 0.8, 8.05, f'GPU {i+4}\nRank {i+4}', sz=7.5, c=GPU_EC)

# Step 2: GPU -> CPU copy
box(0.8, 4.1, 7.9, 2.5, CPU_BG, CPU_EC, lw=1.5, r=0.2)
txt(4.75, 6.25, 'CPU RDMA Buffers', sz=10, c=CPU_EC, w='bold')
txt(4.75, 5.8, 'ibverbs-registered, pinned memory', sz=7.5, c=DIM)

# 8 buffer slots
for i in range(8):
    bx = 1.0 + i * 0.95
    box(bx, 4.35, 0.75, 1.15, '#1a2640', CPU_EC, lw=1, r=0.08, alpha=0.7)
    txt(bx + 0.375, 4.92, f'{i}', sz=8, c=CPU_EC, w='bold')

# GPU->CPU arrows (one per column, representing copy)
for col in range(4):
    cx = 1.1 + col * 1.9 + 0.8
    arr(cx, 7.4, cx - 0.5, 5.5, DIM, lw=1, ms=10)
    arr(cx, 7.4, cx + 0.35, 5.5, DIM, lw=1, ms=10)

# Step labels on learner
box(0.8, 2.1, 7.9, 1.6, STEP_BG, FSDP_EC, lw=1, r=0.12)
txt(4.75, 3.35, 'Learner Steps', sz=9, c=FSDP_EC, w='bold')
txt(4.75, 2.7, '1. FSDP2 backward pass updates GPU shards\n2. refresh_shards(): GPU -> CPU copy (~100ms)\n   Each rank copies its shard independently', sz=7.5, c=DIM)

# ═══════════════════════════════════════════
# NODE 2: GENERATOR (right)
# ═══════════════════════════════════════════
box(13.0, 1.8, 8.5, 10.8, P4D_BG, P4D_EC, lw=2.5, r=0.25)
txt(17.25, 12.2, 'Node 1  --  Generator', sz=13, c=P4D_EC, w='bold')
txt(17.25, 11.7, 'p4d.24xlarge (8x A100)', sz=9, c=DIM)

# vLLM inference zone
box(13.3, 7.0, 7.9, 4.3, VLLM_BG, VLLM_EC, lw=1.5, r=0.2)
txt(17.25, 10.9, 'vLLM Inference', sz=10, c=VLLM_EC, w='bold')

for i in range(4):
    gx = 13.6 + i * 1.9
    box(gx, 9.2, 1.6, 1.3, GPU_BG, GPU_EC, lw=1, r=0.1)
    txt(gx + 0.8, 9.85, f'GPU {i}\nTP {i}', sz=7.5, c=GPU_EC)

for i in range(4):
    gx = 13.6 + i * 1.9
    box(gx, 7.4, 1.6, 1.3, GPU_BG, GPU_EC, lw=1, r=0.1)
    txt(gx + 0.8, 8.05, f'GPU {i+4}\nTP {i+4}', sz=7.5, c=GPU_EC)

# Weight Reconstruction zone
box(13.3, 4.1, 7.9, 2.5, RECON_BG, RECON_EC, lw=1.5, r=0.2)
txt(17.25, 6.25, 'Weight Reconstruction', sz=10, c=RECON_EC, w='bold')
txt(17.25, 5.3, 'Concatenate 8 shards per param\nReshape to full tensor\ncollective_rpc("reload_weights")', sz=7.5, c=DIM)

# Reconstruction -> GPU arrows
for col in range(4):
    cx = 13.6 + col * 1.9 + 0.8
    arr(cx - 0.3, 6.6, cx, 7.4, RECON_EC, lw=1, ms=10)
    arr(cx + 0.3, 6.6, cx, 9.2, RECON_EC, lw=1, ms=10)

# Step labels on generator
box(13.3, 2.1, 7.9, 1.6, STEP_BG, VLLM_EC, lw=1, r=0.12)
txt(17.25, 3.35, 'Generator Steps', sz=9, c=VLLM_EC, w='bold')
txt(17.25, 2.7, '3. RDMA read 8 shards in parallel (~35ms)\n4. Reconstruct full state_dict (~200ms)\n5. Load into vLLM via reload_weights (~50ms)', sz=7.5, c=DIM)

# ═══════════════════════════════════════════
# EFA / RDMA BRIDGE (center)
# ═══════════════════════════════════════════
box(9.3, 3.5, 3.4, 4.5, EFA_BG, EFA_EC, lw=2.5, r=0.2, alpha=0.95)
txt(11.0, 7.3, 'EFA / RDMA', sz=12, c=EFA_EC, w='bold')
txt(11.0, 6.75, '400 Gbps', sz=10, c=WHITE, w='bold')

# 8 parallel read lines
for i in range(8):
    y = 4.0 + i * 0.35
    ax.plot([9.5, 12.5], [y, y], color=EFA_EC, linewidth=0.8, alpha=0.5, zorder=3)

txt(11.0, 5.7, '8 parallel\nRDMA reads', sz=9, c=DIM)
txt(11.0, 4.5, '~375 MB each\n(1.5B model)', sz=8, c=DIM)

# Big arrows: Learner -> EFA -> Generator
arr(8.7, 5.7, 9.4, 5.7, EFA_EC, lw=3, ms=18)
arr(12.6, 5.7, 13.1, 5.7, EFA_EC, lw=3, ms=18)

# Direction label
txt(11.0, 8.3, 'Generator pulls from Learner', sz=8, c=EFA_EC, w='bold')
arr(11.0, 8.05, 12.5, 7.2, EFA_EC, lw=1, cs='arc3,rad=-0.2', ms=10)

# ═══════════════════════════════════════════
# TIMING SUMMARY (bottom)
# ═══════════════════════════════════════════
box(3.0, 0.3, 16.0, 1.2, STEP_BG, BORDER, lw=1.5, r=0.15)
txt(11.0, 1.1, 'Total Weight Sync: ~385ms', sz=11, c=WHITE, w='bold')
txt(11.0, 0.65, 'refresh_shards (~100ms)  +  RDMA read (~35ms)  +  reconstruct (~200ms)  +  vLLM load (~50ms)', sz=8, c=DIM)

fig.savefig('/home/ubuntu/rl_grpo_qwen_math/docs/diagrams/monarch_weight_sync.png',
            dpi=200, bbox_inches='tight', facecolor='#0d1117', edgecolor='none')
print("Saved monarch_weight_sync.png")
