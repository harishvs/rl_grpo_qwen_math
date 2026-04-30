"""
Diagram 3: Weight Sync Across RL Training Steps
Abstract view (no EKS/p4d) showing how weights flow between
Generator and Trainer across multiple GRPO steps.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(22, 12))
ax.set_xlim(0, 22)
ax.set_ylim(0, 12)
ax.set_aspect('equal')
ax.axis('off')
fig.patch.set_facecolor('#0d1117')

# ─── Colors ──────────────────────────────────
BG       = '#161b22'
BORDER   = '#30363d'
GEN_BG   = '#1f3d2d'
GEN_EC   = '#3fb950'
TRAIN_BG = '#3d2d1f'
TRAIN_EC = '#f0883e'
SYNC_BG  = '#1f2d3d'
SYNC_EC  = '#58a6ff'
REWARD_BG = '#2d1f3d'
REWARD_EC = '#d2a8ff'
WEIGHT_C = '#f97316'
DATA_C   = '#79c0ff'
TXT      = '#e6edf3'
DIM      = '#8b949e'
WHITE    = '#ffffff'
STEP_BG  = '#21262d'
ADVAN_BG = '#1a2640'
ADVAN_EC = '#58a6ff'

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
txt(11, 11.6, 'GRPO Training: Weight Sync Across RL Steps', sz=16, c=WHITE, w='bold')
txt(11, 11.15, 'How weights flow between Generator and Trainer over multiple training iterations', sz=9, c=DIM)

# ═══════════════════════════════════════════
# STEP 1
# ═══════════════════════════════════════════
SX1 = 0.5
txt(SX1 + 3.25, 10.55, 'Step 1', sz=12, c=WHITE, w='bold')

# Generate
box(SX1, 8.5, 2.8, 1.6, GEN_BG, GEN_EC, lw=1.5, r=0.12)
txt(SX1 + 1.4, 9.6, 'Generate', sz=9, c=GEN_EC, w='bold')
txt(SX1 + 1.4, 9.05, 'vLLM produces\ncompletions', sz=7, c=DIM)

# Reward
box(SX1 + 3.2, 8.5, 2.3, 1.6, REWARD_BG, REWARD_EC, lw=1.5, r=0.12)
txt(SX1 + 4.35, 9.6, 'Score', sz=9, c=REWARD_EC, w='bold')
txt(SX1 + 4.35, 9.05, 'Reward +\nAdvantage', sz=7, c=DIM)

# Train
box(SX1, 6.2, 5.5, 1.8, TRAIN_BG, TRAIN_EC, lw=1.5, r=0.12)
txt(SX1 + 2.75, 7.6, 'Train (GRPO Update)', sz=9, c=TRAIN_EC, w='bold')
txt(SX1 + 2.75, 6.95, 'Policy gradient with clipped objective\nWeights: W0 -> W1', sz=7.5, c=DIM)

# Arrows within step 1
arr(SX1 + 2.8, 9.3, SX1 + 3.2, 9.3, DIM, lw=1.5, ms=12)
arr(SX1 + 2.75, 8.5, SX1 + 2.75, 8.0, DIM, lw=1.5, ms=12)

# Initial weights label
box(SX1, 10.1, 2.2, 0.5, SYNC_BG, SYNC_EC, lw=1, r=0.08)
txt(SX1 + 1.1, 10.35, 'W0 (init)', sz=8, c=SYNC_EC, w='bold')
arr(SX1 + 1.1, 10.1, SX1 + 1.4, 9.6 + 0.5, SYNC_EC, lw=1, ms=10)

# ═══════════════════════════════════════════
# WEIGHT SYNC 1->2
# ═══════════════════════════════════════════
box(6.3, 6.5, 1.8, 1.2, SYNC_BG, WEIGHT_C, lw=2, r=0.12)
txt(7.2, 7.35, 'Sync', sz=9, c=WEIGHT_C, w='bold')
txt(7.2, 6.85, 'W1', sz=8, c=WHITE, w='bold')

arr(5.5 + 0.5, 7.1, 6.3, 7.1, WEIGHT_C, lw=2, ms=14)
arr(8.1, 7.1, 8.5, 9.3, WEIGHT_C, lw=2, ms=14, cs='arc3,rad=-0.2')

# ═══════════════════════════════════════════
# STEP 2
# ═══════════════════════════════════════════
SX2 = 8.3
txt(SX2 + 3.25, 10.55, 'Step 2', sz=12, c=WHITE, w='bold')

box(SX2, 8.5, 2.8, 1.6, GEN_BG, GEN_EC, lw=1.5, r=0.12)
txt(SX2 + 1.4, 9.6, 'Generate', sz=9, c=GEN_EC, w='bold')
txt(SX2 + 1.4, 9.05, 'Now uses W1\n(updated policy)', sz=7, c=DIM)

box(SX2 + 3.2, 8.5, 2.3, 1.6, REWARD_BG, REWARD_EC, lw=1.5, r=0.12)
txt(SX2 + 4.35, 9.6, 'Score', sz=9, c=REWARD_EC, w='bold')
txt(SX2 + 4.35, 9.05, 'Better rewards\n(hopefully)', sz=7, c=DIM)

box(SX2, 6.2, 5.5, 1.8, TRAIN_BG, TRAIN_EC, lw=1.5, r=0.12)
txt(SX2 + 2.75, 7.6, 'Train (GRPO Update)', sz=9, c=TRAIN_EC, w='bold')
txt(SX2 + 2.75, 6.95, 'Policy gradient on new rollouts\nWeights: W1 -> W2', sz=7.5, c=DIM)

arr(SX2 + 2.8, 9.3, SX2 + 3.2, 9.3, DIM, lw=1.5, ms=12)
arr(SX2 + 2.75, 8.5, SX2 + 2.75, 8.0, DIM, lw=1.5, ms=12)

# ═══════════════════════════════════════════
# WEIGHT SYNC 2->3
# ═══════════════════════════════════════════
box(14.1, 6.5, 1.8, 1.2, SYNC_BG, WEIGHT_C, lw=2, r=0.12)
txt(15.0, 7.35, 'Sync', sz=9, c=WEIGHT_C, w='bold')
txt(15.0, 6.85, 'W2', sz=8, c=WHITE, w='bold')

arr(SX2 + 5.5, 7.1, 14.1, 7.1, WEIGHT_C, lw=2, ms=14)
arr(15.9, 7.1, 16.3, 9.3, WEIGHT_C, lw=2, ms=14, cs='arc3,rad=-0.2')

# ═══════════════════════════════════════════
# STEP 3
# ═══════════════════════════════════════════
SX3 = 16.1
txt(SX3 + 2.5, 10.55, 'Step N...', sz=12, c=WHITE, w='bold')

box(SX3, 8.5, 2.8, 1.6, GEN_BG, GEN_EC, lw=1.5, r=0.12)
txt(SX3 + 1.4, 9.6, 'Generate', sz=9, c=GEN_EC, w='bold')
txt(SX3 + 1.4, 9.05, 'Uses W(N-1)\nconverging...', sz=7, c=DIM)

box(SX3 + 3.2, 8.5, 2.1, 1.6, REWARD_BG, REWARD_EC, lw=1.5, r=0.12)
txt(SX3 + 4.25, 9.6, 'Score', sz=9, c=REWARD_EC, w='bold')
txt(SX3 + 4.25, 9.05, 'Higher\naccuracy', sz=7, c=DIM)

box(SX3, 6.2, 5.3, 1.8, TRAIN_BG, TRAIN_EC, lw=1.5, r=0.12)
txt(SX3 + 2.65, 7.6, 'Train (GRPO Update)', sz=9, c=TRAIN_EC, w='bold')
txt(SX3 + 2.65, 6.95, 'Continued improvement\nWeights: W(N-1) -> WN', sz=7.5, c=DIM)

arr(SX3 + 2.8, 9.3, SX3 + 3.2, 9.3, DIM, lw=1.5, ms=12)
arr(SX3 + 2.65, 8.5, SX3 + 2.65, 8.0, DIM, lw=1.5, ms=12)

# ═══════════════════════════════════════════
# BOTTOM: GRPO Detail Box
# ═══════════════════════════════════════════
box(0.5, 0.3, 10.0, 5.4, STEP_BG, BORDER, lw=1.5, r=0.2)
txt(5.5, 5.35, 'Inside Each Step (GRPO)', sz=11, c=WHITE, w='bold')

# Mini flow diagram
mini_items = [
    (0.8, 3.7, 2.5, 1.2, GEN_BG, GEN_EC, 'Generate\nK completions\nper prompt'),
    (3.7, 3.7, 2.5, 1.2, REWARD_BG, REWARD_EC, 'Compute\nRewards\n(math check)'),
    (6.6, 3.7, 3.5, 1.2, ADVAN_BG, ADVAN_EC, 'GRPO Advantage\nA(i) = (r(i) - mean) / std\n(group relative)'),
]
for mx, my, mw, mh, mfc, mec, mlabel in mini_items:
    box(mx, my, mw, mh, mfc, mec, lw=1, r=0.1)
    txt(mx + mw/2, my + mh/2, mlabel, sz=7.5, c=mec)

arr(3.3, 4.3, 3.7, 4.3, DIM, lw=1, ms=10)
arr(6.2, 4.3, 6.6, 4.3, DIM, lw=1, ms=10)

# Train sub-step
box(0.8, 1.6, 9.3, 1.7, TRAIN_BG, TRAIN_EC, lw=1, r=0.1)
txt(5.45, 2.85, 'Policy Gradient Update', sz=9, c=TRAIN_EC, w='bold')
txt(5.45, 2.2, 'Loss = -E[ min(r(t)*A, clip(r(t), 1-e, 1+e)*A) ] - beta*KL(pi||pi_ref)\n'
    'r(t) = pi_theta(a|s) / pi_old(a|s)    |    Optimized with AdamW', sz=7, c=DIM)

arr(5.5, 3.7, 5.45, 3.3, DIM, lw=1, ms=10)

# Bottom mini: output
box(0.8, 0.5, 9.3, 0.7, SYNC_BG, WEIGHT_C, lw=1, r=0.1)
txt(5.45, 0.85, 'Output: Updated weights W(t+1)  -->  Sync to Generator for next step', sz=8, c=WEIGHT_C, w='bold')
arr(5.45, 1.6, 5.45, 1.2, WEIGHT_C, lw=1.5, ms=12)

# ═══════════════════════════════════════════
# RIGHT: Weight Evolution & Accuracy
# ═══════════════════════════════════════════
box(11.0, 0.3, 10.5, 5.4, STEP_BG, BORDER, lw=1.5, r=0.2)
txt(16.25, 5.35, 'Weight Sync Comparison', sz=11, c=WHITE, w='bold')

# Three methods
methods = [
    (11.3, 3.4, 'Custom Trainer\n(HTTP)', '#f85149',
     'Serialize state_dict over HTTP\n22 seconds per sync\nSeparate pods, low throughput'),
    (11.3, 1.6, 'veRL\n(In-Place)', GEN_EC,
     'DTensor reshard on same GPUs\n~0 seconds (no transfer)\nColocated, high throughput'),
    (11.3, -0.1, 'Monarch\n(RDMA)', WEIGHT_C,
     '8 parallel RDMA reads over EFA\n~385ms per sync\nSeparate nodes, scales to 70B+'),
]

# Draw as a comparison table
box(11.3, 3.5, 3.0, 1.5, BG, '#f85149', lw=1.5, r=0.1)
txt(12.8, 4.65, 'Custom Trainer', sz=9, c='#f85149', w='bold')
txt(12.8, 4.0, 'HTTP serialization\n22s per sync', sz=7.5, c=DIM)

box(14.6, 3.5, 6.5, 1.5, '#1a1012', '#f85149', lw=1, r=0.1)
txt(17.85, 4.6, 'Trainer Pod                      Generator Pod', sz=8, c=DIM)
txt(17.85, 4.05, '---------- HTTP (22s, 3GB serialized) ---------->', sz=7.5, c='#f85149')

box(11.3, 1.7, 3.0, 1.5, BG, GEN_EC, lw=1.5, r=0.1)
txt(12.8, 2.85, 'veRL', sz=9, c=GEN_EC, w='bold')
txt(12.8, 2.2, 'In-place reshard\n0s transfer', sz=7.5, c=DIM)

box(14.6, 1.7, 6.5, 1.5, '#0a1a12', GEN_EC, lw=1, r=0.1)
txt(17.85, 2.85, '[ Same GPU Pool ]', sz=9, c=GEN_EC, w='bold')
txt(17.85, 2.2, 'DTensor: TP placement <-> FSDP placement (0ms)', sz=7.5, c=GEN_EC)

box(11.3, -0.1, 3.0, 1.5, BG, WEIGHT_C, lw=1.5, r=0.1)
txt(12.8, 1.05, 'Monarch', sz=9, c=WEIGHT_C, w='bold')
txt(12.8, 0.4, 'RDMA reads\n~385ms sync', sz=7.5, c=DIM)

box(14.6, -0.1, 6.5, 1.5, '#1a120a', WEIGHT_C, lw=1, r=0.1)
txt(17.85, 1.05, 'Learner Node                  Generator Node', sz=8, c=DIM)
txt(17.85, 0.4, '-------- EFA/RDMA (385ms, 8 parallel reads) -------->', sz=7.5, c=WEIGHT_C)

fig.savefig('/home/ubuntu/rl_grpo_qwen_math/docs/diagrams/rl_step_weight_sync.png',
            dpi=200, bbox_inches='tight', facecolor='#0d1117', edgecolor='none')
print("Saved rl_step_weight_sync.png")
