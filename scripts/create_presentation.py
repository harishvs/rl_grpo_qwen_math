#!/usr/bin/env python3
"""Generate the Monarch GRPO presentation as a clean PPTX."""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE
import os

# Colors
ORANGE = RGBColor(0xFF, 0x99, 0x00)
DARK = RGBColor(0x23, 0x27, 0x2A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xF5, 0xF5, 0xF5)
GRAY = RGBColor(0x75, 0x75, 0x75)
MED_GRAY = RGBColor(0x55, 0x55, 0x55)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


def title_bar(slide, title):
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, Inches(1.1))
    shape.fill.solid()
    shape.fill.fore_color.rgb = DARK
    shape.line.fill.background()
    txBox = slide.shapes.add_textbox(Inches(0.8), Inches(0.15), Inches(11), Inches(0.8))
    p = txBox.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(34)
    p.font.bold = True
    p.font.color.rgb = WHITE


def add_text(slide, left, top, width, height, lines, font_size=22, color=DARK, bold=False, line_spacing=1.3):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        # Handle tuples for (text, color, size, bold)
        if isinstance(line, tuple):
            p.text = line[0]
            p.font.color.rgb = line[1] if len(line) > 1 else color
            p.font.size = Pt(line[2] if len(line) > 2 else font_size)
            p.font.bold = line[3] if len(line) > 3 else bold
        else:
            p.text = line
            p.font.size = Pt(font_size)
            p.font.color.rgb = color
            p.font.bold = bold
        p.space_after = Pt(font_size * (line_spacing - 1) + 4)


def add_table(slide, left, top, width, headers, rows, font_size=16):
    n_rows = len(rows) + 1
    n_cols = len(headers)
    row_height = Inches(0.55)
    tbl = slide.shapes.add_table(n_rows, n_cols, Inches(left), Inches(top), Inches(width), row_height * n_rows).table

    for j, h in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.text = h
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(font_size)
            p.font.bold = True
            p.font.color.rgb = WHITE
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK

    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i + 1, j)
            cell.text = str(val)
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(font_size - 1)
                p.font.color.rgb = DARK
            if i % 2 == 0:
                cell.fill.solid()
                cell.fill.fore_color.rgb = LIGHT_GRAY


# ============================================================
# SLIDE 1: TITLE
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
bg = slide.background.fill
bg.solid()
bg.fore_color.rgb = DARK

add_text(slide, 1, 1.8, 11, 2, [
    ("Teaching LLMs to Solve Math", WHITE, 48, True),
    ("with Reinforcement Learning", WHITE, 48, True),
])
add_text(slide, 1, 4.2, 11, 1, [
    ("From RL Basics to Training Qwen on GSM8K", ORANGE, 24, False),
])
add_text(slide, 1, 5.5, 11, 1, [
    ("Custom Trainer  →  veRL  →  Monarch on EKS", GRAY, 20, False),
])
add_text(slide, 1, 6.3, 11, 0.5, [
    ("Harish Rao, SA  |  April 2026", GRAY, 16, False),
])


# ============================================================
# SLIDE 2: WHY RL
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
title_bar(slide, "Why Reinforcement Learning for LLMs?")

add_text(slide, 0.8, 1.5, 5.5, 5, [
    ("Supervised Learning", DARK, 26, True),
    ("Here's the question AND the answer.", MED_GRAY, 20, False),
    ("Learn to copy it.", MED_GRAY, 20, False),
    ("", DARK, 12, False),
    ("Reinforcement Learning", DARK, 26, True),
    ("Here's the question. Try something.", MED_GRAY, 20, False),
    ("I'll tell you: right or wrong.", MED_GRAY, 20, False),
    ("Figure it out.", MED_GRAY, 20, False),
])

add_text(slide, 7, 1.5, 5.5, 5, [
    ("Why this matters", ORANGE, 26, True),
    ("", DARK, 8, False),
    ("No human-written solutions needed", DARK, 22, False),
    ("Model discovers its own strategies", DARK, 22, False),
    ("Works for math, code, logic — anything checkable", DARK, 22, False),
    ("", DARK, 12, False),
    ("Our result:", DARK, 22, True),
    ("14.5%  →  87% accuracy on grade-school math", ORANGE, 26, True),
])


# ============================================================
# SLIDE 3: RL TRAINING LOOP
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
title_bar(slide, "The RL Training Loop")

steps = [
    ("1  GENERATE", "Model writes 8 attempts per problem"),
    ("2  SCORE", "Check each answer: correct = 1, wrong = 0"),
    ("3  COMPARE", "Which attempts were better than average?"),
    ("4  UPDATE", "Make good patterns more likely, bad ones less"),
]

for i, (label, desc) in enumerate(steps):
    y = 1.8 + i * 1.3
    # Step box
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(y), Inches(2.5), Inches(0.8))
    shape.fill.solid()
    shape.fill.fore_color.rgb = DARK
    shape.line.fill.background()
    p = shape.text_frame.paragraphs[0]
    p.text = label
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

    # Description
    add_text(slide, 3.8, y + 0.1, 5, 0.7, [(desc, DARK, 22, False)])

add_text(slide, 9, 2.5, 4, 3, [
    ("Repeat 233 times", ORANGE, 26, True),
    ("(one pass through GSM8K)", GRAY, 18, False),
    ("", DARK, 12, False),
    ("That's it.", DARK, 22, True),
])


# ============================================================
# SLIDE 4: KEY CONCEPTS
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
title_bar(slide, "Key RL Concepts")

concepts = [
    ("Policy", "The model's current strategy"),
    ("Reward", "Right = 1, Wrong = 0. Only feedback."),
    ("Advantage", "How much better than the group average?"),
    ("Clipping", "Limits change per step (safety rail)"),
    ("KL Penalty", "Prevents drifting too far from start"),
]

for i, (term, desc) in enumerate(concepts):
    y = 1.6 + i * 1.05
    add_text(slide, 0.8, y, 3, 0.8, [(term, ORANGE, 24, True)])
    add_text(slide, 4.2, y, 8, 0.8, [(desc, DARK, 22, False)])

    if i < len(concepts) - 1:
        shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.8), Inches(y + 0.85), Inches(11.5), Inches(0.02))
        shape.fill.solid()
        shape.fill.fore_color.rgb = LIGHT_GRAY
        shape.line.fill.background()


# ============================================================
# SLIDE 5: GRPO vs PPO
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
title_bar(slide, "GRPO vs PPO")

add_table(slide, 0.8, 1.5, 11.5,
    ["", "PPO (classic)", "GRPO (DeepSeek-R1)"],
    [
        ["Baseline", "Learned critic network", "Group average of rewards"],
        ["Extra model", "Yes (2× GPU memory)", "No"],
        ["Advantage", "Complex: GAE from critic", "Simple: (reward - mean) / std"],
        ["Attempts/problem", "1", "8 (group comparison)"],
        ["Best for", "General RL", "Verifiable rewards (math, code)"],
    ],
    font_size=17
)

add_text(slide, 0.8, 5.5, 11, 1.5, [
    ("GRPO trades more generation for simpler training.", DARK, 20, False),
    ("For tasks where checking answers is free — it's a clear win.", GRAY, 18, False),
])


# ============================================================
# SLIDE 6: DATASET
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
title_bar(slide, "The Dataset: GSM8K")

add_text(slide, 0.8, 1.5, 5.5, 1, [
    ("7,473 train  /  1,319 test problems", DARK, 22, False),
])

# Example box
shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(2.5), Inches(11.5), Inches(3))
shape.fill.solid()
shape.fill.fore_color.rgb = LIGHT_GRAY
shape.line.fill.background()

add_text(slide, 1.2, 2.7, 10.5, 1, [
    ("Q:  Janet's ducks lay 16 eggs per day. She eats 3 for breakfast", DARK, 20, False),
    ("     and bakes muffins with 4. She sells the rest at $2 each.", DARK, 20, False),
    ("     How much does she make daily?", DARK, 20, False),
])

add_text(slide, 1.2, 4.1, 10.5, 1, [
    ("A:  16 - 3 - 4 = 9.   9 × $2 = $18.   #### 18", DARK, 20, True),
])

add_text(slide, 0.8, 5.8, 11, 1.2, [
    ("We only check the final number. Model discovers reasoning on its own.", DARK, 22, False),
    ("Base model accuracy: 14.5%", ORANGE, 22, True),
])


# ============================================================
# SLIDE 7: THREE APPROACHES
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
title_bar(slide, "What We Built: Three Approaches")

add_table(slide, 0.8, 1.5, 11.5,
    ["Approach", "Accuracy", "Time", "Key Insight"],
    [
        ["Custom Trainer (from scratch)", "~60%", "26 hrs", "Learning exercise — every RL bug"],
        ["veRL (ByteDance)", "77%", "35 min", "Colocated GPUs, zero sync cost"],
        ["Monarch (Meta)", "87%", "100 min", "Split placement, weight sync bottleneck"],
    ],
    font_size=18
)

add_text(slide, 0.8, 4, 11, 2.5, [
    ("Same model (Qwen 1.5B), same data (GSM8K), same cluster (16× A100)", GRAY, 18, False),
    ("", DARK, 10, False),
    ("Custom: 800 lines Python, 5 attempts, 12 hours debugging", DARK, 20, False),
    ("veRL: 50 lines config, worked first try", DARK, 20, False),
    ("Monarch: 12 training runs, 20+ issues fixed, no prior FSDP example existed", DARK, 20, False),
])


# ============================================================
# SLIDE 8: RESULTS
# ============================================================
img_path = os.path.join("docs", "run-2026-04-09-monarch-run12", "overview.png")
if os.path.exists(img_path):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title_bar(slide, "Monarch Result: 14.5% → 87%")
    slide.shapes.add_picture(img_path, Inches(0.8), Inches(1.4), Inches(11.5), Inches(5.2))
    add_text(slide, 0.8, 6.7, 11, 0.5, [
        ("233 steps, 100 minutes, 1 epoch. KL stable with frozen reference model.", GRAY, 16, False),
    ])
else:
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    title_bar(slide, "Monarch Result: 14.5% → 87%")
    add_text(slide, 1, 3, 11, 2, [
        ("[Training curve image: overview.png]", GRAY, 24, False),
    ])


# ============================================================
# SLIDE 9: SPEED DIFFERENCE
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
title_bar(slide, "Why the Speed Difference?")

add_text(slide, 0.8, 1.5, 11, 1, [
    ("The #1 bottleneck: getting updated weights from trainer → generator", DARK, 24, True),
])

add_table(slide, 0.8, 2.6, 11.5,
    ["", "veRL (35 min)", "Monarch (100 min)"],
    [
        ["Architecture", "All GPUs switch roles", "Separate nodes per role"],
        ["Weight sync", "Zero — same GPUs", "47s per step (3GB over network)"],
        ["GPU utilization", "16 / 16", "5 / 16"],
    ],
    font_size=18
)

add_text(slide, 0.8, 5.2, 11, 2, [
    ("veRL: no data moves. Same GPU memory, just resharded.", DARK, 20, False),
    ("Monarch: 3GB transfer every step. 45% of training time.", DARK, 20, False),
    ("", DARK, 8, False),
    ("RDMA would fix this (~1s) but p4d EFA doesn't support it. P5+ does.", GRAY, 18, False),
])


# ============================================================
# SLIDE 10: TAKEAWAYS
# ============================================================
slide = prs.slides.add_slide(prs.slide_layouts[6])
title_bar(slide, "Key Takeaways")

takeaways = [
    ("RL + binary reward = LLMs that reason", "14.5% → 87%, no human solutions needed"),
    ("The algorithm is simple, the infra is hard", "GRPO fits on one slide. Weight sync took 12 runs."),
    ("Hyperparameters > Architecture", "kl_coef 100× wrong mattered more than any infra optimization"),
    ("Code: github.com/harishvs/rl_grpo_qwen_math", "3 implementations, all logs, Terraform, K8s manifests"),
]

for i, (main, sub) in enumerate(takeaways):
    y = 1.6 + i * 1.35
    add_text(slide, 0.8, y, 11, 0.6, [(f"{i+1}.  {main}", DARK, 24, True)])
    add_text(slide, 1.3, y + 0.5, 10, 0.5, [(sub, GRAY, 18, False)])


# ============================================================
out_path = "docs/presentation/monarch-grpo-presentation.pptx"
prs.save(out_path)
print(f"Saved {out_path} ({os.path.getsize(out_path) // 1024} KB)")
