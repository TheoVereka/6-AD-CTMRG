import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ============================================================
# Figure: CTMRG Complete Algorithm Flowchart
# Slide 3.5: Initialize → Absorb → Compute P → Truncate → Converged?
# ============================================================

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Colors
init_color = '#E3F2FD'
init_edge = '#1976D2'
step_color = '#FFF3E0'
step_edge = '#F57C00'
decision_color = '#E8F5E9'
decision_edge = '#388E3C'
output_color = '#FCE4EC'
output_edge = '#C2185B'

# Box parameters
box_width = 6.0
box_height = 0.85
box_x = 7  # center x

# ============================================================
# Step 0: Initialize
# ============================================================
y0 = 9.0
init_rect = patches.FancyBboxPatch(
    (box_x - box_width/2, y0 - box_height/2),
    box_width, box_height,
    boxstyle="round,pad=0.15",
    edgecolor=init_edge,
    facecolor=init_color,
    linewidth=3,
    zorder=2
)
ax.add_patch(init_rect)
ax.text(box_x, y0, r'Initialize: $C_1, C_2, C_3, C_4$ and $T_1, T_2, T_3, T_4$',
       ha='center', va='center', fontsize=20, fontweight='bold', color=init_edge, zorder=3)

# Arrow down
ax.annotate('', xy=(box_x, y0 - box_height/2 - 0.6), xytext=(box_x, y0 - box_height/2 - 0.1),
           arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

# ============================================================
# Step 1: Absorb (Grow)
# ============================================================
y1 = 7.2
step1_rect = patches.FancyBboxPatch(
    (box_x - box_width/2, y1 - box_height/2),
    box_width, box_height,
    boxstyle="round,pad=0.15",
    edgecolor=step_edge,
    facecolor=step_color,
    linewidth=3,
    zorder=2
)
ax.add_patch(step1_rect)
ax.text(box_x, y1, r"1. Absorb: $C' = C \cdot T \cdot a$, $T' = T \cdot a$",
       ha='center', va='center', fontsize=19, fontweight='bold', color=step_edge, zorder=3)

# Step number on left
ax.text(box_x - box_width/2 - 0.5, y1, '①', ha='center', va='center',
       fontsize=24, fontweight='bold', color=step_edge)

# Arrow down
ax.annotate('', xy=(box_x, y1 - box_height/2 - 0.6), xytext=(box_x, y1 - box_height/2 - 0.1),
           arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

# ============================================================
# Step 2: Compute Projector
# ============================================================
y2 = 5.4
step2_rect = patches.FancyBboxPatch(
    (box_x - box_width/2, y2 - box_height/2),
    box_width, box_height,
    boxstyle="round,pad=0.15",
    edgecolor=step_edge,
    facecolor=step_color,
    linewidth=3,
    zorder=2
)
ax.add_patch(step2_rect)
ax.text(box_x, y2, r"2. Compute projector: $\rho \to P$ (SVD)",
       ha='center', va='center', fontsize=19, fontweight='bold', color=step_edge, zorder=3)

# Step number on left
ax.text(box_x - box_width/2 - 0.5, y2, '②', ha='center', va='center',
       fontsize=24, fontweight='bold', color=step_edge)

# Arrow down
ax.annotate('', xy=(box_x, y2 - box_height/2 - 0.6), xytext=(box_x, y2 - box_height/2 - 0.1),
           arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

# ============================================================
# Step 3: Truncate
# ============================================================
y3 = 3.6
step3_rect = patches.FancyBboxPatch(
    (box_x - box_width/2, y3 - box_height/2),
    box_width, box_height,
    boxstyle="round,pad=0.15",
    edgecolor=step_edge,
    facecolor=step_color,
    linewidth=3,
    zorder=2
)
ax.add_patch(step3_rect)
ax.text(box_x, y3, r"3. Truncate: $C_{\mathrm{new}} = P^\dagger C' P$, same for $T$",
       ha='center', va='center', fontsize=19, fontweight='bold', color=step_edge, zorder=3)

# Step number on left
ax.text(box_x - box_width/2 - 0.5, y3, '③', ha='center', va='center',
       fontsize=24, fontweight='bold', color=step_edge)

# Arrow down
ax.annotate('', xy=(box_x, y3 - box_height/2 - 0.6), xytext=(box_x, y3 - box_height/2 - 0.1),
           arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

# ============================================================
# Step 4: Convergence check (diamond shape)
# ============================================================
y4 = 1.7
diamond_size = 1.3
diamond = patches.RegularPolygon(
    (box_x, y4), numVertices=4, radius=diamond_size,
    orientation=np.pi/4,
    edgecolor=decision_edge,
    facecolor=decision_color,
    linewidth=3,
    zorder=2
)
ax.add_patch(diamond)
ax.text(box_x, y4, 'Converged?', ha='center', va='center',
       fontsize=16, fontweight='bold', color=decision_edge, zorder=3)

# ============================================================
# Loop back arrow (No)
# ============================================================
# Right side of diamond → up → back to step 1
loop_x = box_x + box_width/2 + 1.5
ax.plot([box_x + diamond_size*0.7, loop_x], [y4, y4], 'k-', linewidth=2.5)
ax.plot([loop_x, loop_x], [y4, y1], 'k-', linewidth=2.5)
ax.annotate('', xy=(box_x + box_width/2, y1), xytext=(loop_x, y1),
           arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

# "No" label
ax.text(box_x + diamond_size*0.7 + 0.5, y4 + 0.35, 'No',
       ha='center', va='center', fontsize=16, fontweight='bold', color='darkred')

# ============================================================
# Output arrow (Yes) - goes down
# ============================================================
ax.annotate('', xy=(box_x, y4 - diamond_size - 0.5), xytext=(box_x, y4 - diamond_size*0.7),
           arrowprops=dict(arrowstyle='->', color='black', lw=2.5))

# "Yes" label
ax.text(box_x - 0.6, y4 - diamond_size*0.85, 'Yes',
       ha='center', va='center', fontsize=16, fontweight='bold', color='darkgreen')

# Output box
yo = 0.0
output_rect = patches.FancyBboxPatch(
    (box_x - 2.5, yo - 0.4),
    5, 0.8,
    boxstyle="round,pad=0.15",
    edgecolor=output_edge,
    facecolor=output_color,
    linewidth=3,
    zorder=2
)
ax.add_patch(output_rect)
ax.text(box_x, yo, r'Output: Fixed-point $C^*, T^*$',
       ha='center', va='center', fontsize=20, fontweight='bold', color=output_edge, zorder=3)

# ============================================================
# Side annotations
# ============================================================
# Left side: what grows
ax.text(1.0, y1, r'$\chi \to \chi d$',
       ha='center', va='center', fontsize=16, fontweight='bold', color='darkred',
       bbox=dict(boxstyle='round,pad=0.25', facecolor='#FFCCCB',
                edgecolor='darkred', linewidth=1.5))

# Left side: what truncates
ax.text(1.0, y3, r'$\chi d \to \chi$',
       ha='center', va='center', fontsize=16, fontweight='bold', color='darkgreen',
       bbox=dict(boxstyle='round,pad=0.25', facecolor='#98FB98',
                edgecolor='darkgreen', linewidth=1.5))

# Right side: iteration count
ax.text(loop_x + 0.8, (y1 + y4)/2, 'Iterate',
       ha='center', va='center', fontsize=16, fontweight='bold', color='gray',
       rotation=90)

# ============================================================
# Save figure
# ============================================================
plt.tight_layout()
plt.savefig('fig_ctmrg_algorithm.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('fig_ctmrg_algorithm.png', format='png', bbox_inches='tight', dpi=300)
print("Saved: figures/fig_ctmrg_algorithm.pdf")
print("Saved: figures/fig_ctmrg_algorithm.png")
plt.close()
