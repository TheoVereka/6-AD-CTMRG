import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ============================================================
# Figure: Plaquette tensor - 4 T's on edges → W
# Slide 3.2: W_{s1,s2,s3,s4} = T_{s1,s2} * T_{s2,s3} * T_{s3,s4} * T_{s4,s1}
# NO central spin! Just 4 corner spin indices.
# ============================================================

fig, ax = plt.subplots(figsize=(14, 7.5))
ax.set_xlim(0, 14)
ax.set_ylim(0.5, 7.5)
ax.axis('off')

# ============================================================
# LEFT SIDE: 4 T matrices on edges of a plaquette
# ============================================================
left_cx, left_cy = 3.5, 3.5

# 4 corner spin indices (s1, s2, s3, s4) at corners of a square
diag = 1.6  # half-diagonal of square

corner_positions = [
    (0, diag),     # s1 (top)
    (diag, 0),     # s2 (right)
    (0, -diag),    # s3 (bottom)
    (-diag, 0)     # s4 (left)
]
corner_labels = [r'$s_1$', r'$s_2$', r'$s_3$', r'$s_4$']

# Draw edges between corners
bond_width = 3
for i in range(4):
    x1, y1 = corner_positions[i]
    x2, y2 = corner_positions[(i+1) % 4]
    ax.plot([left_cx + x1, left_cx + x2], [left_cy + y1, left_cy + y2],
           'k-', linewidth=bond_width, zorder=1)

# Draw T matrices on each edge (4 total)
T_size = 0.55
edge_midpoints = [
    (diag/2, diag/2),    # T_{s1,s2} (top-right edge)
    (diag/2, -diag/2),   # T_{s2,s3} (bottom-right edge)
    (-diag/2, -diag/2),  # T_{s3,s4} (bottom-left edge)
    (-diag/2, diag/2)    # T_{s4,s1} (top-left edge)
]
T_labels = [r'$T_{12}$', r'$T_{23}$', r'$T_{34}$', r'$T_{41}$']

for i, (tx, ty) in enumerate(edge_midpoints):
    T_rect = patches.FancyBboxPatch(
        (left_cx + tx - T_size/2, left_cy + ty - T_size/2),
        T_size, T_size,
        boxstyle="round,pad=0.05",
        edgecolor='#B8860B',
        facecolor='#FFD700',
        linewidth=2.5,
        zorder=3
    )
    ax.add_patch(T_rect)
    ax.text(left_cx + tx, left_cy + ty, T_labels[i],
           ha='center', va='center', fontsize=14, fontweight='bold',
           color='#8B4513', zorder=4)

# Draw corner spin labels (s1, s2, s3, s4)
label_offset = 0.4
for i, (cx, cy) in enumerate(corner_positions):
    # Position labels outside the corners
    lx = cx * 1.25
    ly = cy * 1.25
    ax.text(left_cx + lx, left_cy + ly, corner_labels[i],
           ha='center', va='center', fontsize=22, fontweight='bold',
           color='darkblue', zorder=5)

# Label below
ax.text(left_cx, left_cy - 2.3, r'Plaquette: 4 edges with $T$',
       ha='center', va='center', fontsize=20, fontweight='bold',
       color='#333333')

# ============================================================
# ARROW: Multiplication
# ============================================================
arrow_x = 7
ax.annotate('', xy=(arrow_x + 0.8, 3.5), xytext=(arrow_x - 1.0, 3.5),
           arrowprops=dict(arrowstyle='-|>', color='darkgreen', lw=4))
ax.text(arrow_x-0.1, 4.1, 'scalar', ha='center', va='center',
       fontsize=18, fontweight='bold', color='darkgreen')
ax.text(arrow_x-0.1, 3.7, 'multiply', ha='center', va='center',
       fontsize=18, fontweight='bold', color='darkgreen')

# ============================================================
# RIGHT SIDE: Single plaquette tensor W
# ============================================================
right_cx, right_cy = 10.5, 3.5

# Draw the W tensor as a square with 4 legs
W_size = 1.8
W_rect = patches.FancyBboxPatch(
    (right_cx - W_size/2, right_cy - W_size/2),
    W_size, W_size,
    boxstyle="round,pad=0.1",
    edgecolor='#B8860B',
    facecolor='#FFD700',
    linewidth=4,
    zorder=2
)
ax.add_patch(W_rect)

ax.text(right_cx, right_cy, r'$W$', ha='center', va='center',
       fontsize=36, fontweight='bold', color='#8B4513', zorder=3)

# 4 external legs of W (matching the T directions: up, right, down, left)
leg_length = 1.2
leg_width = 3.5

# Cardinal directions (matching T on edges)
leg_dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # up, right, down, left
leg_labels = [r'$s_1$', r'$s_2$', r'$s_3$', r'$s_4$']

for i, (dx, dy) in enumerate(leg_dirs):
    start_x = right_cx + dx * W_size/2
    start_y = right_cy + dy * W_size/2
    end_x = start_x + dx * leg_length
    end_y = start_y + dy * leg_length
    ax.plot([start_x, end_x], [start_y, end_y], 'k-', linewidth=leg_width, zorder=1)
    # Label at end
    ax.text(end_x + dx*0.4, end_y + dy*0.4, leg_labels[i],
           ha='center', va='center', fontsize=20, fontweight='bold', color='darkblue')

# Label below
ax.text(right_cx, right_cy - 2.3, r'Plaquette tensor $W$',
       ha='center', va='center', fontsize=20, fontweight='bold',
       color='#333333')
ax.text(right_cx, right_cy - 2.8, r'rank-4: $(s_1, s_2, s_3, s_4)$',
       ha='center', va='center', fontsize=16, color='#555555')

# ============================================================
# Top formula (CORRECTED: no sum over σ, just product)
# ============================================================
ax.text(7, 6.8, r'$W_{s_1,s_2,s_3,s_4} = T_{s_1,s_2} \cdot T_{s_2,s_3} \cdot T_{s_3,s_4} \cdot T_{s_4,s_1}$',
       ha='center', va='center', fontsize=24, fontweight='bold',
       color='black',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F5E9',
                edgecolor='darkgreen', linewidth=2.5))

# Bottom note about T - REMOVED (latex has it)

# ============================================================
# Save figure
# ============================================================
plt.tight_layout()
plt.savefig('fig_plaquette_tensor.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('fig_plaquette_tensor.png', format='png', bbox_inches='tight', dpi=300)
print("Saved: figures/fig_plaquette_tensor.pdf")
print("Saved: figures/fig_plaquette_tensor.png")
plt.close()
