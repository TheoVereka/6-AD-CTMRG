import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ============================================================
# Figure: CTMRG Environment Structure - 4 Corners + 4 Edges
# Slide 3.1: The key structure of CTMRG
# ============================================================

fig, ax = plt.subplots(figsize=(14, 11))
ax.set_xlim(0, 14)
ax.set_ylim(0, 11)
ax.axis('off')

# Center of the structure
cx, cy = 7, 5.5

# ============================================================
# Layout parameters
# ============================================================
corner_size = 1.8       # Corner squares
edge_width = 2.2        # Edge rectangles (longer dimension)
edge_height = 1.0       # Edge rectangles (shorter dimension)
local_size = 1.0        # Local tensor in center
spacing = 2.8           # Distance from center to edges/corners

# Colors
corner_color = '#98FB98'    # Light green for corners
edge_color = '#87CEEB'      # Sky blue for edges
local_color = '#FFD700'     # Gold for local tensor
line_color = 'black'

# ============================================================
# Draw 4 Corners (squares)
# ============================================================
corner_positions = {
    'C1': (cx + spacing, cy + spacing),      # Top-right
    'C2': (cx + spacing, cy - spacing),      # Bottom-right
    'C3': (cx - spacing, cy - spacing),      # Bottom-left
    'C4': (cx - spacing, cy + spacing),      # Top-left
}

for name, (x, y) in corner_positions.items():
    rect = patches.FancyBboxPatch(
        (x - corner_size/2, y - corner_size/2),
        corner_size, corner_size,
        boxstyle="round,pad=0.08",
        edgecolor='darkgreen',
        facecolor=corner_color,
        linewidth=3,
        zorder=2
    )
    ax.add_patch(rect)
    ax.text(x, y, f'${name}$', ha='center', va='center',
           fontsize=32, fontweight='bold', color='darkgreen', zorder=3)

# ============================================================
# Draw 4 Edges (rectangles)
# ============================================================
# T1: right edge (vertical)
T1_x, T1_y = cx + spacing, cy
T1_rect = patches.FancyBboxPatch(
    (T1_x - edge_height/2, T1_y - edge_width/2),
    edge_height, edge_width,
    boxstyle="round,pad=0.06",
    edgecolor='darkblue',
    facecolor=edge_color,
    linewidth=3,
    zorder=2
)
ax.add_patch(T1_rect)
ax.text(T1_x, T1_y, r'$T_1$', ha='center', va='center',
       fontsize=28, fontweight='bold', color='darkblue', zorder=3)

# T2: bottom edge (horizontal)
T2_x, T2_y = cx, cy - spacing
T2_rect = patches.FancyBboxPatch(
    (T2_x - edge_width/2, T2_y - edge_height/2),
    edge_width, edge_height,
    boxstyle="round,pad=0.06",
    edgecolor='darkblue',
    facecolor=edge_color,
    linewidth=3,
    zorder=2
)
ax.add_patch(T2_rect)
ax.text(T2_x, T2_y, r'$T_2$', ha='center', va='center',
       fontsize=28, fontweight='bold', color='darkblue', zorder=3)

# T3: left edge (vertical)
T3_x, T3_y = cx - spacing, cy
T3_rect = patches.FancyBboxPatch(
    (T3_x - edge_height/2, T3_y - edge_width/2),
    edge_height, edge_width,
    boxstyle="round,pad=0.06",
    edgecolor='darkblue',
    facecolor=edge_color,
    linewidth=3,
    zorder=2
)
ax.add_patch(T3_rect)
ax.text(T3_x, T3_y, r'$T_3$', ha='center', va='center',
       fontsize=28, fontweight='bold', color='darkblue', zorder=3)

# T4: top edge (horizontal)
T4_x, T4_y = cx, cy + spacing
T4_rect = patches.FancyBboxPatch(
    (T4_x - edge_width/2, T4_y - edge_height/2),
    edge_width, edge_height,
    boxstyle="round,pad=0.06",
    edgecolor='darkblue',
    facecolor=edge_color,
    linewidth=3,
    zorder=2
)
ax.add_patch(T4_rect)
ax.text(T4_x, T4_y, r'$T_4$', ha='center', va='center',
       fontsize=28, fontweight='bold', color='darkblue', zorder=3)

# ============================================================
# Draw local tensor in center
# ============================================================
local_rect = patches.FancyBboxPatch(
    (cx - local_size/2, cy - local_size/2),
    local_size, local_size,
    boxstyle="round,pad=0.06",
    edgecolor='darkred',
    facecolor=local_color,
    linewidth=3,
    zorder=2
)
ax.add_patch(local_rect)
ax.text(cx, cy, r'$a$', ha='center', va='center',
       fontsize=32, fontweight='bold', color='darkred', zorder=3)

# ============================================================
# Draw connecting bonds
# ============================================================
bond_width = 3.5

# Corners to edges
# C1 to T1 (vertical)
ax.plot([corner_positions['C1'][0], T1_x],
       [corner_positions['C1'][1] - corner_size/2, T1_y + edge_width/2],
       'k-', linewidth=bond_width, zorder=1)
# C1 to T4 (horizontal)
ax.plot([corner_positions['C1'][0] - corner_size/2, T4_x + edge_width/2],
       [corner_positions['C1'][1], T4_y],
       'k-', linewidth=bond_width, zorder=1)

# C2 to T1 (vertical)
ax.plot([corner_positions['C2'][0], T1_x],
       [corner_positions['C2'][1] + corner_size/2, T1_y - edge_width/2],
       'k-', linewidth=bond_width, zorder=1)
# C2 to T2 (horizontal)
ax.plot([corner_positions['C2'][0] - corner_size/2, T2_x + edge_width/2],
       [corner_positions['C2'][1], T2_y],
       'k-', linewidth=bond_width, zorder=1)

# C3 to T2 (horizontal)
ax.plot([corner_positions['C3'][0] + corner_size/2, T2_x - edge_width/2],
       [corner_positions['C3'][1], T2_y],
       'k-', linewidth=bond_width, zorder=1)
# C3 to T3 (vertical)
ax.plot([corner_positions['C3'][0], T3_x],
       [corner_positions['C3'][1] + corner_size/2, T3_y - edge_width/2],
       'k-', linewidth=bond_width, zorder=1)

# C4 to T3 (vertical)
ax.plot([corner_positions['C4'][0], T3_x],
       [corner_positions['C4'][1] - corner_size/2, T3_y + edge_width/2],
       'k-', linewidth=bond_width, zorder=1)
# C4 to T4 (horizontal)
ax.plot([corner_positions['C4'][0] + corner_size/2, T4_x - edge_width/2],
       [corner_positions['C4'][1], T4_y],
       'k-', linewidth=bond_width, zorder=1)

# Edges to local tensor
# T1 to a
ax.plot([T1_x - edge_height/2, cx + local_size/2],
       [T1_y, cy],
       'k-', linewidth=bond_width, zorder=1)
# T2 to a
ax.plot([T2_x, cx],
       [T2_y + edge_height/2, cy - local_size/2],
       'k-', linewidth=bond_width, zorder=1)
# T3 to a
ax.plot([T3_x + edge_height/2, cx - local_size/2],
       [T3_y, cy],
       'k-', linewidth=bond_width, zorder=1)
# T4 to a
ax.plot([T4_x, cx],
       [T4_y - edge_height/2, cy + local_size/2],
       'k-', linewidth=bond_width, zorder=1)

# ============================================================
# Add dimension labels
# ============================================================
# Corner dimension
ax.text(corner_positions['C1'][0] + corner_size/2 + 0.3, 
       corner_positions['C1'][1] + corner_size/2 + 0.2,
       r'$\chi \times \chi$', ha='left', va='bottom',
       fontsize=20, color='darkgreen', fontweight='bold')

# Edge dimension  
ax.text(T1_x + edge_height/2 + 0.3, T1_y,
       r'$\chi \times d \times \chi$', ha='left', va='center',
       fontsize=20, color='darkblue', fontweight='bold')

# Local tensor dimension
ax.text(cx + local_size/2 + 0.2, cy - local_size/2 - 0.2,
       r'$d^4$', ha='left', va='top',
       fontsize=20, color='darkred', fontweight='bold')

# ============================================================
# Legend at bottom
# ============================================================
legend_y = 0.8

# Corner legend
corner_legend = patches.FancyBboxPatch(
    (1.5, legend_y - 0.3), 0.6, 0.6,
    boxstyle="round,pad=0.05",
    edgecolor='darkgreen', facecolor=corner_color, linewidth=2
)
ax.add_patch(corner_legend)
ax.text(2.5, legend_y, r'Corner $C_i$: $\chi \times \chi$',
       ha='left', va='center', fontsize=20, color='black')

# Edge legend
edge_legend = patches.FancyBboxPatch(
    (6.5, legend_y - 0.25), 0.8, 0.5,
    boxstyle="round,pad=0.05",
    edgecolor='darkblue', facecolor=edge_color, linewidth=2
)
ax.add_patch(edge_legend)
ax.text(7.7, legend_y, r'Edge $T_i$: $\chi \times d \times \chi$',
       ha='left', va='center', fontsize=20, color='black')

# ============================================================
# Title/message
# ============================================================
ax.text(cx, 10.3,
       r'CTMRG: Decompose 2D environment into 4 Corners + 4 Edges',
       ha='center', va='center', fontsize=26, fontweight='bold',
       color='darkblue',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='#E6F2FF',
                edgecolor='darkblue', linewidth=2.5))

# ============================================================
# Save figure
# ============================================================
plt.tight_layout()
plt.savefig('figures/fig_2d_ctmrg_structure.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/fig_2d_ctmrg_structure.png', format='png', bbox_inches='tight', dpi=150)
print("Saved: figures/fig_2d_ctmrg_structure.pdf")
print("Saved: figures/fig_2d_ctmrg_structure.png")
plt.close()
