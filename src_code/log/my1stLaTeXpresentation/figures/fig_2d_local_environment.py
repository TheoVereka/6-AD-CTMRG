import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ============================================================
# Figure: Local site surrounded by 2D environment
# Slide 2.3: Environment encodes the rest of the infinite lattice
# ============================================================

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Center of figure
cx, cy = 7, 5

# ============================================================
# Draw the outer environment (large rounded rectangle)
# ============================================================
env_width = 10
env_height = 7
env_rect = patches.FancyBboxPatch(
    (cx - env_width/2, cy - env_height/2),
    env_width, env_height,
    boxstyle="round,pad=0.3",
    edgecolor='darkblue',
    facecolor='#E6F2FF',
    linewidth=4,
    zorder=1
)
ax.add_patch(env_rect)

# Environment label (adjusted position)
ax.text(cx, cy + env_height/2 - 0.5, 'Environment',
       ha='center', va='center', fontsize=32, fontweight='bold',
       color='darkblue', zorder=2)

# ============================================================
# Draw the local site in the center (highlighted)
# ============================================================
local_size = 1.0
local_circle = patches.Circle((cx, cy), local_size/2,
                              edgecolor='darkred',
                              facecolor='#FFD700',
                              linewidth=4,
                              zorder=3)
ax.add_patch(local_circle)

ax.text(cx, cy, r'$\sigma$', ha='center', va='center',
       fontsize=36, fontweight='bold', color='darkred', zorder=4)

# Label: "local" (moved down to avoid overlapping Edge)
ax.annotate('local site',
           xy=(cx + local_size/2 + 0.1, cy - 0.3),
           xytext=(cx + 2.5, cy - 1.2),
           ha='left', va='center', fontsize=24, color='darkred',
           fontweight='bold',
           arrowprops=dict(arrowstyle='->', color='darkred', lw=2.5),
           zorder=4)

# ============================================================
# Draw 4 diagonal bonds to neighboring sites (45° rotated geometry)
# ============================================================
bond_length = 2.2
bond_width = 3
site_radius = 0.15

# Diagonal directions: NE, NW, SW, SE
diag_offset = bond_length / np.sqrt(2)
diagonal_positions = [
    (diag_offset, diag_offset),   # NE
    (-diag_offset, diag_offset),  # NW
    (-diag_offset, -diag_offset), # SW
    (diag_offset, -diag_offset)   # SE
]

# Draw bonds (black lines) from center to diagonal neighbors
for dx, dy in diagonal_positions:
    ax.plot([cx, cx + dx], [cy, cy + dy],
           'k-', linewidth=bond_width, zorder=2)

# Draw transfer matrices T on each bond (gold squares, 45° rotated)
T_size = 0.55
for dx, dy in diagonal_positions:
    # Position at midpoint of bond
    tx = cx + dx / 2
    ty = cy + dy / 2
    T_rect = patches.FancyBboxPatch(
        (tx - T_size/2, ty - T_size/2),
        T_size, T_size,
        boxstyle="round,pad=0.05",
        edgecolor='#B8860B',
        facecolor='#FFD700',
        linewidth=2,
        zorder=4
    )
    ax.add_patch(T_rect)
    ax.text(tx, ty, r'$T$', ha='center', va='center',
           fontsize=18, fontweight='bold', color='#8B4513', zorder=5)

# Draw neighboring sites at diagonal positions (black dots)
for dx, dy in diagonal_positions:
    dot = patches.Circle((cx + dx, cy + dy), site_radius,
                         edgecolor='black', facecolor='black',
                         linewidth=2, zorder=3)
    ax.add_patch(dot)

# ============================================================
# Label the 4 Edges (orthogonal directions)
# ============================================================
edge_distance = 2.5
edge_fontsize = 26

# Top Edge
ax.text(cx, cy + edge_distance, 'Edge 1',
       ha='center', va='center', fontsize=edge_fontsize,
       fontweight='bold', color='#1565C0',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='#87CEEB',
                edgecolor='#1565C0', linewidth=2.5))

# Bottom Edge
ax.text(cx, cy - edge_distance, 'Edge 3',
       ha='center', va='center', fontsize=edge_fontsize,
       fontweight='bold', color='#1565C0',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='#87CEEB',
                edgecolor='#1565C0', linewidth=2.5))

# Left Edge
ax.text(cx - edge_distance, cy, 'Edge 4',
       ha='center', va='center', fontsize=edge_fontsize,
       fontweight='bold', color='#1565C0',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='#87CEEB',
                edgecolor='#1565C0', linewidth=2.5))

# Right Edge
ax.text(cx + edge_distance, cy, 'Edge 2',
       ha='center', va='center', fontsize=edge_fontsize,
       fontweight='bold', color='#1565C0',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='#87CEEB',
                edgecolor='#1565C0', linewidth=2.5))

# ============================================================
# Label the 4 Corners (diagonal directions, outside edges)
# ============================================================
corner_distance = 3.6
corner_fontsize = 24

# NE Corner
ax.text(cx + corner_distance/np.sqrt(2), cy + corner_distance/np.sqrt(2), 'Corner',
       ha='center', va='center', fontsize=corner_fontsize,
       fontweight='bold', color='darkgreen',
       bbox=dict(boxstyle='round,pad=0.35', facecolor='#98FB98',
                edgecolor='darkgreen', linewidth=2.5))

# NW Corner
ax.text(cx - corner_distance/np.sqrt(2), cy + corner_distance/np.sqrt(2), 'Corner',
       ha='center', va='center', fontsize=corner_fontsize,
       fontweight='bold', color='darkgreen',
       bbox=dict(boxstyle='round,pad=0.35', facecolor='#98FB98',
                edgecolor='darkgreen', linewidth=2.5))

# SW Corner
ax.text(cx - corner_distance/np.sqrt(2), cy - corner_distance/np.sqrt(2), 'Corner',
       ha='center', va='center', fontsize=corner_fontsize,
       fontweight='bold', color='darkgreen',
       bbox=dict(boxstyle='round,pad=0.35', facecolor='#98FB98',
                edgecolor='darkgreen', linewidth=2.5))

# SE Corner
ax.text(cx + corner_distance/np.sqrt(2), cy - corner_distance/np.sqrt(2), 'Corner',
       ha='center', va='center', fontsize=corner_fontsize,
       fontweight='bold', color='darkgreen',
       bbox=dict(boxstyle='round,pad=0.35', facecolor='#98FB98',
                edgecolor='darkgreen', linewidth=2.5))

# ============================================================
# Key message
# ============================================================
ax.text(cx, cy - env_height/2 - 1.0,
       r'$\langle \sigma \rangle = \frac{\mathrm{Tr}(\text{Environment} \cdot \sigma)}{\mathrm{Tr}(\text{Environment})}$',
       ha='center', va='top', fontsize=28, fontweight='bold',
       color='black',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F5E9',
                edgecolor='darkgreen', linewidth=2.5))

# Question at top
ax.text(cx, 9.3,
       r'Only need local $\langle\sigma\rangle$? $\Rightarrow$ Compress the environment!',
       ha='center', va='center', fontsize=26, fontweight='bold',
       color='#F57C00',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF3E0',
                edgecolor='#F57C00', linewidth=2))

# ============================================================
# Save figure
# ============================================================
plt.tight_layout()
plt.savefig('figures/fig_2d_local_environment.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/fig_2d_local_environment.png', format='png', bbox_inches='tight', dpi=150)
print("Saved: figures/fig_2d_local_environment.pdf")
print("Saved: figures/fig_2d_local_environment.png")
plt.close()
