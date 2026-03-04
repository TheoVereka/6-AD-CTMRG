import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ============================================================
# Figure: 2D Tensor Network - T matrices on bonds (45° tilted)
# Slide 2.1: T is rank-2 on each bond, but forms 2D network
# ============================================================

fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(-1, 13)
ax.set_ylim(-1, 11)
ax.axis('off')

# Layout parameters (45° tilted lattice)
site_radius = 0.12  # Small dots
T_size = 0.7        # Transfer matrix size
spacing = 2.0       # Distance between adjacent sites

# Colors
site_color = 'black'
T_color = '#FFD700'  # Gold
line_color = 'black'

# 45° rotation: use diagonal directions
# Sites form a tilted square lattice
# dx, dy for "right" direction: (1, 1) * spacing/sqrt(2)
# dx, dy for "up" direction: (-1, 1) * spacing/sqrt(2)

diag = spacing / np.sqrt(2)

# Grid: 4x4 sites in tilted coordinates
n_size = 4

# Center of figure
cx, cy = 6, 5

# Calculate site positions (tilted 45°)
sites = []
for i in range(n_size):
    for j in range(n_size):
        # Tilted coordinates
        x = cx + (i - j) * diag
        y = cy + (i + j - n_size + 1) * diag
        sites.append((x, y, i, j))

# ============================================================
# Draw bonds with T matrices
# ============================================================

# Draw diagonal bonds (these are the "horizontal" and "vertical" in tilted frame)
for x, y, i, j in sites:
    # "Right" bond (i+1, j)
    if i < n_size - 1:
        x2 = cx + (i + 1 - j) * diag
        y2 = cy + (i + 1 + j - n_size + 1) * diag
        
        # Draw bond line
        ax.plot([x, x2], [y, y2], 'k-', linewidth=2.5, zorder=1)
        
        # T matrix in the middle
        T_x, T_y = (x + x2) / 2, (y + y2) / 2
        
        # Rotated rectangle for T (45° tilted)
        T_rect = patches.FancyBboxPatch(
            (T_x - T_size/2, T_y - T_size/2),
            T_size, T_size,
            boxstyle="round,pad=0.05",
            edgecolor='black',
            facecolor=T_color,
            linewidth=2.5,
            zorder=2
        )
        ax.add_patch(T_rect)
        ax.text(T_x, T_y, r'$T$', ha='center', va='center',
               fontsize=24, fontweight='bold', color='black', zorder=3)
    
    # "Up" bond (i, j+1)
    if j < n_size - 1:
        x2 = cx + (i - (j + 1)) * diag
        y2 = cy + (i + (j + 1) - n_size + 1) * diag
        
        # Draw bond line
        ax.plot([x, x2], [y, y2], 'k-', linewidth=2.5, zorder=1)
        
        # T matrix in the middle
        T_x, T_y = (x + x2) / 2, (y + y2) / 2
        
        T_rect = patches.FancyBboxPatch(
            (T_x - T_size/2, T_y - T_size/2),
            T_size, T_size,
            boxstyle="round,pad=0.05",
            edgecolor='black',
            facecolor=T_color,
            linewidth=2.5,
            zorder=2
        )
        ax.add_patch(T_rect)
        ax.text(T_x, T_y, r'$T$', ha='center', va='center',
               fontsize=24, fontweight='bold', color='black', zorder=3)

# Draw all sites (small dots) on top
for x, y, i, j in sites:
    circle = patches.Circle((x, y), site_radius,
                           edgecolor='black', facecolor=site_color,
                           linewidth=2, zorder=4)
    ax.add_patch(circle)

# ============================================================
# Add annotations
# ============================================================

# Main message at top - NO rank-4, just network complexity
ax.text(6, 10.2, r'2D: $T$ on each bond $\Rightarrow$ No natural contraction order!',
       ha='center', va='center', fontsize=26, fontweight='bold',
       color='darkred',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFE4E1', 
                edgecolor='darkred', linewidth=2.5))

# Label T dimension
ax.text(10.5, 5, r'$T$: $2 \times 2$',
       ha='left', va='center', fontsize=22, fontweight='bold',
       color='black',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4',
                edgecolor='black', linewidth=2))

# Ellipsis to indicate continuation
ax.text(cx + 3.5 * diag, cy + 0.5 * diag, '...',
       ha='left', va='center', fontsize=36, fontweight='bold', color='black')
ax.text(cx - 3.5 * diag, cy + 0.5 * diag, '...',
       ha='right', va='center', fontsize=36, fontweight='bold', color='black')
ax.text(cx, cy + 4 * diag, '...',
       ha='center', va='bottom', fontsize=36, fontweight='bold', color='black')

# ============================================================
# Save figure
# ============================================================
plt.tight_layout()
plt.savefig('figures/fig_2d_tensor_network.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/fig_2d_tensor_network.png', format='png', bbox_inches='tight', dpi=150)
print("Saved: figures/fig_2d_tensor_network.pdf")
print("Saved: figures/fig_2d_tensor_network.png")
plt.close()
