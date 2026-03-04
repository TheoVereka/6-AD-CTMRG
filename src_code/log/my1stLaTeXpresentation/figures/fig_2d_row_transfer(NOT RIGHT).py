"""
Generate figure: 2D lattice with row-to-row transfer matrix
For CTMRG presentation - Part 1, Slide 1.2

Concept: Three rows showing how local transfer matrices combine into transfer slab
         Row n-1: ○──○──○──○──○
                  [T][T][T][T][T]  (local transfer matrices)
         Row n:   ○──○──○──○──○     } these combine into
                  │  │  │  │  │       
                [====T_row====]      (giant transfer slab)
                  │  │  │  │  │
         Row n+1: ○──○──○──○──○
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrow
import numpy as np

# Set up the figure
fig, ax = plt.subplots(1, 1, figsize=(10, 7))
ax.set_aspect('equal')

# Define colors
spin_color = '#4A90E2'  # Blue for spins
transfer_slab_color = '#FFE6E6'  # Light red/pink for transfer slab
transfer_edge_color = '#8B0000'  # Dark red edge
local_transfer_color = '#E3F2FD'  # Light blue for local transfers
local_transfer_edge = '#1565C0'  # Blue edge for local transfers
bond_color = '#333333'  # Dark gray for bonds
text_color = '#333333'

# Parameters
num_spins = 5  # L = 5 spins per row
y_row_n_minus_1 = 5.0  # y position of row n-1 (top)
y_row_n = 3.2  # y position of row n (middle)
y_row_n1 = 0.5  # y position of row n+1 (bottom)
x_start = 0.5
x_spacing = 1.5
spin_radius = 0.18

# Local transfer matrix dimensions (small boxes between n-1 and n)
local_t_width = 0.5
local_t_height = 0.4
local_t_y = (y_row_n_minus_1 + y_row_n) / 2  # Midpoint between row n-1 and row n

# Giant transfer slab dimensions (between n and n+1)
slab_x_center = x_start + (num_spins - 1) * x_spacing / 2
slab_y_center = (y_row_n + y_row_n1) / 2
slab_width = 3.5
slab_height = 0.8

# ========== Draw Row n-1 (top row) ==========
for i in range(num_spins):
    x_spin = x_start + i * x_spacing
    
    # Spin node
    spin_circle = Circle((x_spin, y_row_n_minus_1), spin_radius,
                        color=spin_color, ec='#003366',
                        linewidth=2, zorder=4)
    ax.add_patch(spin_circle)
    
    # Horizontal bonds within row n-1
    if i < num_spins - 1:
        ax.plot([x_spin + spin_radius, x_start + (i+1)*x_spacing - spin_radius],
               [y_row_n_minus_1, y_row_n_minus_1],
               'k-', linewidth=2, zorder=3)

# ========== Draw local transfer matrices (between n-1 and n) ==========
for i in range(num_spins):
    x_spin = x_start + i * x_spacing
    
    # Draw small local transfer matrix box
    local_t_box = FancyBboxPatch(
        (x_spin - local_t_width/2, local_t_y - local_t_height/2),
        local_t_width, local_t_height,
        boxstyle="round,pad=0.02",
        facecolor=local_transfer_color,
        edgecolor=local_transfer_edge,
        linewidth=2,
        zorder=2
    )
    ax.add_patch(local_t_box)
    
    # Label small T
    ax.text(x_spin, local_t_y, r'$T$',
           ha='center', va='center', fontsize=10,
           weight='bold', color=local_transfer_edge, zorder=3)
    
    # Vertical bonds through local transfer
    # From row n-1 to local T
    ax.plot([x_spin, x_spin],
           [y_row_n_minus_1 - spin_radius, local_t_y - local_t_height/2],
           color=bond_color, linewidth=1.5, zorder=1, linestyle='-', alpha=0.5)
    # From local T to row n
    ax.plot([x_spin, x_spin],
           [local_t_y + local_t_height/2, y_row_n - spin_radius],
           color=bond_color, linewidth=1.5, zorder=1, linestyle='-', alpha=0.5)

# Add brace showing local Ts combine into T_row
brace_y_top = local_t_y + local_t_height/2 + 0.15
brace_y_bottom = local_t_y - local_t_height/2 - 0.15
brace_x_right = x_start + (num_spins - 1) * x_spacing + 0.6

# Draw decorative brace on the right
ax.annotate('', xy=(brace_x_right, brace_y_bottom), xytext=(brace_x_right, brace_y_top),
           arrowprops=dict(arrowstyle=']-[', lw=2.5, color='#FF6F00'))
ax.text(brace_x_right + 0.35, local_t_y, 'combine\ninto',
       ha='left', va='center', fontsize=9, color='#FF6F00',
       weight='bold', style='italic')

# ========== Draw Row n (middle row) ==========
for i in range(num_spins):
    x_spin = x_start + i * x_spacing
    
    # Spin node
    spin_circle = Circle((x_spin, y_row_n), spin_radius,
                        color=spin_color, ec='#003366',
                        linewidth=2, zorder=4)
    ax.add_patch(spin_circle)
    
    # Horizontal bonds within row n
    if i < num_spins - 1:
        ax.plot([x_spin + spin_radius, x_start + (i+1)*x_spacing - spin_radius],
               [y_row_n, y_row_n],
               'k-', linewidth=2, zorder=3)

# ========== Draw the giant transfer matrix slab (between n and n+1) ==========
slab = FancyBboxPatch(
    (slab_x_center - slab_width/2, slab_y_center - slab_height/2),
    slab_width, slab_height,
    boxstyle="round,pad=0.1",
    facecolor=transfer_slab_color,
    edgecolor=transfer_edge_color,
    linewidth=3,
    zorder=1,
    alpha=0.7
)
ax.add_patch(slab)

# Label for transfer slab
ax.text(slab_x_center, slab_y_center, r'$T_{\mathrm{row}}$',
       ha='center', va='center', fontsize=22,
       weight='bold', color=transfer_edge_color, zorder=2)

# Add dimension label
ax.text(slab_x_center, slab_y_center - 0.35, r'$2^L \times 2^L$',
       ha='center', va='center', fontsize=12,
       style='italic', color='#8B0000', zorder=2)

# Draw vertical bonds through the slab
for i in range(num_spins):
    x_spin = x_start + i * x_spacing
    # Top part (from row n to slab)
    ax.plot([x_spin, x_spin],
           [y_row_n - spin_radius, slab_y_center + slab_height/2],
           color=bond_color, linewidth=2, zorder=2, linestyle='--', alpha=0.6)
    # Bottom part (from slab to row n+1)
    ax.plot([x_spin, x_spin],
           [slab_y_center - slab_height/2, y_row_n1 + spin_radius],
           color=bond_color, linewidth=2, zorder=2, linestyle='--', alpha=0.6)

# ========== Draw Row n+1 (bottom row) ==========
for i in range(num_spins):
    x_spin = x_start + i * x_spacing
    
    # Spin node
    spin_circle = Circle((x_spin, y_row_n1), spin_radius,
                        color=spin_color, ec='#003366',
                        linewidth=2, zorder=4)
    ax.add_patch(spin_circle)
    
    # Horizontal bonds within row n+1
    if i < num_spins - 1:
        ax.plot([x_spin + spin_radius, x_start + (i+1)*x_spacing - spin_radius],
               [y_row_n1, y_row_n1],
               'k-', linewidth=2, zorder=3)

# ========== Add row labels with braces on the left ==========
brace_x_left = x_start - 0.5

# Row n-1 label
ax.annotate('', xy=(brace_x_left, y_row_n_minus_1 - 0.25), 
           xytext=(brace_x_left, y_row_n_minus_1 + 0.25),
           arrowprops=dict(arrowstyle='|-|', lw=2, color='#555555'))
ax.text(brace_x_left - 0.25, y_row_n_minus_1, 'Row $n-1$',
       ha='right', va='center', fontsize=13, weight='bold', color='#555555')

# Row n label
ax.annotate('', xy=(brace_x_left, y_row_n - 0.25), 
           xytext=(brace_x_left, y_row_n + 0.25),
           arrowprops=dict(arrowstyle='|-|', lw=2, color='#555555'))
ax.text(brace_x_left - 0.25, y_row_n, 'Row $n$',
       ha='right', va='center', fontsize=13, weight='bold', color='#555555')

# Row n+1 label
ax.annotate('', xy=(brace_x_left, y_row_n1 - 0.25), 
           xytext=(brace_x_left, y_row_n1 + 0.25),
           arrowprops=dict(arrowstyle='|-|', lw=2, color='#555555'))
ax.text(brace_x_left - 0.25, y_row_n1, 'Row $n+1$',
       ha='right', va='center', fontsize=13, weight='bold', color='#555555')

# Add "L spins" annotation for row n-1 (top)
ax.annotate('', xy=(x_start, y_row_n_minus_1 + 0.4), 
           xytext=(x_start + (num_spins-1)*x_spacing, y_row_n_minus_1 + 0.4),
           arrowprops=dict(arrowstyle='<->', lw=2, color='#2E7D32'))
ax.text(slab_x_center, y_row_n_minus_1 + 0.65, '$L$ spins',
       ha='center', va='bottom', fontsize=12, color='#2E7D32', weight='bold')

# Add annotation emphasizing the problem (right side)
ax.text(slab_x_center + slab_width/2 + 0.3, slab_y_center + 0.25,
       'One "super-spin"',
       ha='left', va='center', fontsize=11, color='#B71C1C',
       weight='bold')
ax.text(slab_x_center + slab_width/2 + 0.3, slab_y_center - 0.25,
       r'has $2^L$ configs!',
       ha='left', va='center', fontsize=11, color='#B71C1C',
       weight='bold',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFEBEE',
                edgecolor='#B71C1C', linewidth=2))

# Set axis limits and remove axes
ax.set_xlim(brace_x_left - 1.0, slab_x_center + slab_width/2 + 2.5)
ax.set_ylim(-0.2, y_row_n_minus_1 + 1.2)
ax.axis('off')

# Tight layout
plt.tight_layout()

# Save figure
output_path = 'figures/fig_2d_row_transfer.pdf'
plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
print(f"Figure saved to: {output_path}")

# Also save as PNG for preview
output_path_png = 'figures/fig_2d_row_transfer.png'
plt.savefig(output_path_png, format='png', bbox_inches='tight', dpi=300)
print(f"Figure saved to: {output_path_png}")

plt.show()
