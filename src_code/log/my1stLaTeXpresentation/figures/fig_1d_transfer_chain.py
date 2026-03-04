"""
Generate figure: 1D chain showing state space and transfer matrix contraction
For CTMRG presentation - Part 1, Slide 1.1

Concept: 
- Each site has 2 states (up/down)
- Transfer matrix T between adjacent sites encodes interaction
- Partition function Z = Tr(T·T·T·...·T)
- Local observable ⟨σₖ⟩ = Tr(T^(k-1)·f(↑,↓)·T^(N-k)) / Z
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch
import numpy as np

# Create figure with larger size
fig, ax = plt.subplots(1, 1, figsize=(15, 6.5))
ax.set_aspect('equal')

# Colors
spin_up_color = '#E3F2FD'      # Light blue for spin up
spin_down_color = '#FFEBEE'    # Light pink for spin down
transfer_box_color = '#FFF9C4' # Light yellow for T
border_color = '#1565C0'       # Dark blue for borders
text_color = '#263238'         # Dark gray for text

# Layout parameters - increased spacing
num_sites = 4
y_center = 2.0
x_start = 0.8
site_spacing = 2.8

# Vertical offset for up/down states - increased
state_radius = 0.35
state_vertical_offset = 0.6

# Transfer matrix box size - increased
T_width = 0.9
T_height = 1.1

# ========================================
# Draw sites with up/down state options
# ========================================
for i in range(num_sites):
    x_site = x_start + i * site_spacing
    
    # Draw vertical line connecting states
    ax.plot([x_site, x_site], 
           [y_center - state_vertical_offset - state_radius - 0.1,
            y_center + state_vertical_offset + state_radius + 0.1],
           color='#BDBDBD', linewidth=1.5, linestyle='--', alpha=0.5, zorder=1)
    
    # Spin UP state (top)
    circle_up = Circle((x_site, y_center + state_vertical_offset), state_radius,
                      facecolor=spin_up_color, edgecolor=border_color,
                      linewidth=2.5, zorder=3)
    ax.add_patch(circle_up)
    ax.text(x_site, y_center + state_vertical_offset, r'$\uparrow$',
           ha='center', va='center', fontsize=24, weight='bold',
           color=border_color, zorder=4)
    
    # Spin DOWN state (bottom)
    circle_down = Circle((x_site, y_center - state_vertical_offset), state_radius,
                        facecolor=spin_down_color, edgecolor=border_color,
                        linewidth=2.5, zorder=3)
    ax.add_patch(circle_down)
    ax.text(x_site, y_center - state_vertical_offset, r'$\downarrow$',
           ha='center', va='center', fontsize=24, weight='bold',
           color=border_color, zorder=4)
    
    # Site label below
    ax.text(x_site, y_center - state_vertical_offset - 0.7,
           f'site ${i+1}$',
           ha='center', va='top', fontsize=18, color='#616161')

# ========================================
# Draw transfer matrices T between sites
# ========================================
for i in range(num_sites - 1):
    x_T = x_start + (i + 0.5) * site_spacing
    
    # Transfer matrix box
    T_box = FancyBboxPatch(
        (x_T - T_width/2, y_center - T_height/2),
        T_width, T_height,
        boxstyle="round,pad=0.08",
        facecolor=transfer_box_color,
        edgecolor=border_color,
        linewidth=2.5,
        zorder=2
    )
    ax.add_patch(T_box)
    
    # Label T
    ax.text(x_T, y_center + 0.1, r'$T$',
           ha='center', va='center', fontsize=30,
           weight='bold', color=border_color, zorder=4)
    
    # Dimension label
    ax.text(x_T, y_center - 0.35, r'$2\!\times\!2$',
           ha='center', va='center', fontsize=16,
           style='italic', color='#757575', zorder=4)
    
    # Connecting lines - thicker
    # Left connection
    x_left_site = x_start + i * site_spacing
    ax.plot([x_left_site + state_radius + 0.08, x_T - T_width/2],
           [y_center + state_vertical_offset, y_center + T_height/2 - 0.15],
           color='#78909C', linewidth=2, alpha=0.7, zorder=1)
    ax.plot([x_left_site + state_radius + 0.08, x_T - T_width/2],
           [y_center - state_vertical_offset, y_center - T_height/2 + 0.15],
           color='#78909C', linewidth=2, alpha=0.7, zorder=1)
    
    # Right connection
    x_right_site = x_start + (i + 1) * site_spacing
    ax.plot([x_T + T_width/2, x_right_site - state_radius - 0.08],
           [y_center + T_height/2 - 0.15, y_center + state_vertical_offset],
           color='#78909C', linewidth=2, alpha=0.7, zorder=1)
    ax.plot([x_T + T_width/2, x_right_site - state_radius - 0.08],
           [y_center - T_height/2 + 0.15, y_center - state_vertical_offset],
           color='#78909C', linewidth=2, alpha=0.7, zorder=1)

# ========================================
# Add trace connections (periodic boundary)
# ========================================
# Top arc from last site to first site
x_first = x_start
x_last = x_start + (num_sites - 1) * site_spacing
y_top = y_center + state_vertical_offset + state_radius + 0.05

# Add ellipsis after the last site
ellipsis_x = x_last + state_radius + 0.3
ax.text(ellipsis_x, y_center, '...', 
       ha='left', va='center', fontsize=32, 
       fontweight='bold', color='#263238')

# ========================================
# Add formula annotations
# ========================================
# Partition function Z
ax.text(x_start + (num_sites - 1) * site_spacing / 2, y_center - 2.0,
       r'$Z = \mathrm{Tr}(T \cdot T \cdot T \cdots T) = \mathrm{Tr}(T^N)$',
       ha='center', va='center', fontsize=22,
       bbox=dict(boxstyle='round,pad=0.6', facecolor='#E8F5E9',
                edgecolor='#2E7D32', linewidth=2.5))

# Commuting property
ax.text(x_start + (num_sites - 1) * site_spacing / 2, y_center + state_vertical_offset + 1.4,
       r'All $T$ identical $\Rightarrow$ commute $\Rightarrow$ diagonalize once!',
       ha='center', va='center', fontsize=20, color='#F57C00',
       weight='bold',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF3E0',
                edgecolor='#F57C00', linewidth=2))

# ========================================
# Add local observable annotation
# ========================================
# Highlight site 2 for local observable
highlight_site = 1  # 0-indexed, so site 2
x_highlight = x_start + highlight_site * site_spacing

# Draw emphasis box around site 2
emphasis_box = FancyBboxPatch(
    (x_highlight - 0.6, y_center - state_vertical_offset - state_radius - 0.2),
    1.2, 2 * state_vertical_offset + 2 * state_radius + 0.4,
    boxstyle="round,pad=0.05",
    facecolor='none',
    edgecolor='#D32F2F',
    linewidth=3,
    linestyle='--',
    zorder=5,
    alpha=0.8
)
ax.add_patch(emphasis_box)

# Label for local observable
ax.annotate(r'$f(\uparrow,\downarrow)$',
           xy=(x_highlight, y_center - state_vertical_offset - state_radius - 0.2),
           xytext=(x_highlight, y_center - 3.2),
           ha='center', fontsize=20, color='#D32F2F',
           weight='bold',
           arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=2.5))

ax.text(x_highlight + 2.0, y_center - 3.2,
       r'$\langle \sigma_k \rangle = \frac{\mathrm{Tr}(T^{k-1} \cdot f \cdot T^{N-k})}{\mathrm{Tr}(T^N)}$',
       ha='left', va='center', fontsize=18, color='#D32F2F',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFEBEE',
                edgecolor='#D32F2F', linewidth=2))

# ========================================
# Set limits and clean up
# ========================================
ax.set_xlim(-0.3, x_start + (num_sites - 1) * site_spacing + 1.2)
ax.set_ylim(y_center - 3.8, y_center + state_vertical_offset + 2.2)
ax.axis('off')

plt.tight_layout()

# Save outputs
output_pdf = 'figures/fig_1d_transfer_chain.pdf'
output_png = 'figures/fig_1d_transfer_chain.png'

plt.savefig(output_pdf, format='pdf', bbox_inches='tight', dpi=300)
print(f"Saved: {output_pdf}")

plt.savefig(output_png, format='png', bbox_inches='tight', dpi=300)
print(f"Saved: {output_png}")

plt.show()
