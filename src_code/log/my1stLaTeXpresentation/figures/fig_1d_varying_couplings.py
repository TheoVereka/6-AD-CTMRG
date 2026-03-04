import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ============================================================
# Figure: 1D Chain with Varying Couplings (Non-uniform J_i)
# ============================================================

# Figure size and layout parameters
fig, ax = plt.subplots(figsize=(15, 6.5))
ax.set_xlim(-0.5, 13.5)
ax.set_ylim(-3, 3)
ax.axis('off')

# Layout parameters (using optimized spacing from fig_1d_transfer_chain.py)
site_spacing = 2.8
state_radius = 0.35
T_width = 0.9
T_height = 1.1
y_center = 0.0

# Number of sites to show
n_sites = 5

# Starting x position
start_x = 1.0

# ============================================================
# Draw sites (circles) and couplings (with varying J labels)
# ============================================================

# Different colors for different T matrices to emphasize non-uniformity
T_colors = ['#FFD700', '#FF8C00', '#FF6347', '#FF4500', '#DC143C']  # Gold to red gradient
J_values = ['J_1', 'J_2', 'J_3', 'J_4', 'J_5']

for i in range(n_sites):
    x_pos = start_x + i * site_spacing
    
    # Draw site as circle
    circle = patches.Circle((x_pos, y_center), state_radius, 
                           edgecolor='black', facecolor='lightgray', 
                           linewidth=2.5, zorder=3)
    ax.add_patch(circle)
    
    # Site label
    ax.text(x_pos, y_center - 0.7, f'Site {i+1}', 
           ha='center', va='top', fontsize=17, color='black')
    
    # Draw T matrix between sites (except after last site)
    if i < n_sites - 1:
        T_x = x_pos + site_spacing / 2
        
        # T box with different color for each T
        T_rect = patches.FancyBboxPatch(
            (T_x - T_width/2, y_center - T_height/2), 
            T_width, T_height,
            boxstyle="round,pad=0.05", 
            edgecolor='black', 
            facecolor=T_colors[i],
            linewidth=2.5, 
            zorder=2
        )
        ax.add_patch(T_rect)
        
        # T label with subscript
        ax.text(T_x, y_center, f'$T_{i+1}$', 
               ha='center', va='center', fontsize=30, 
               fontweight='bold', color='black')
        
        # Coupling strength label above T
        ax.text(T_x, y_center + T_height/2 + 0.5, f'${J_values[i]}$', 
               ha='center', va='bottom', fontsize=24, 
               color='darkred', fontweight='bold')
        
        # Connection lines from sites to T
        # Left connection
        ax.plot([x_pos + state_radius, T_x - T_width/2], 
               [y_center, y_center], 
               'k-', linewidth=2, zorder=1)
        
        # Right connection
        ax.plot([T_x + T_width/2, x_pos + site_spacing - state_radius], 
               [y_center, y_center], 
               'k-', linewidth=2, zorder=1)

# ============================================================
# Add ellipsis to indicate continuation
# ============================================================
ellipsis_x = start_x + (n_sites - 1) * site_spacing + state_radius + 0.3
ax.text(ellipsis_x, y_center, '...', 
       ha='left', va='center', fontsize=32, 
       fontweight='bold', color='black')

# ============================================================
# Formula: Z = Tr(T₁ · T₂ · T₃ · T₄ · ...)
# ============================================================
formula_y = y_center - 2.0
formula_x = 6.5

# Green box for Z formula
formula_box = patches.FancyBboxPatch(
    (formula_x - 3.5, formula_y - 0.4), 
    7.0, 0.8,
    boxstyle="round,pad=0.1", 
    edgecolor='darkgreen', 
    facecolor='lightgreen',
    linewidth=2.5, 
    alpha=0.7,
    zorder=1
)
ax.add_patch(formula_box)

ax.text(formula_x, formula_y, 
       r'$Z = \mathrm{Tr}(T_1 \cdot T_2 \cdot T_3 \cdot T_4 \cdots)$',
       ha='center', va='center', fontsize=22, 
       fontweight='bold', color='darkgreen')

# ============================================================
# Warning box: No single eigenvalue trick!
# ============================================================
warning_y = 2.3  # Keep warning position fixed
warning_x = 6.5

# Red warning box
warning_box = patches.FancyBboxPatch(
    (warning_x - 3.8, warning_y - 0.45), 
    7.6, 0.9,
    boxstyle="round,pad=0.1", 
    edgecolor='darkred', 
    facecolor='#FFE4E1',
    linewidth=3, 
    alpha=0.8,
    zorder=1
)
ax.add_patch(warning_box)

ax.text(warning_x, warning_y, 
       r'Each $T_i$ different $\Rightarrow$ No single $\lambda^N$ trick!',
       ha='center', va='center', fontsize=21, 
       fontweight='bold', color='darkred')

# ============================================================
# Additional annotation: Contraction must be sequential
# ============================================================
annotation_y = y_center - 1.1
annotation_x = 6.5

ax.text(annotation_x, annotation_y, 
       r'Must compute: $T_1 \cdot T_2$, then $(T_1 T_2) \cdot T_3$, ...',
       ha='center', va='center', fontsize=18, 
       style='italic', color='gray')

# ============================================================
# Save figure
# ============================================================
plt.tight_layout()
plt.savefig('figures/fig_1d_varying_couplings.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/fig_1d_varying_couplings.png', format='png', bbox_inches='tight', dpi=150)
print("Saved: figures/fig_1d_varying_couplings.pdf")
print("Saved: figures/fig_1d_varying_couplings.png")
plt.close()
