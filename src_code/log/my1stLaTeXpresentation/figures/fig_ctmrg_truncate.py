import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ============================================================
# Figure: CTMRG Truncate Step
# Slide 3.4: Build density matrix from half-environment, SVD, truncate
# ============================================================

fig, ax = plt.subplots(figsize=(14, 8.5))
ax.set_xlim(0, 14)
ax.set_ylim(-0.5, 8.5)
ax.axis('off')

# Colors
corner_color = '#FFCCCB'
corner_edge = 'darkred'
edge_color = '#FFCCCB'
edge_edge_color = 'darkred'
proj_color = '#E6E6FA'
proj_edge = 'purple'

# ============================================================
# TOP: Half-environment forming density matrix
# ============================================================
top_cx, top_cy = 4.5, 6.8

# Draw the half-environment (upper half)
# C4' -- T4' -- C1'
#  |            |
# T3'          T1'
#  |            |
# ------ ρ ------

# C4' (top-left corner)
C_size = 0.95
C4_x, C4_y = top_cx - 2.5, top_cy
C4_rect = patches.FancyBboxPatch(
    (C4_x - C_size/2, C4_y - C_size/2),
    C_size, C_size,
    boxstyle="round,pad=0.06",
    edgecolor=corner_edge,
    facecolor=corner_color,
    linewidth=2.5,
    zorder=2
)
ax.add_patch(C4_rect)
ax.text(C4_x, C4_y, r"$C'_4$", ha='center', va='center',
       fontsize=16, fontweight='bold', color=corner_edge, zorder=3)

# T4' (top edge, horizontal)
T_w, T_h = 1.1, 0.6
T4_x, T4_y = top_cx, top_cy
T4_rect = patches.FancyBboxPatch(
    (T4_x - T_w/2, T4_y - T_h/2),
    T_w, T_h,
    boxstyle="round,pad=0.06",
    edgecolor=edge_edge_color,
    facecolor=edge_color,
    linewidth=2.5,
    zorder=2
)
ax.add_patch(T4_rect)
ax.text(T4_x, T4_y, r"$T'_4$", ha='center', va='center',
       fontsize=14, fontweight='bold', color=edge_edge_color, zorder=3)

# C1' (top-right corner)
C1_x, C1_y = top_cx + 2.5, top_cy
C1_rect = patches.FancyBboxPatch(
    (C1_x - C_size/2, C1_y - C_size/2),
    C_size, C_size,
    boxstyle="round,pad=0.06",
    edgecolor=corner_edge,
    facecolor=corner_color,
    linewidth=2.5,
    zorder=2
)
ax.add_patch(C1_rect)
ax.text(C1_x, C1_y, r"$C'_1$", ha='center', va='center',
       fontsize=16, fontweight='bold', color=corner_edge, zorder=3)

# T3' (left edge, vertical)
T3_x, T3_y = top_cx - 2.5, top_cy - 1.2
T3_rect = patches.FancyBboxPatch(
    (T3_x - T_h/2, T3_y - T_w/2),
    T_h, T_w,
    boxstyle="round,pad=0.06",
    edgecolor=edge_edge_color,
    facecolor=edge_color,
    linewidth=2.5,
    zorder=2
)
ax.add_patch(T3_rect)
ax.text(T3_x, T3_y, r"$T'_3$", ha='center', va='center',
       fontsize=14, fontweight='bold', color=edge_edge_color, zorder=3)

# T1' (right edge, vertical)
T1_x, T1_y = top_cx + 2.5, top_cy - 1.2
T1_rect = patches.FancyBboxPatch(
    (T1_x - T_h/2, T1_y - T_w/2),
    T_h, T_w,
    boxstyle="round,pad=0.06",
    edgecolor=edge_edge_color,
    facecolor=edge_color,
    linewidth=2.5,
    zorder=2
)
ax.add_patch(T1_rect)
ax.text(T1_x, T1_y, r"$T'_1$", ha='center', va='center',
       fontsize=14, fontweight='bold', color=edge_edge_color, zorder=3)

# Bonds
bond_width = 3
ax.plot([C4_x + C_size/2, T4_x - T_w/2], [C4_y, T4_y], 'k-', linewidth=bond_width, zorder=1)
ax.plot([T4_x + T_w/2, C1_x - C_size/2], [T4_y, C1_y], 'k-', linewidth=bond_width, zorder=1)
ax.plot([C4_x, T3_x], [C4_y - C_size/2, T3_y + T_w/2], 'k-', linewidth=bond_width, zorder=1)
ax.plot([C1_x, T1_x], [C1_y - C_size/2, T1_y + T_w/2], 'k-', linewidth=bond_width, zorder=1)

# Open bonds at bottom (forming ρ)
ax.plot([T3_x, T3_x], [T3_y - T_w/2, T3_y - T_w/2 - 0.5], 'k-', linewidth=bond_width, zorder=1)
ax.plot([T1_x, T1_x], [T1_y - T_w/2, T1_y - T_w/2 - 0.5], 'k-', linewidth=bond_width, zorder=1)

# Density matrix label
ax.text(top_cx, top_cy - 2.2, r'$\rho$ (half-environment)',
       ha='center', va='center', fontsize=16, fontweight='bold', color='darkblue')

# Brace for the open bonds
ax.annotate('', xy=(T3_x - 0.3, T3_y - T_w/2 - 0.35), 
           xytext=(T1_x + 0.3, T1_y - T_w/2 - 0.35),
           arrowprops=dict(arrowstyle='-', connectionstyle='bar,fraction=-0.12',
                          color='darkblue', lw=2))

# ============================================================
# ARROW: SVD
# ============================================================
ax.annotate('', xy=(9.5, 6.3), xytext=(7.5, 6.3),
           arrowprops=dict(arrowstyle='->', color='purple', lw=4))
ax.text(8.5, 6.8, 'SVD', ha='center', va='center',
       fontsize=20, fontweight='bold', color='purple')

# ============================================================
# RIGHT: SVD result and projector
# ============================================================
right_cx, right_cy = 11.5, 6.5

ax.text(right_cx, right_cy + 0.8, r'$\rho = U \Sigma V^\dagger$',
       ha='center', va='center', fontsize=20, fontweight='bold', color='purple')

ax.text(right_cx, right_cy + 0.1, r'Keep $\chi$ largest:',
       ha='center', va='center', fontsize=16, color='#333333')

ax.text(right_cx, right_cy - 0.6, r'$P = U_{:\chi}$',
       ha='center', va='center', fontsize=22, fontweight='bold', color='purple',
       bbox=dict(boxstyle='round,pad=0.3', facecolor=proj_color,
                edgecolor=proj_edge, linewidth=2))

# ============================================================
# BOTTOM: Truncation application
# ============================================================
bot_cy = 1.5

# Before truncation (C')
ax.text(2.5, bot_cy + 2.0, 'Apply Projector $P$:', ha='center', va='center',
       fontsize=20, fontweight='bold', color='#333333')

Cp_size = 1.5
Cp_x, Cp_y = 2.5, bot_cy
Cp_rect = patches.FancyBboxPatch(
    (Cp_x - Cp_size/2, Cp_y - Cp_size/2),
    Cp_size, Cp_size,
    boxstyle="round,pad=0.08",
    edgecolor='darkred',
    facecolor='#FFCCCB',
    linewidth=3,
    zorder=2
)
ax.add_patch(Cp_rect)
ax.text(Cp_x, Cp_y, r"$C'$", ha='center', va='center',
       fontsize=22, fontweight='bold', color='darkred', zorder=3)
ax.text(Cp_x, Cp_y - 1.2, r'$(\chi d) \times (\chi d)$',
       ha='center', va='center', fontsize=14, color='darkred')

# Arrow
ax.annotate('', xy=(5.5, bot_cy), xytext=(4.0, bot_cy),
           arrowprops=dict(arrowstyle='->', color='purple', lw=3))

# P† C' P
ax.text(7, bot_cy, r"$P^\dagger C' P$", ha='center', va='center',
       fontsize=22, fontweight='bold', color='purple')

# Arrow
ax.annotate('', xy=(9.5, bot_cy), xytext=(8.5, bot_cy),
           arrowprops=dict(arrowstyle='->', color='darkgreen', lw=3))

# After truncation (C_new)
Cnew_size = 1.3
Cnew_x, Cnew_y = 11, bot_cy
Cnew_rect = patches.FancyBboxPatch(
    (Cnew_x - Cnew_size/2, Cnew_y - Cnew_size/2),
    Cnew_size, Cnew_size,
    boxstyle="round,pad=0.08",
    edgecolor='darkgreen',
    facecolor='#98FB98',
    linewidth=3,
    zorder=2
)
ax.add_patch(Cnew_rect)
ax.text(Cnew_x, Cnew_y, r"$C_{\mathrm{new}}$", ha='center', va='center',
       fontsize=18, fontweight='bold', color='darkgreen', zorder=3)
ax.text(Cnew_x, Cnew_y - 1.2, r'$\chi \times \chi$',
       ha='center', va='center', fontsize=14, color='darkgreen')

# Similarly for T
ax.text(7, bot_cy - 1.6, r"Similarly: $T_{\mathrm{new}} = P^\dagger T' P$",
       ha='center', va='center', fontsize=18, fontweight='bold', color='#555555')

# ============================================================
# Top title
# ============================================================
ax.text(7, 8.2,
       r"Truncation: Use environment to find optimal projector $P$",
       ha='center', va='center', fontsize=22, fontweight='bold',
       color='white',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='purple',
                edgecolor='purple', linewidth=2.5))

# ============================================================
# Save figure
# ============================================================
plt.tight_layout()
plt.savefig('fig_ctmrg_truncate.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('fig_ctmrg_truncate.png', format='png', bbox_inches='tight', dpi=300)
print("Saved: figures/fig_ctmrg_truncate.pdf")
print("Saved: figures/fig_ctmrg_truncate.png")
plt.close()
