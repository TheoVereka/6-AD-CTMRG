import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ============================================================
# Figure: CTMRG Grow Step (Absorption)
# Slide 3.3: Corner absorbs edge and local tensor
# ============================================================

fig, ax = plt.subplots(figsize=(14, 7.5))
ax.set_xlim(0, 14)
ax.set_ylim(0.5, 8)
ax.axis('off')

# Colors
corner_color = '#98FB98'
corner_edge = 'darkgreen'
edge_color = '#87CEEB'
edge_edge_color = '#1565C0'
local_color = '#FFD700'
local_edge_color = '#B8860B'

# ============================================================
# LEFT SIDE: Before absorption
# ============================================================
left_cx, left_cy = 3.5, 4.5

# Corner C (top-left)
C_size = 1.4
C_x, C_y = left_cx - 1.5, left_cy + 1.5
C_rect = patches.FancyBboxPatch(
    (C_x - C_size/2, C_y - C_size/2),
    C_size, C_size,
    boxstyle="round,pad=0.08",
    edgecolor=corner_edge,
    facecolor=corner_color,
    linewidth=3,
    zorder=2
)
ax.add_patch(C_rect)
ax.text(C_x, C_y, r'$C$', ha='center', va='center',
       fontsize=24, fontweight='bold', color=corner_edge, zorder=3)

# Edge T1 (top, horizontal)
T1_w, T1_h = 1.8, 0.9
T1_x, T1_y = left_cx + 1.0, left_cy + 1.5
T1_rect = patches.FancyBboxPatch(
    (T1_x - T1_w/2, T1_y - T1_h/2),
    T1_w, T1_h,
    boxstyle="round,pad=0.08",
    edgecolor=edge_edge_color,
    facecolor=edge_color,
    linewidth=3,
    zorder=2
)
ax.add_patch(T1_rect)
ax.text(T1_x, T1_y, r'$T$', ha='center', va='center',
       fontsize=22, fontweight='bold', color=edge_edge_color, zorder=3)

# Edge T4 (left, vertical)
T4_w, T4_h = 0.9, 1.8
T4_x, T4_y = left_cx - 1.5, left_cy - 1.0
T4_rect = patches.FancyBboxPatch(
    (T4_x - T4_w/2, T4_y - T4_h/2),
    T4_w, T4_h,
    boxstyle="round,pad=0.08",
    edgecolor=edge_edge_color,
    facecolor=edge_color,
    linewidth=3,
    zorder=2
)
ax.add_patch(T4_rect)
ax.text(T4_x, T4_y, r'$T$', ha='center', va='center',
       fontsize=22, fontweight='bold', color=edge_edge_color, zorder=3)

# Local tensor a (center)
a_size = 1.0
a_x, a_y = left_cx + 1.0, left_cy - 1.0
a_rect = patches.FancyBboxPatch(
    (a_x - a_size/2, a_y - a_size/2),
    a_size, a_size,
    boxstyle="round,pad=0.08",
    edgecolor=local_edge_color,
    facecolor=local_color,
    linewidth=3,
    zorder=2
)
ax.add_patch(a_rect)
ax.text(a_x, a_y, r'$W$', ha='center', va='center',
       fontsize=22, fontweight='bold', color='#8B4513', zorder=3)

# Bonds connecting them
bond_width = 3.5
# C to T1
ax.plot([C_x + C_size/2, T1_x - T1_w/2], [C_y, T1_y], 'k-', linewidth=bond_width, zorder=1)
# C to T4
ax.plot([C_x, T4_x], [C_y - C_size/2, T4_y + T4_h/2], 'k-', linewidth=bond_width, zorder=1)
# T1 to a
ax.plot([T1_x, a_x], [T1_y - T1_h/2, a_y + a_size/2], 'k-', linewidth=bond_width, zorder=1)
# T4 to a
ax.plot([T4_x + T4_w/2, a_x - a_size/2], [T4_y, a_y], 'k-', linewidth=bond_width, zorder=1)

# External bonds (showing continuation)
ax.plot([T1_x + T1_w/2, T1_x + T1_w/2 + 0.8], [T1_y, T1_y], 'k-', linewidth=bond_width, zorder=1)
ax.plot([T4_x, T4_x], [T4_y - T4_h/2, T4_y - T4_h/2 - 0.8], 'k-', linewidth=bond_width, zorder=1)
ax.plot([a_x + a_size/2, a_x + a_size/2 + 0.8], [a_y, a_y], 'k-', linewidth=bond_width, zorder=1)
ax.plot([a_x, a_x], [a_y - a_size/2, a_y - a_size/2 - 0.8], 'k-', linewidth=bond_width, zorder=1)

# Ellipsis for continuation
ax.text(T1_x + T1_w/2 + 1.1, T1_y, '···', ha='center', va='center', fontsize=28, color='gray')
ax.text(T4_x, T4_y - T4_h/2 - 1.1, '⋮', ha='center', va='center', fontsize=28, color='gray')
ax.text(a_x + a_size/2 + 1.1, a_y, '···', ha='center', va='center', fontsize=28, color='gray')
ax.text(a_x, a_y - a_size/2 - 1.1, '⋮', ha='center', va='center', fontsize=28, color='gray')

# Label: "Before"
ax.text(left_cx, left_cy + 3.3, 'Before Absorption',
       ha='center', va='center', fontsize=22, fontweight='bold', color='#333333')

# Bond dimension label
ax.text(C_x + 0.3, C_y - 0.9, r'$\chi$', ha='center', va='center',
       fontsize=18, fontweight='bold', color='purple')

# ============================================================
# ARROW
# ============================================================
arrow_x = 7
ax.annotate('', xy=(arrow_x + 1.0, 4.5), xytext=(arrow_x - 1.0, 4.5),
           arrowprops=dict(arrowstyle='->', color='darkred', lw=4))
ax.text(arrow_x, 5.1, 'Absorb', ha='center', va='center',
       fontsize=20, fontweight='bold', color='darkred')

# ============================================================
# RIGHT SIDE: After absorption
# ============================================================
right_cx, right_cy = 10.5, 4.5

# New corner C' (larger, highlighting growth)
Cp_size = 2.2
Cp_x, Cp_y = right_cx - 0.8, right_cy + 0.8
Cp_rect = patches.FancyBboxPatch(
    (Cp_x - Cp_size/2, Cp_y - Cp_size/2),
    Cp_size, Cp_size,
    boxstyle="round,pad=0.1",
    edgecolor='darkred',
    facecolor='#FFCCCB',
    linewidth=4,
    zorder=2
)
ax.add_patch(Cp_rect)
ax.text(Cp_x, Cp_y, r"$C'$", ha='center', va='center',
       fontsize=26, fontweight='bold', color='darkred', zorder=3)

# New edge T1' (horizontal)
T1p_w, T1p_h = 1.8, 0.9
T1p_x, T1p_y = right_cx + 1.8, right_cy + 0.8
T1p_rect = patches.FancyBboxPatch(
    (T1p_x - T1p_w/2, T1p_y - T1p_h/2),
    T1p_w, T1p_h,
    boxstyle="round,pad=0.08",
    edgecolor='darkred',
    facecolor='#FFCCCB',
    linewidth=3,
    zorder=2
)
ax.add_patch(T1p_rect)
ax.text(T1p_x, T1p_y, r"$T'$", ha='center', va='center',
       fontsize=20, fontweight='bold', color='darkred', zorder=3)

# New edge T4' (vertical)
T4p_w, T4p_h = 0.9, 1.8
T4p_x, T4p_y = right_cx - 0.8, right_cy - 1.5
T4p_rect = patches.FancyBboxPatch(
    (T4p_x - T4p_w/2, T4p_y - T4p_h/2),
    T4p_w, T4p_h,
    boxstyle="round,pad=0.08",
    edgecolor='darkred',
    facecolor='#FFCCCB',
    linewidth=3,
    zorder=2
)
ax.add_patch(T4p_rect)
ax.text(T4p_x, T4p_y, r"$T'$", ha='center', va='center',
       fontsize=20, fontweight='bold', color='darkred', zorder=3)

# Bonds
ax.plot([Cp_x + Cp_size/2, T1p_x - T1p_w/2], [Cp_y, T1p_y], 'k-', linewidth=bond_width, zorder=1)
ax.plot([Cp_x, T4p_x], [Cp_y - Cp_size/2, T4p_y + T4p_h/2], 'k-', linewidth=bond_width, zorder=1)

# External bonds
ax.plot([T1p_x + T1p_w/2, T1p_x + T1p_w/2 + 0.6], [T1p_y, T1p_y], 'k-', linewidth=bond_width, zorder=1)
ax.plot([T4p_x, T4p_x], [T4p_y - T4p_h/2, T4p_y - T4p_h/2 - 0.6], 'k-', linewidth=bond_width, zorder=1)

# Internal bonds to center (d dimension) - from T1' going down and T4' going right
d_leg_length = 0.7
ax.plot([T1p_x, T1p_x], [T1p_y - T1p_h/2, T1p_y - T1p_h/2 - d_leg_length], 'k-', linewidth=bond_width, zorder=1)
ax.plot([T4p_x + T4p_w/2, T4p_x + T4p_w/2 + d_leg_length], [T4p_y, T4p_y], 'k-', linewidth=bond_width, zorder=1)

# d labels near the internal bonds
ax.text(T1p_x + 0.35, T1p_y - T1p_h/2 - d_leg_length/2, r'$d$', ha='center', va='center',
       fontsize=16, fontweight='bold', color='purple')
ax.text(T4p_x -5 + T4p_w/2 + d_leg_length/2, T4p_y + 0.3, r'$d$', ha='center', va='center',
       fontsize=16, fontweight='bold', color='purple')

# Ellipsis
ax.text(T1p_x + T1p_w/2 + 0.9, T1p_y, '···', ha='center', va='center', fontsize=28, color='gray')
ax.text(T4p_x, T4p_y - T4p_h/2 - 0.9, '⋮', ha='center', va='center', fontsize=28, color='gray')

# Label: "After"
ax.text(right_cx, right_cy + 3.3, 'After Absorption',
       ha='center', va='center', fontsize=22, fontweight='bold', color='#333333')

# Bond dimension label (grown!) - near the bond from C' to T1'
ax.text(Cp_x + 4.0, Cp_y + 0.3, r'$\chi \cdot d$', ha='center', va='center',
       fontsize=16, fontweight='bold', color='darkred')

# ============================================================
# Bottom message
# ============================================================
ax.text(7, 0.75,
       r"Bond dimension grows: $\chi \to \chi \cdot d$ — Need truncation!",
       ha='center', va='center', fontsize=22, fontweight='bold',
       color='white',
       bbox=dict(boxstyle='round,pad=0.4', facecolor='darkred',
                edgecolor='darkred', linewidth=2.5))

# ============================================================
# Formulas at top
# ============================================================
ax.text(3.5, 7.1, r"$C' = C \cdot T \cdot a$", ha='center', va='center',
       fontsize=20, fontweight='bold', color='darkgreen',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                edgecolor='darkgreen', linewidth=2))
ax.text(10.5, 7.1, r"$T' = T \cdot a$", ha='center', va='center',
       fontsize=20, fontweight='bold', color='darkgreen',
       bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9',
                edgecolor='darkgreen', linewidth=2))

# ============================================================
# Save figure
# ============================================================
plt.tight_layout()
plt.savefig('fig_ctmrg_grow.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('fig_ctmrg_grow.png', format='png', bbox_inches='tight', dpi=300)
print("Saved: figures/fig_ctmrg_grow.pdf")
print("Saved: figures/fig_ctmrg_grow.png")
plt.close()
