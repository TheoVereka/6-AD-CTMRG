import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# ============================================================
# Figure: 1D TMRG - Environments and Contraction
# ============================================================

fig, ax = plt.subplots(figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Layout parameters
site_radius = 0.07  # Small dots, just thicker than lines
T_size = 0.65
env_width = 2.0
env_height = 1.0
x_spacing = 0.9

# Colors
site_color = 'lightgray'
T_color = '#FFD700'
env_color = '#87CEEB'

# ============================================================
# Row 1: Original Z = Tr(T1·T2···TN) - with trace
# ============================================================
y1 = 6.2
x1_start = 3.5
n_elements = 6  # Number of site-T pairs to show

x_pos = x1_start
for i in range(n_elements):
    # Site (circle)
    circle = patches.Circle((x_pos, y1), site_radius,
                           edgecolor='black', facecolor=site_color,
                           linewidth=2.5, zorder=3)
    ax.add_patch(circle)
    
    # Transfer matrix (square)
    T_x = x_pos + site_radius + x_spacing*0.45
    T_rect = patches.FancyBboxPatch(
        (T_x - T_size/2, y1 - T_size/2),
        T_size, T_size,
        boxstyle="round,pad=0.04",
        edgecolor='black',
        facecolor=T_color,
        linewidth=2.5,
        zorder=2
    )
    ax.add_patch(T_rect)
    ax.text(T_x, y1, r'$T$', ha='center', va='center',
           fontsize=30, fontweight='bold', color='black')
    
    # Connection lines
    ax.plot([x_pos + site_radius, T_x - T_size/2],
           [y1, y1], 'k-', linewidth=2.5, zorder=1)
    if i < n_elements - 1:
        next_site_x = x_pos + site_radius + x_spacing
        ax.plot([T_x + T_size/2, next_site_x],
               [y1, y1], 'k-', linewidth=2.5, zorder=1)
    
    x_pos += site_radius + x_spacing

# Close the loop with a flat arc (trace)
first_x = x1_start - site_radius
last_x = T_x + T_size/2
arc_y = y1 - 1.2

# Draw flat arc using SOLID line segments (physical T trace)
ax.plot([first_x, first_x], [y1 - site_radius - 0.1, arc_y],
       'k-', linewidth=2.5, zorder=1)
ax.plot([first_x, last_x], [arc_y, arc_y],
       'k-', linewidth=2.5, zorder=1)
ax.plot([last_x, last_x], [arc_y, y1 - T_size/2 - 0.1],
       'k-', linewidth=2.5, zorder=1)

# Label for row 1
ax.text(x1_start - 1.2, y1, r'$Z=$', ha='right', va='center',
       fontsize=33, fontweight='bold', color='black')

# ============================================================
# Row 2: Compressed - L (χ×2) - site - R (2×χ) with χ trace
# ============================================================
y2 = 3.9
x2_center = 7.0

# Left environment (ellipse)
L_x = x2_center - 2.2
L_ellipse = patches.Ellipse((L_x, y2), env_width, env_height,
                           edgecolor='darkblue',
                           facecolor=env_color,
                           linewidth=2.5,
                           zorder=2)
ax.add_patch(L_ellipse)
ax.text(L_x, y2, r'$L$', ha='center', va='center',
       fontsize=36, fontweight='bold', color='darkblue')
ax.text(L_x, y2 - env_height/2 -0.05, r'$\chi \times 2$',
       ha='center', va='top', fontsize=20, style='italic', color='darkblue')

# Single site
site_x = x2_center
circle = patches.Circle((site_x, y2), site_radius,
                       edgecolor='black', facecolor=site_color,
                       linewidth=2.5, zorder=3)
ax.add_patch(circle)

# Right environment (ellipse)
R_x = x2_center + 2.2
R_ellipse = patches.Ellipse((R_x, y2), env_width, env_height,
                           edgecolor='darkblue',
                           facecolor=env_color,
                           linewidth=2.5,
                           zorder=2)
ax.add_patch(R_ellipse)
ax.text(R_x, y2, r'$R$', ha='center', va='center',
       fontsize=36, fontweight='bold', color='darkblue')
ax.text(R_x, y2 - env_height/2 -0.05, r'$2 \times \chi$',
       ha='center', va='top', fontsize=20, style='italic', color='darkblue')

# Connections
ax.plot([L_x + env_width/2, site_x - site_radius],
       [y2, y2], 'k-', linewidth=2.5, zorder=1)
ax.plot([site_x + site_radius, R_x - env_width/2],
       [y2, y2], 'k-', linewidth=2.5, zorder=1)

# Double arc for χ bonds trace
arc_y2 = y2 - 1.0
arc_x_left = L_x - env_width/2
arc_x_right = R_x + env_width/2

# Draw double arc
for offset in [-0.1, 0.1]:
    ax.plot([arc_x_left, arc_x_left], [y2 - env_height/2 + 0.6, arc_y2 + offset],
           '--', color='darkblue', linewidth=2.5, zorder=1)
    ax.plot([arc_x_left, arc_x_right], [arc_y2 + offset, arc_y2 + offset],
           '--', color='darkblue', linewidth=2.5, zorder=1)
    ax.plot([arc_x_right, arc_x_right], [arc_y2 + offset, y2 - env_height/2 + 0.6],
           '--', color='darkblue', linewidth=2.5, zorder=1)

# Label χ on the double arc
ax.text(x2_center, arc_y2 - 0.4, r'$\chi$',
       ha='center', va='top', fontsize=24,
       fontweight='bold', color='darkblue')

# ============================================================
# Row 3: After absorption - L' - site - T - site - T - site - R'
# ============================================================
y3 = 1.2
x3_center = 7.0

# Left environment (larger, green)
L_new_x = x3_center - 3.0
L_new_ellipse = patches.Ellipse((L_new_x, y3), env_width*1.3, env_height,
                               edgecolor='darkgreen',
                               facecolor='#98FB98',
                               linewidth=2.5,
                               zorder=2)
ax.add_patch(L_new_ellipse)
ax.text(L_new_x, y3, r"$L'$", ha='center', va='center',
       fontsize=36, fontweight='bold', color='darkgreen')

# Site 1
s1_x = L_new_x + env_width*0.65 + x_spacing*0.5
circle1 = patches.Circle((s1_x, y3), site_radius,
                        edgecolor='black', facecolor=site_color,
                        linewidth=2.5, zorder=3)
ax.add_patch(circle1)

# T1
t1_x = s1_x + site_radius + x_spacing*0.5
t1_rect = patches.FancyBboxPatch(
    (t1_x - T_size/2, y3 - T_size/2),
    T_size, T_size,
    boxstyle="round,pad=0.04",
    edgecolor='black',
    facecolor=T_color,
    linewidth=2.5,
    zorder=2
)
ax.add_patch(t1_rect)
ax.text(t1_x, y3, r'$T$', ha='center', va='center',
       fontsize=30, fontweight='bold', color='black')

# Site 2
s2_x = t1_x + T_size/2 + x_spacing*0.5
circle2 = patches.Circle((s2_x, y3), site_radius,
                        edgecolor='black', facecolor=site_color,
                        linewidth=2.5, zorder=3)
ax.add_patch(circle2)

# T2
t2_x = s2_x + site_radius + x_spacing*0.5
t2_rect = patches.FancyBboxPatch(
    (t2_x - T_size/2, y3 - T_size/2),
    T_size, T_size,
    boxstyle="round,pad=0.04",
    edgecolor='black',
    facecolor=T_color,
    linewidth=2.5,
    zorder=2
)
ax.add_patch(t2_rect)
ax.text(t2_x, y3, r'$T$', ha='center', va='center',
       fontsize=30, fontweight='bold', color='black')

# Site 3
s3_x = t2_x + T_size/2 + x_spacing*0.5
circle3 = patches.Circle((s3_x, y3), site_radius,
                        edgecolor='black', facecolor=site_color,
                        linewidth=2.5, zorder=3)
ax.add_patch(circle3)

# Right environment (larger, green)
R_new_x = x3_center + 3.0
R_new_ellipse = patches.Ellipse((R_new_x, y3), env_width*1.3, env_height,
                               edgecolor='darkgreen',
                               facecolor='#98FB98',
                               linewidth=2.5,
                               zorder=2)
ax.add_patch(R_new_ellipse)
ax.text(R_new_x, y3, r"$R'$", ha='center', va='center',
       fontsize=36, fontweight='bold', color='darkgreen')

# All connections
ax.plot([L_new_x + env_width*0.65, s1_x - site_radius],
       [y3, y3], 'k-', linewidth=2.5, zorder=1)
ax.plot([s1_x + site_radius, t1_x - T_size/2],
       [y3, y3], 'k-', linewidth=2.5, zorder=1)
ax.plot([t1_x + T_size/2, s2_x - site_radius],
       [y3, y3], 'k-', linewidth=2.5, zorder=1)
ax.plot([s2_x + site_radius, t2_x - T_size/2],
       [y3, y3], 'k-', linewidth=2.5, zorder=1)
ax.plot([t2_x + T_size/2, s3_x - site_radius],
       [y3, y3], 'k-', linewidth=2.5, zorder=1)
ax.plot([s3_x + site_radius, R_new_x - env_width*0.65],
       [y3, y3], 'k-', linewidth=2.5, zorder=1)

# Double arc for χ trace
arc_y3 = y3 - 0.8
arc_x_left3 = L_new_x - env_width*0.65
arc_x_right3 = R_new_x + env_width*0.65

for offset in [-0.1, 0.1]:
    ax.plot([arc_x_left3, arc_x_left3], [y3 - env_height/2 + 0.6, arc_y3 + offset],
           '--', color='darkgreen', linewidth=2.5, zorder=1)
    ax.plot([arc_x_left3, arc_x_right3], [arc_y3 + offset, arc_y3 + offset],
           '--', color='darkgreen', linewidth=2.5, zorder=1)
    ax.plot([arc_x_right3, arc_x_right3], [arc_y3 + offset, y3 - env_height/2 + 0.6],
           '--', color='darkgreen', linewidth=2.5, zorder=1)

# Label χ
ax.text(x3_center, arc_y3 - 0.35, r'$\chi$',
       ha='center', va='top', fontsize=24,
       fontweight='bold', color='darkgreen')

# ============================================================
# Arrows between rows
# ============================================================
arrow_x = 1.5
ax.annotate('', xy=(arrow_x+1.4, y2 + 0.5), xytext=(arrow_x+1.4, y1 - 0.8),
           arrowprops=dict(arrowstyle='->', lw=3, color='#FF4500'))
ax.text(arrow_x +0.8, (y1 + y2)/2, 'Compress',
       ha='right', va='center', fontsize=23, rotation=90,
       color='#FF4500', fontweight='bold')

ax.annotate('', xy=(arrow_x+0.6, y3 + 0.4), xytext=(arrow_x+0.6, y2 - 0.6),
           arrowprops=dict(arrowstyle='->', lw=3, color='#FF4500'))
ax.text(arrow_x, (y2 + y3)/2, 'Grow+Truncate',
       ha='right', va='center', fontsize=23, rotation=90,
       color='#FF4500', fontweight='bold')

# ============================================================
# Save figure
# ============================================================
plt.tight_layout()
plt.savefig('figures/fig_1d_tmrg_environments.pdf', format='pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/fig_1d_tmrg_environments.png', format='png', bbox_inches='tight', dpi=150)
print("Saved: figures/fig_1d_tmrg_environments.pdf")
print("Saved: figures/fig_1d_tmrg_environments.png")
plt.close()
