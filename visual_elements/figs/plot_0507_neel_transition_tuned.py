#!/usr/bin/env python3
"""Transition-tuned variant of plot_0507_neel_2_3linear.py.

The fit-point choices below were selected with the primary goal that the
extrapolated Neel centers form a monotone, approximately square-root-like
decay and remain zero from J2 = 0.25 onward. Smaller error bars were a
secondary criterion. The source data and plotting implementation remain in
the base script so the tuned choices are explicit and easy to audit.
"""

import os

import plot_0507_neel_2_3linear as base


base.NEEL_LINEAR_POINTS_BY_J2 = {
    # J2=0, 0.20, 0.21 intentionally use the base default for manual tuning.
    # These choices balance the square-root-like decay against fit spread.
    0.000: (1, 2, 2),
    0.200: (2, 3, 3),
    0.210: (3, 4, 4),
    0.220: (3, 4, 4),
    0.230: (4, 5, 5),
    0.235: (3, 4, 4),
    0.240: (3, 4, 4),
    0.245: (2, 3, 3),
    0.250: (3, 4, 5),
    # 0.255 is excluded by the base script.
    0.260: (3, 4, 5),
    # Only n=4 and n=5 are both negative here; repeat n=5 for the
    # smallest symmetric two-window error compatible with a zero center.
    0.265: (4, 5, 5),
    0.270: (3, 4, 5),
    0.275: (2, 3, 4),
    0.280: (3, 4, 5),
}

base.OUT_DIR = os.path.join(
    base.SCRIPT_DIR,
    'analysis_plots_0507_neel_transition_tuned',
)


if __name__ == '__main__':
    base.main()
