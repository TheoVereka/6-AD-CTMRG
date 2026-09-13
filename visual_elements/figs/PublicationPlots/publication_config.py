"""User-editable controls for the publication figures.

All point keys are ``(J2, D)``.  J2 comparisons are rounded to six decimal
places.  Global bans remove points from plots, fits, and exported data.  Fit
bans remove points only from the fit/statistical extrapolation of that figure.
"""

from pathlib import Path


HERE = Path(__file__).resolve().parent
NEEL_DATA_ROOT = Path(
    r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\D345678910"
)
TWOC3_DATA_ROOT = Path(
    r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary"
)
PROCESSED_OUTPUT_DIR = Path(
    r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\processed\publicationPlots"
)
FIGURE_OUTPUT_DIR = HERE / "figures"
ARCHIVED_FIGURE_OUTPUT_DIR = FIGURE_OUTPUT_DIR / "archivedFigures"
STYLE_PATH = HERE / "plottingStyle" / "everyday_stylesheet.mplstyle"

# These numbered PDFs are retained for reference but kept out of the main
# publication-figure directory.  The bis figures deliberately remain main.
ARCHIVED_FIGURES = {
    3, 4, 5,
    8, 9, 10, 11, 12,
    18, 19, 20, 21,
    23, 24, 25, 26,
}

FIGURE02_J2_RANGES = {
    "a": (0.200, 0.235),
    "b": (0.240, 0.265),
    "c": (0.270, 0.280),
}

# Exact global exclusions.  Keep empty to use every discovered observable.
GLOBAL_BANS = {
    "Neel": {
        (0.275, 6),
        (0.235, 4),#
    },
    "2C3": {(0.265, 3), 
    },
}

# Whole-D exclusions for each ansatz.
GLOBAL_BANNED_DS = {
    "Neel": {3},
    "2C3": {3, 4},
}

# Observable-specific lower-D cutoffs, applied consistently to plotting,
# fitting/statistics, and each figure's exported CSV rows.
OBSERVABLE_MIN_D = {
    "2C3": {
        "m": 5,
        "delta": 6,
        "E": 6,
    },
}

# J2 values hidden from every PDF but retained in exported raw data, errors,
# and fit-parameter tables.
PLOT_BANNED_J2 = {
    "Neel": {0.0, 0.32},
    "2C3": {0.0},
}

# Independent exclusions for every figure that performs a fit/statistic.
# Points remain visible as raw data.  The mapping is separated by ansatz so a
# Neel-m ban in a twin-axis figure can never alter the 2C3 Delta statistic.
TWOC3_M_FIT_BANS = {
        
    (0.23 , 5),
    (0.235, 9),#
    (0.24 , 5),
    (0.245, 8),   

        (0.25 , 5),
        (0.265, 5), (0.265, 6), (0.265, 7), (0.265, 8),
        (0.270, 5), (0.270, 6),
        (0.275, 5), (0.275, 6), (0.275, 7), 
}

NEEL_M_EXTRAP_BANS = {(0.23 , 8),#
                      (0.24 , 4),
                      (0.245, 4),
                      (0.245, 7),
                      (0.26 , 9),
                      (0.265, 8),#
                      (0.265, 9),
                      (0.27 , 8),#
                      (0.28 , 8),}

# Shared by figures 14, 16, and 17 so their identical Delta_extrap curve is
# computed from identical points.  At most one of the original largest-three-D
# candidates may be banned for any J2.
TWOC3_DELTA_EXTRAP_BANS = set()

# Shared by figure 25 and every downstream panel that reuses the same 2C3
# gapless-energy extrapolation (figures 27 and 28).  Raw points remain visible.
TWOC3_GAPLESS_ENERGY_FIT_BANS = {(0.27, 6)}

# The combined and single-J2 comparison panels use the same physical fits as
# the corresponding ansatz-specific extrapolations.
MAGNETIZATION_COMPARISON_FIT_BANS = {
    "Neel": NEEL_M_EXTRAP_BANS,
    "2C3": TWOC3_M_FIT_BANS,
}

FIT_BANS = {
    2: {"Neel": NEEL_M_EXTRAP_BANS},
    3: {"Neel": NEEL_M_EXTRAP_BANS},
    4: {"Neel": NEEL_M_EXTRAP_BANS},
    5: {"Neel": NEEL_M_EXTRAP_BANS},
    7: {"2C3": TWOC3_M_FIT_BANS},
    8: {"2C3": TWOC3_M_FIT_BANS},
    9: {"2C3": TWOC3_M_FIT_BANS},
    10: {"2C3": TWOC3_M_FIT_BANS},
    13: {"Neel": NEEL_M_EXTRAP_BANS},
    14: {"2C3": TWOC3_DELTA_EXTRAP_BANS},
    16: {
        "Neel": NEEL_M_EXTRAP_BANS,
        "2C3": TWOC3_DELTA_EXTRAP_BANS,
    },
    17: {
        "Neel": NEEL_M_EXTRAP_BANS,
        "2C3": TWOC3_DELTA_EXTRAP_BANS,
    },
    19: {"Neel": set()}, 20: {"Neel": set()},
    22: {"Neel": set()},
    24: {"2C3": set()}, 25: {"2C3": set()},
    27: {"2C3": set()},
    28: {"Neel": set(), "2C3": set()},
}

# Data selection and numerical fit controls.
MIN_D = 3
MAX_D = 11
TWOC3_M_VS_XI_J2_MAX = 0.275
FIGURE07_VISIBLE_J2_COUNT = 4
ANSATZ_COMPARISON_J2 = (0.23, 0.235)
DELTA_STATISTIC_N_LARGEST_D = 3
ENERGY_SWITCH_J2 = 0.270
POWER_ALPHA_BOUNDS = (1.50, 3.00)
FIT_MAX_NFEV = 100_000
FIT_CURVE_POINTS = 400
PNG_DPI = 300

# Plot controls intended for quick manual tuning after the first inspection.
FIGSIZE = (6.65, 5.2)
MAIN_AXES_INCHES = (1.15, 0.85, 5.00, 4.0)
COLORBAR_EXTRA_WIDTH = 1.20
LEGEND_EXTRA_WIDTH = 3.80
TWO_COLUMN_LEGEND_EXTRA_WIDTH = 6.00
COLORBAR_GAP = 0.25
COLORBAR_WIDTH = 0.22
MARKER_SIZE = 6.0
CAPSIZE = 3.0
RAW_ALPHA_MIN = 0.25
VERTICAL_LINES_J2 = (0.24, 0.27)
