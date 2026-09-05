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
STYLE_PATH = HERE / "plottingStyle" / "everyday_stylesheet.mplstyle"

# Exact global exclusions.  Keep empty to use every discovered observable.
GLOBAL_BANS = {
    "Neel": {
        # (0.250, 7),
    },
    "2C3": {
        # (0.265, 8),
    },
}

# J2 values hidden from every PDF but retained in exported raw data, errors,
# and fit-parameter tables.
PLOT_BANNED_J2 = {0.0}

# Independent exclusions for every figure that performs a fit/statistic.
# Points remain visible as raw data.  Figure 14 is deliberately checked to
# contain at most one entry, as its statistic uses the largest three D values.
TWOC3_M_FIT_BANS = {
    # Remove only the non-monotone low-D prefixes at the higher J2 values.
    (0.265, 3), (0.265, 4), (0.265, 5), (0.265, 6),
    (0.270, 3), (0.270, 4), (0.270, 5), (0.270, 6),
    (0.275, 3), (0.275, 4), (0.275, 5), (0.275, 6), (0.275, 7),
}

FIT_BANS = {
    2: set(), 3: set(), 4: set(), 5: set(),
    7: set(TWOC3_M_FIT_BANS), 8: set(TWOC3_M_FIT_BANS),
    9: set(TWOC3_M_FIT_BANS), 10: set(TWOC3_M_FIT_BANS),
    13: set(), 14: set(), 16: set(), 17: {(0.255, 4)},
    19: set(), 20: set(), 22: set(),
    24: set(), 25: set(), 27: set(), 28: set(),
}

# Data selection and numerical fit controls.
MIN_D = 3
MAX_D = 11
TWOC3_M_VS_XI_J2_MAX = 0.275
DELTA_STATISTIC_N_LARGEST_D = 3
ENERGY_SWITCH_J2 = 0.270
POWER_ALPHA_BOUNDS = (1.50, 3.00)
FIT_MAX_NFEV = 100_000
FIT_CURVE_POINTS = 400

# Plot controls intended for quick manual tuning after the first inspection.
FIGSIZE = (7.2, 5.2)
MARKER_SIZE = 6.0
CAPSIZE = 3.0
RAW_ALPHA_MIN = 0.25
VERTICAL_LINES_J2 = (0.243, 0.272)
