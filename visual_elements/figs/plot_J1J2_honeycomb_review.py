"""
Literature review + phase-diagram plots:
Spin-1/2 J1-J2 Heisenberg antiferromagnet on the HONEYCOMB lattice, 2015-2026.

VERIFIED source of every numerical boundary:
  - Albuquerque+ 2011: user-provided abstract (PRB 84, 024406 / arXiv:1102.5325)
  - Gong+ 2013  : user-provided abstract / text  (PRB 88, 165138)
  - Ganesh+ 2013: abstract / text  (PRL 110, 127203)
  - Zhu+ 2013   : abstract / text  (PRL 110, 127205)  ← from the 2020 review passage
  - Ghorbani+ 2016: abstract text  (JPCM 28, 406001 / arXiv:1603.08896)
  - Ferrari+ 2017 : user-provided example image + 2020 review passage [13]  (PRB 96, 104401)
  - Merino+ 2018  : Google Scholar snippet + 2020 review passage [7]  (PRB 97, 205112)
  - Gu+ 2022      : abstract (PRB 105, 174403) – spin DYNAMICS paper, no new boundaries
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# ① PHASE COLOUR PALETTE  (same colour = same phase across all studies)
# ---------------------------------------------------------------------------
COLORS = {
    "Néel":     "#C0392B",   # red
    "PVB":      "#E67E22",   # orange   – Plaquette Valence-Bond Solid
    "Dimer":    "#2980B9",   # blue     – Dimer/Columnar VBS
    "QSL":      "#27AE60",   # green    – Quantum Spin Liquid
    "Stripe":   "#8E44AD",   # purple   – Stripe / Collinear AFM
    "Unknown":  "#BDC3C7",   # grey     – phase not determined
}

# ---------------------------------------------------------------------------
# ② STUDY DATABASE
# ---------------------------------------------------------------------------
# Each entry:
#   label   : displayed on left axis
#   year    : publication year
#   method  : brief method string
#   ref     : journal reference
#   phases  : list of (J2/J1_lo, J2/J1_hi, phase_key, display_name)
#   notes   : extra string printed in the review
# ---------------------------------------------------------------------------
studies = [
    # ---- PRE-2015 CONTEXT (four studies: 2011–2013) -----------------------
    dict(
        label="Albuquerque 2011",
        year=2011,
        method="ED + QDM + CMF",
        ref="PRB 84, 024406",
        phases=[
            (0.00, 0.20, "Néel",   "Néel"),
            (0.20, 0.35, "PVB",    "PVB"),
            (0.35, 0.50, "Dimer",  "Staggered VBC?"),  # see notes — ambiguous in paper
        ],
        notes=(
            "Combination of exact diagonalization (ED, clusters up to 42 sites), "
            "Hamiltonian projected into nearest-neighbour short-range VB basis, "
            "effective quantum dimer model (QDM), and self-consistent cluster mean-field (CMF). "
            "Full (J2,J3) phase diagram explored; J3=0 slice quoted here. "
            "At J3=0: Néel AFM for J2/J1 < ~0.20; plaquette VBC (PVB) for ~0.20 < J2/J1 < ~0.35; "
            "staggered VBC (= dimer order) for J2/J1 > ~0.35 — though the paper explicitly notes "
            "this staggered-VBC region may alternatively reflect magnetically ordered phases that "
            "break lattice rotational symmetry, so the assignment is cautious. "
            "Boundaries are approximate (finite-size ED, no bulk extrapolation). "
            "Finds a Gutzwiller projected tight-binding wavefunction with unusually accurate energy "
            "near the Néel→PVB transition line, pointing to possible deconfined criticality."
        ),
    ),
    dict(
        label="Gong 2013",
        year=2013,
        method="DMRG",
        ref="PRB 88, 165138",
        phases=[
            (0.00, 0.22, "Néel",    "Néel"),
            (0.22, 0.25, "QSL",     "QSL?"),     # narrow possible SL window
            (0.25, 0.35, "PVB",     "PVB"),
            (0.35, 0.50, "Unknown", "?"),
        ],
        notes=(
            "DMRG with SU(2) symmetry on cylinders. Néel order for J2/J1 < 0.22; "
            "possible narrow spin-liquid or direct Néel→PVB crossover at J2/J1 ≈ 0.22–0.25; "
            "plaquette VBS (PVB) for J2/J1 ≈ 0.25–0.35. Transition nature above 0.35 not resolved."
        ),
    ),
    dict(
        label="Ganesh 2013",
        year=2013,
        method="DMRG",
        ref="PRL 110, 127203",
        phases=[
            (0.00, 0.22, "Néel",   "Néel"),
            (0.22, 0.35, "PVB",    "PVB"),
            (0.35, 0.50, "Dimer",  "Dimer"),
        ],
        notes=(
            "DMRG (van den Brink group). Both Néel→PVB (at 0.22) and PVB→Dimer (at 0.35) "
            "transitions claimed to be continuous with simultaneously vanishing spin gap and order "
            "parameters — interpreted as deconfined quantum criticality. "
            '"Dimer" here means a staggered valence-bond solid (alternating dimers).'
        ),
    ),
    dict(
        label="Zhu 2013",
        year=2013,
        method="DMRG",
        ref="PRL 110, 127205",
        phases=[
            (0.00, 0.26, "Néel",   "Néel"),
            (0.26, 0.36, "PVB",    "PVB"),
            (0.36, 0.50, "Dimer",  "Dimer"),
        ],
        notes=(
            "DMRG (White group). Weak PVB order for 0.26 < J2/J1 < 0.36; "
            "dimer (= staggered VBS) order for J2/J1 > 0.36. "
            "Transition at J2/J1 ≈ 0.26 compatible with deconfined criticality scenario."
        ),
    ),
    # ---- 2015–2026 MAIN REVIEW --------------------------------------------
    dict(
        label="Ghorbani 2016",
        year=2016,
        method="Modified Spin Wave",
        ref="JPCM 28, 406001",
        phases=[
            (0.00,  0.207, "Néel",    "Néel"),
            (0.207, 0.369, "QSL",     "QSL (gapped)"),
            # gap 0.369–0.396 not labelled in paper
            (0.396, 0.500, "Stripe",  "Stripe"),
        ],
        notes=(
            "Modified Spin Wave (MSW) theory — introduces quantum fluctuations beyond linear "
            "spin wave, treats disordered states self-consistently. "
            "Finds three phases for equal J1-couplings: "
            "(1) Néel-I for J2/J1 ≲ 0.207; "
            "(2) gapped symmetry-preserving QSL for 0.207 ≲ J2/J1 ≲ 0.369; "
            "(3) Néel-II (collinear stripe) for J2/J1 ≳ 0.396. "
            "A small unlabelled gap exists between 0.369 and 0.396."
        ),
    ),
    dict(
        label="Ferrari 2017",
        year=2017,
        method="VMC (Gutzwiller proj.)",
        ref="PRB 96, 104401",
        phases=[
            (0.00, 0.23, "Néel",   "Néel"),
            (0.23, 0.36, "PVB",    "PVB"),      # d±id SL lower in energy than Néel but above PVB
            (0.36, 0.50, "Dimer",  "Columnar"),
        ],
        notes=(
            "Variational Monte Carlo with Gutzwiller-projected bosonic and fermionic "
            "wave functions (Z2 SL, d±id Dirac SL, PVB, Columnar VBS). "
            "Phase diagram: Néel for J2/J1 < 0.23; PVB (plaquette VBS) for 0.23–0.36; "
            "Columnar VBS for J2/J1 > 0.36. "
            "The d±id SL is the lowest-energy SL state and beats the Néel state near J2/J1 ≈ 0.23, "
            "but is always higher in energy than the PVB — hence PVB is the true ground state. "
            '"Columnar VBS" here = staggered dimer order (= "Dimer" of Ganesh/Zhu 2013).'
        ),
    ),
    dict(
        label="Merino 2018",
        year=2018,
        method="Schwinger Boson MF",
        ref="PRB 97, 205112",
        phases=[
            (0.00, 0.20, "Néel",    "Néel"),
            (0.20, 0.40, "QSL",     "QSL"),     # See note — boundaries approximate at J3=0
            (0.40, 0.50, "Stripe",  "Stripe"),
        ],
        notes=(
            "Schwinger boson mean-field theory (SU(2) and SU(3)), fully unrestricted, "
            "for the J1-J2-J3 model (S=1/2 and S=1) — not a pure J1-J2 study. "
            "At J3=0 and S=1/2: QSL sandwiched between Néel and collinear (stripe) AFM; "
            "boundaries quoted from the 2020 review: ~0.20 < J2/J1 < ~0.40 (approx.). "
            "The Schwinger-boson QSL overlaps in J2/J1 with the PVB/Columnar VBS of DMRG/VMC; "
            "the method does not break translation symmetry so it cannot produce VBS order."
        ),
    ),
    # ---- 2022+ DYNAMICS / OTHER -------------------------------------------
    # Gu, Yu, Li (2022) is a spin-dynamics paper; it adopts its phase diagram from
    # prior literature and does NOT determine new phase boundaries.
    # Included in review text only.
]

# ---- Gu 2022 note (not plotted as a phase diagram) -----------------------
GU2022_NOTE = (
    "Gu, Yu, Li (2022) — PRB 105, 174403 — Spin cluster perturbation theory (SCPT) "
    "on a 24-site C6-symmetric cluster. "
    "Computes dynamical spin structure factors in all phases (Néel, PVB, Dimer VBS, Stripe). "
    "Not a phase-boundary study: phase boundaries are taken from prior literature. "
    "Identifies broad spinon continua in the Néel phase near the BZ corner; "
    "triplon dispersions in the VBS phases; gap at the M point in the stripe phase. "
    "Results consistent with INS experiments on YbCl3 and YbBr3."
)

# ===========================================================================
# ③ INDIVIDUAL PHASE-DIAGRAM PLOTS (one per study)
# ===========================================================================
def draw_phase_bar(ax, phases, y=0.5, bar_h=0.35, xlim=(0, 0.5)):
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1)
    for (lo, hi, key, name) in phases:
        ax.barh(y, hi - lo, left=lo, height=bar_h,
                color=COLORS[key], edgecolor="white", linewidth=0.8)
        mid = (lo + hi) / 2
        if hi - lo > 0.025:
            ax.text(mid, y, name, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color="white",
                    clip_on=True)
    # transition markers
    boundaries = sorted({lo for (lo, hi, k, n) in phases if lo > 0}
                        | {hi for (lo, hi, k, n) in phases if hi < xlim[1]})
    for b in boundaries:
        ax.axvline(b, color="black", lw=1.2, ls="--", alpha=0.6)
        ax.text(b, 0.91, f"{b:.3g}", ha="center", va="bottom",
                fontsize=7, color="black")


n_studies = len(studies)
fig_ind, axes = plt.subplots(n_studies, 1, figsize=(10, 1.4 * n_studies),
                             sharex=True, gridspec_kw={"hspace": 0.85})

for i, (s, ax) in enumerate(zip(studies, axes)):
    draw_phase_bar(ax, s["phases"])
    ax.set_yticks([])
    ax.set_ylabel(f"{s['label']}\n({s['method']})", fontsize=8.5,
                  rotation=0, labelpad=80, va="center")
    ax.set_title(f"{s['ref']}", fontsize=7.5, loc="right", pad=2, color="#444")

axes[-1].set_xlabel(r"$J_2/J_1$", fontsize=11)
fig_ind.suptitle(r"J1–J2 Heisenberg Honeycomb Lattice — Individual Phase Diagrams",
                 fontsize=11, y=1.01)

# legend
handles = [mpatches.Patch(color=COLORS[k], label=k)
           for k in ["Néel", "PVB", "Dimer", "QSL", "Stripe", "Unknown"]]
label_names = {
    "Néel":    "Néel AFM",
    "PVB":     "PVB (Plaquette VBS)",
    "Dimer":   "Dimer/Columnar VBS",
    "QSL":     "QSL / Spin Liquid",
    "Stripe":  "Stripe AFM",
    "Unknown": "not determined",
}
handles2 = [mpatches.Patch(color=COLORS[k], label=label_names[k])
            for k in ["Néel", "PVB", "Dimer", "QSL", "Stripe", "Unknown"]]
fig_ind.legend(handles=handles2, loc="lower center", ncol=3,
               fontsize=8, bbox_to_anchor=(0.5, -0.06), frameon=True)

plt.tight_layout()
fig_ind.savefig("J1J2_honeycomb_individual.pdf", bbox_inches="tight")
fig_ind.savefig("J1J2_honeycomb_individual.png", bbox_inches="tight", dpi=180)
print("Saved individual plots.")

# ===========================================================================
# ④ COMBINED STACKED PHASE-DIAGRAM PLOT
# ===========================================================================
fig_stk, ax_stk = plt.subplots(figsize=(10, 0.85 * n_studies + 1.2))

bar_height = 0.60
y_gap      = 1.0          # spacing between rows
xlim       = (0.0, 0.50)

ax_stk.set_xlim(*xlim)
ax_stk.set_ylim(-0.5, n_studies * y_gap)
ax_stk.set_yticks([])
ax_stk.set_xlabel(r"$J_2/J_1$", fontsize=12, labelpad=6)
ax_stk.set_title(
    r"Spin-1/2 $J_1$–$J_2$ Heisenberg model on the honeycomb lattice"
    "\n(stacked comparison — same colour = same phase)",
    fontsize=10.5, pad=8)

# draw vertical reference lines at 0.22, 0.23, 0.26, 0.36 for quick orientation
for xref in [0.20, 0.23, 0.26, 0.36]:
    ax_stk.axvline(xref, color="grey", lw=0.6, ls=":", alpha=0.5)

for i, s in enumerate(studies):
    y_center = (n_studies - 1 - i) * y_gap

    # coloured segments
    for (lo, hi, key, name) in s["phases"]:
        ax_stk.barh(y_center, hi - lo, left=lo, height=bar_height,
                    color=COLORS[key], edgecolor="white", linewidth=0.7)
        if hi - lo > 0.025:
            mid = (lo + hi) / 2
            ax_stk.text(mid, y_center, name,
                        ha="center", va="center",
                        fontsize=7.5, fontweight="bold", color="white",
                        clip_on=True)

    # transition boundary ticks on top of each bar
    boundaries_i = sorted(
        {lo for (lo, hi, k, n) in s["phases"] if lo > 0}
        | {hi for (lo, hi, k, n) in s["phases"] if hi < xlim[1]}
    )
    for b in boundaries_i:
        ax_stk.plot(b, y_center + bar_height / 2 + 0.02, "v",
                    ms=4, color="black", zorder=5)
        ax_stk.text(b, y_center + bar_height / 2 + 0.12,
                    f"{b:.3g}", ha="center", va="bottom",
                    fontsize=6.5, color="#222")

    # left-side label: "Author Year  (method)"
    ax_stk.text(-0.005, y_center,
                f"{s['label']}  ({s['method']})",
                ha="right", va="center", fontsize=8.5,
                transform=ax_stk.get_yaxis_transform())

# divider between pre-2015 (4 studies) and main review
div_y = (n_studies - 5) * y_gap + y_gap * 0.5
ax_stk.axhline(div_y, color="black", lw=0.8, ls="--", alpha=0.5)
ax_stk.text(0.495, div_y + 0.05, "2015+", ha="right", va="bottom",
            fontsize=7.5, color="grey",
            transform=ax_stk.transData)
ax_stk.text(0.495, div_y - 0.18, "pre-2015", ha="right", va="top",
            fontsize=7.5, color="grey",
            transform=ax_stk.transData)

# legend
ax_stk.legend(handles=handles2, loc="lower center", ncol=3,
              fontsize=8.5, bbox_to_anchor=(0.5, -0.22), frameon=True)

# # x-axis tick marks
ax_stk.set_xticks(np.arange(0, 0.51, 0.05))
ax_stk.xaxis.set_tick_params(labelsize=9)

plt.tight_layout()
fig_stk.savefig("J1J2_honeycomb_stacked.pdf", bbox_inches="tight")
fig_stk.savefig("J1J2_honeycomb_stacked.png", bbox_inches="tight", dpi=180)
print("Saved stacked comparison plot.")

plt.show()

# ===========================================================================
# ⑤ PRINTED LITERATURE REVIEW
# ===========================================================================
SEP = "=" * 78

lines = [
    SEP,
    "LITERATURE REVIEW: Spin-1/2 J1-J2 Heisenberg Antiferromagnet on the Honeycomb Lattice",
    "Focus period: 2015-2026  (pre-2015 key studies included for context)",
    SEP,
    "",
    "MODEL",
    "  H = J1 sum_<ij> S_i.S_j + J2 sum_<<ij>> S_i.S_j     (J1, J2 > 0)",
    "  Honeycomb lattice, coordination number z=3.",
    "",
    SEP,
    "PRE-2015 CONTEXT (not main focus, included for comparison)",
    SEP,
    "",
    "1. Albuquerque, Schwandt, Hetenyi, Capponi, Mambrini, Lauchli (2011) -- PRB 84, 024406",
    "   arXiv  : 1102.5325",
    "   Method : Combination of exact diagonalization (ED, clusters up to 42 sites),",
    "            Hamiltonian projected into nearest-neighbour short-range VB basis,",
    "            effective quantum dimer model (QDM), and self-consistent cluster mean-field (CMF).",
    "            Full (J2,J3) phase diagram explored; J3=0 slice quoted below.",
    "   Phases : (approximate, finite-size ED; no thermodynamic-limit extrapolation)",
    "            Neel AFM for J2/J1 < ~0.20;",
    "            Plaquette VBC (PVB) for ~0.20 < J2/J1 < ~0.35;",
    "            Staggered VBC (dimer order) for J2/J1 > ~0.35.",
    "            CAVEAT: the paper explicitly states the staggered-VBC region may alternatively",
    "            be a magnetically ordered phase with broken rotational symmetry -- assignment",
    "            is cautious.",
    "   Special: Finds Gutzwiller projected tight-binding wavefunction with remarkably accurate",
    "            energies near the Neel->PVB transition line -- consistent with deconfined",
    "            criticality at this transition.",
    "",
    "2. Gong, Sheng, Motrunich, Fisher (2013) -- PRB 88, 165138",
    "   Method : DMRG with SU(2) symmetry on cylinders (width up to ~15) and torus N=2x6x6.",
    "   Phases : Neel AFM for J2/J1 < 0.22 (Neel order extrapolates to zero at ~0.22);",
    "            possible spin-liquid for 0.22 < J2/J1 < 0.25 (both spin and dimer order vanish);",
    "            Plaquette VBS (PVB) for 0.25 < J2/J1 < 0.35 (fast-growing PVB decay length);",
    "            nature above 0.35 unresolved.",
    "",
    "3. Ganesh, van den Brink, Nishimoto (2013) -- PRL 110, 127203",
    "   Method : DMRG.",
    "   Phases : Neel AFM for J2/J1 < 0.22; PVB for 0.22 < J2/J1 < 0.35;",
    "            Dimer VBS for J2/J1 > 0.35.",
    "   Special: Both transitions (0.22 and 0.35) interpreted as deconfined quantum critical",
    "            points with simultaneously vanishing spin gap and order parameters.",
    "",
    "4. Zhu, Huse, White (2013) -- PRL 110, 127205",
    "   Method : DMRG.",
    "   Phases : Neel AFM for J2/J1 < 0.26; (weak) PVB for 0.26 < J2/J1 < 0.36;",
    "            Dimer (staggered VBS) for J2/J1 > 0.36.",
    "",
    SEP,
    "MAIN REVIEW: 2015-2026",
    SEP,
    "",
    "5. Ghorbani, Shahbazi, Mosadeq (2016) -- J. Phys.: Condens. Matter 28, 406001",
    "   arXiv  : 1603.08896",
    "   Method : Modified Spin Wave (MSW) theory. Quantum and thermal fluctuations treated",
    "            self-consistently via bosonic field equations; can describe magnetically",
    "            disordered (QSL-like) states without breaking lattice symmetry.",
    "   Phases : (i)  Neel-I (staggered AFM) for J2/J1 < 0.207;",
    "            (ii) Gapped magnetically-disordered phase preserving ALL symmetries",
    "                 (= quantum spin liquid by definition) for 0.207 < J2/J1 < 0.369;",
    "            (iii) Small transition gap 0.369-0.396;",
    "            (iv) Neel-II (collinear stripe AFM, '2 of 3 NN antiparallel') for J2/J1 > 0.396.",
    "   Note   : The MSW cannot distinguish between a true QSL and a VBS; the disordered",
    "            phase (ii) may correspond to the PVB found by DMRG/VMC.",
    "",
    "6. Ferrari, Bieri, Becca (2017) -- PRB 96, 104401",
    "   Method : Variational Monte Carlo (VMC) with Gutzwiller-projected bosonic and fermionic",
    "            wave functions. Candidates: Z2 SL (gapped), U(1) Dirac SL, d+id Dirac SL,",
    "            plaquette VBS (PVB), columnar VBS.",
    "   Phases : Neel AFM for J2/J1 < 0.23;",
    "            Plaquette VBS (PVB) for 0.23 < J2/J1 < 0.36;",
    "            Columnar VBS for J2/J1 > 0.36.",
    "   Note   : The d+/-id Dirac SL is energetically competitive with Neel near J2/J1 ~ 0.23",
    "            (lower energy than Neel) but always higher than PVB -- no true SL ground state.",
    "            'Columnar VBS' = staggered-dimer pattern; same phase as 'Dimer' of Ganesh/Zhu.",
    "",
    "7. Merino, Ralko (2018) -- PRB 97, 205112",
    "   arXiv  : 1801.07042",
    "   Method : Schwinger boson mean-field theory (SU(2) and SU(3) representations),",
    "            fully unrestricted Ansatz, applied to the J1-J2-J3 Heisenberg model at S=1/2",
    "            and S=1. Phase diagram explored in the (J2, J3) plane.",
    "   Phases at J3=0, S=1/2 (approximate): Neel AFM for J2/J1 < ~0.20;",
    "            QSL sandwiched between Neel and collinear (stripe) AFM at ~0.20 < J2/J1 < ~0.40",
    "            (boundaries approximate; Google Scholar snippet suggests upper bound ~0.32-0.40);",
    "            Stripe (collinear) AFM for J2/J1 > ~0.40.",
    "   Note   : (a) Schwinger-boson MF cannot break translational symmetry, so VBS states are",
    "            inaccessible; the 'QSL' likely overlaps with the PVB/Columnar VBS of DMRG/VMC.",
    "            (b) For S=1, the QSL is quickly destroyed by weaker quantum fluctuations.",
    "",
    "8. Gu, Yu, Li (2022) -- PRB 105, 174403     [DYNAMICS study, NOT a phase-diagram paper]",
    "   arXiv  : 2205.12822",
    "   Method : Spin cluster perturbation theory (SCPT) on a 24-site C6-symmetric cluster",
    "            via exact diagonalization of the cluster, then coupled to the bath perturbatively",
    "            to compute the dynamical spin structure factor S(q,w).",
    "   Focus  : Dynamical properties (spin spectra) in all known phases; phase boundaries taken",
    "            from prior literature (Neel, PVB, Dimer VBS, Stripe AFM).",
    "   Key findings: (i) Neel phase shows a dome-shaped continuum near the 2nd BZ and strong",
    "            continuum at the BZ corner -- hallmarks of fractionalized spinon excitations.",
    "            (ii) VBS phases show strong spinon continua coexisting with triplon modes.",
    "            (iii) Stripe phase shows a spin gap at the M point (unlike linear spin-wave).",
    "            (iv) Neel-phase spectral features consistent with INS on YbCl3 and YbBr3.",
    "",
    SEP,
    "CROSS-STUDY PHASE IDENTIFICATION",
    SEP,
    "",
    "A) 'Plaquette VBS' = 'PVB' = 'plaquette valence-bond solid':",
    "   Same phase in: Gong 2013, Ganesh 2013, Zhu 2013, Ferrari 2017.",
    "   Hexagonal singlets form on alternate honeycomb plaquettes; doubles unit cell.",
    "",
    "B) 'Dimer VBS' = 'Columnar VBS' = 'staggered VBS':",
    "   Same phase in: Ganesh 2013 ('Dimer'), Zhu 2013 ('Dimer'), Ferrari 2017 ('Columnar VBS').",
    "   Bonds dimerize in a columnar/staggered pattern; called 'Columnar VBC' in some texts.",
    "   NOT the same as PVB -- PVB has full hexagonal resonance, Dimer/Columnar has bond dimers.",
    "",
    "C) QSL in Ghorbani 2016 and Merino 2018 overlaps in J2/J1 range with PVB+Dimer of DMRG/VMC.",
    "   These methods (modified spin wave, Schwinger boson) cannot resolve VBS order;",
    "   the 'QSL' they find likely corresponds to (partially) the PVB+Dimer VBS regime.",
    "",
    "D) 'Stripe AFM' (Ghorbani 2016) = 'collinear AFM' (Merino 2018) = 'Neel-II':",
    "   The classical stripe order at larger J2/J1 > ~0.36-0.50.",
    "",
    SEP,
    "SUMMARY TABLE",
    SEP,
    "",
    "Study             Method          Neel ends  Disordered/SL         2nd ordered phase",
    "----------------  --------------- ---------- --------------------- -------------------------",
    "Albuquerque+ 2011 ED+QDM+CMF      ~0.20      PVB ~0.20-~0.35(est.) Stag. VBC >~0.35 (ambig.)",
    "Gong+ 2013        DMRG            ~0.22      SL? 0.22-0.25(narrow) PVB 0.25-0.35",
    "Ganesh+ 2013      DMRG            0.22       PVB 0.22-0.35         Dimer >0.35",
    "Zhu+ 2013         DMRG            0.26       PVB 0.26-0.36         Dimer >0.36",
    "Ghorbani+ 2016    Modified SW     0.207      QSL 0.207-0.369       Stripe >0.396",
    "Ferrari+ 2017     VMC             0.23       PVB 0.23-0.36         Columnar >0.36",
    "Merino+ 2018      Schwinger Boson ~0.20      QSL ~0.20-~0.40(est.) Stripe >~0.40",
    "Gu+ 2022          SCPT (dynamics) --- (adopts prior phase diagram) ---",
    SEP,
]
review = "\n".join(lines)
print(review)
