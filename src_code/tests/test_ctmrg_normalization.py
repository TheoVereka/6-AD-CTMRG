#!/usr/bin/env python3
"""
Test CTMRG normalization using the exact energy-function contractions
from core_unrestricted.py.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

import opt_einsum as oe
import torch

torch.set_default_dtype(torch.float64)

from core_unrestricted import (
    initialize_abcdef,
    set_dtype,
    build_heisenberg_H,
    abcdef_to_ABCDEF,
    normalize_tensor,
    CTMRG_from_init_to_stop,
    energy_expectation_nearest_neighbor_3ebadcf_bonds,
    energy_expectation_nearest_neighbor_3afcbed_bonds,
    energy_expectation_nearest_neighbor_other_3_bonds,
)


def norm_env_1(a, b, c, d, e, f, chi, D_bond, C21CD, C32EF, C13AB, T1F, T2A, T2B, T3C, T3D, T1E):
    # Exact code from energy_expectation_nearest_neighbor_3ebadcf_bonds
    T1F = T1F.reshape(chi, chi, D_bond, D_bond)
    T2A = T2A.reshape(chi, chi, D_bond, D_bond)
    T2B = T2B.reshape(chi, chi, D_bond, D_bond)
    T3C = T3C.reshape(chi, chi, D_bond, D_bond)
    T3D = T3D.reshape(chi, chi, D_bond, D_bond)
    T1E = T1E.reshape(chi, chi, D_bond, D_bond)

    open_E = oe.contract(
        "YX,MYar,abci,rstj->MbsXctij",
        C21CD,
        T1F,
        e,
        e.conj(),
        optimize=[(0, 1), (0, 1), (0, 1)],
        backend="torch",
    )
    open_D = oe.contract(
        "MYct,abci,rstj->YarMbsij",
        T3C,
        d,
        d.conj(),
        optimize=[(0, 1), (0, 1)],
        backend="torch",
    )
    open_A = oe.contract(
        "ZY,NZbs,abci,rstj->NctYarij",
        C32EF,
        T2B,
        a,
        a.conj(),
        optimize=[(0, 1), (0, 1), (0, 1)],
        backend="torch",
    )
    open_F = oe.contract(
        "NZar,abci,rstj->ZbsNctij",
        T1E,
        f,
        f.conj(),
        optimize=[(0, 1), (0, 1)],
        backend="torch",
    )
    open_C = oe.contract(
        "XZ,LXct,abci,rstj->LarZbsij",
        C13AB,
        T3D,
        c,
        c.conj(),
        optimize=[(0, 1), (0, 1), (0, 1)],
        backend="torch",
    )
    open_B = oe.contract(
        "LXbs,abci,rstj->XctLarij",
        T2A,
        b,
        b.conj(),
        optimize=[(0, 1), (0, 1)],
        backend="torch",
    )

    closed_E = oe.contract("MbsXctii->MbsXct", open_E, backend="torch").reshape(
        chi * D_bond * D_bond, chi * D_bond * D_bond
    )
    closed_D = oe.contract("YarMbsii->YarMbs", open_D, backend="torch").reshape(
        chi * D_bond * D_bond, chi * D_bond * D_bond
    )
    closed_A = oe.contract("NctYarii->NctYar", open_A, backend="torch").reshape(
        chi * D_bond * D_bond, chi * D_bond * D_bond
    )
    closed_F = oe.contract("ZbsNctii->ZbsNct", open_F, backend="torch").reshape(
        chi * D_bond * D_bond, chi * D_bond * D_bond
    )
    closed_C = oe.contract("LarZbsii->LarZbs", open_C, backend="torch").reshape(
        chi * D_bond * D_bond, chi * D_bond * D_bond
    )
    closed_B = oe.contract("XctLarii->XctLar", open_B, backend="torch").reshape(
        chi * D_bond * D_bond, chi * D_bond * D_bond
    )

    AD = torch.mm(closed_A, closed_D)
    CF = torch.mm(closed_C, closed_F)
    EB = torch.mm(closed_E, closed_B)

    return oe.contract("xy,yz,zx->", AD, EB, CF, backend="torch")


def norm_env_2(a, b, c, d, e, f, chi, D_bond, C21EB, C32AD, C13CF, T1D, T2C, T2F, T3E, T3B, T1A):
    # Exact code from energy_expectation_nearest_neighbor_3afcbed_bonds
    T1D = T1D.reshape(chi, chi, D_bond, D_bond)
    T2C = T2C.reshape(chi, chi, D_bond, D_bond)
    T2F = T2F.reshape(chi, chi, D_bond, D_bond)
    T3E = T3E.reshape(chi, chi, D_bond, D_bond)
    T3B = T3B.reshape(chi, chi, D_bond, D_bond)
    T1A = T1A.reshape(chi, chi, D_bond, D_bond)

    open_A = oe.contract(
        "YX,MYar,abci,rstj->MbsXctij",
        C21EB,
        T1D,
        a,
        a.conj(),
        optimize=[(0, 1), (0, 1), (0, 1)],
        backend="torch",
    )
    open_B = oe.contract(
        "MYct,abci,rstj->YarMbsij",
        T3E,
        b,
        b.conj(),
        optimize=[(0, 1), (0, 1)],
        backend="torch",
    )
    open_C = oe.contract(
        "ZY,NZbs,abci,rstj->NctYarij",
        C32AD,
        T2F,
        c,
        c.conj(),
        optimize=[(0, 1), (0, 1), (0, 1)],
        backend="torch",
    )
    open_D = oe.contract(
        "NZar,abci,rstj->ZbsNctij",
        T1A,
        d,
        d.conj(),
        optimize=[(0, 1), (0, 1)],
        backend="torch",
    )
    open_E = oe.contract(
        "XZ,LXct,abci,rstj->LarZbsij",
        C13CF,
        T3B,
        e,
        e.conj(),
        optimize=[(0, 1), (0, 1), (0, 1)],
        backend="torch",
    )
    open_F = oe.contract(
        "LXbs,abci,rstj->XctLarij",
        T2C,
        f,
        f.conj(),
        optimize=[(0, 1), (0, 1)],
        backend="torch",
    )

    closed_A = oe.contract("MbsXctii->MbsXct", open_A, backend="torch").reshape(
        chi * D_bond * D_bond, chi * D_bond * D_bond
    )
    closed_B = oe.contract("YarMbsii->YarMbs", open_B, backend="torch").reshape(
        chi * D_bond * D_bond, chi * D_bond * D_bond
    )
    closed_C = oe.contract("NctYarii->NctYar", open_C, backend="torch").reshape(
        chi * D_bond * D_bond, chi * D_bond * D_bond
    )
    closed_D = oe.contract("ZbsNctii->ZbsNct", open_D, backend="torch").reshape(
        chi * D_bond * D_bond, chi * D_bond * D_bond
    )
    closed_E = oe.contract("LarZbsii->LarZbs", open_E, backend="torch").reshape(
        chi * D_bond * D_bond, chi * D_bond * D_bond
    )
    closed_F = oe.contract("XctLarii->XctLar", open_F, backend="torch").reshape(
        chi * D_bond * D_bond, chi * D_bond * D_bond
    )

    CB = torch.mm(closed_C, closed_B)
    ED = torch.mm(closed_E, closed_D)
    AF = torch.mm(closed_A, closed_F)

    return oe.contract("xy,yz,zx->", CB, AF, ED, backend="torch")


def norm_env_3(a, b, c, d, e, f, chi, D_bond, C21AF, C32CB, C13ED, T1B, T2E, T2D, T3A, T3F, T1C):
    # Exact code from energy_expectation_nearest_neighbor_other_3_bonds
    T1B = T1B.reshape(chi, chi, D_bond, D_bond)
    T2E = T2E.reshape(chi, chi, D_bond, D_bond)
    T2D = T2D.reshape(chi, chi, D_bond, D_bond)
    T3A = T3A.reshape(chi, chi, D_bond, D_bond)
    T3F = T3F.reshape(chi, chi, D_bond, D_bond)
    T1C = T1C.reshape(chi, chi, D_bond, D_bond)

    open_C = oe.contract(
        "YX,MYar,abci,rstj->MbsXctij",
        C21AF,
        T1B,
        c,
        c.conj(),
        optimize=[(0, 1), (0, 1), (0, 1)],
        backend="torch",
    )
    open_F = oe.contract(
        "MYct,abci,rstj->YarMbsij",
        T3A,
        f,
        f.conj(),
        optimize=[(0, 1), (0, 1)],
        backend="torch",
    )
    open_E = oe.contract(
        "ZY,NZbs,abci,rstj->NctYarij",
        C32CB,
        T2D,
        e,
        e.conj(),
        optimize=[(0, 1), (0, 1), (0, 1)],
        backend="torch",
    )
    open_B = oe.contract(
        "NZar,abci,rstj->ZbsNctij",
        T1C,
        b,
        b.conj(),
        optimize=[(0, 1), (0, 1)],
        backend="torch",
    )
    open_A = oe.contract(
        "XZ,LXct,abci,rstj->LarZbsij",
        C13ED,
        T3F,
        a,
        a.conj(),
        optimize=[(0, 1), (0, 1), (0, 1)],
        backend="torch",
    )
    open_D = oe.contract(
        "LXbs,abci,rstj->XctLarij",
        T2E,
        d,
        d.conj(),
        optimize=[(0, 1), (0, 1)],
        backend="torch",
    )

    closed_C = oe.contract("MbsXctii->MbsXct", open_C, backend="torch").reshape(
        chi * D_bond * D_bond, chi * D_bond * D_bond
    )
    closed_F = oe.contract("YarMbsii->YarMbs", open_F, backend="torch").reshape(
        chi * D_bond * D_bond, chi * D_bond * D_bond
    )
    closed_E = oe.contract("NctYarii->NctYar", open_E, backend="torch").reshape(
        chi * D_bond * D_bond, chi * D_bond * D_bond
    )
    closed_B = oe.contract("ZbsNctii->ZbsNct", open_B, backend="torch").reshape(
        chi * D_bond * D_bond, chi * D_bond * D_bond
    )
    closed_A = oe.contract("LarZbsii->LarZbs", open_A, backend="torch").reshape(
        chi * D_bond * D_bond, chi * D_bond * D_bond
    )
    closed_D = oe.contract("XctLarii->XctLar", open_D, backend="torch").reshape(
        chi * D_bond * D_bond, chi * D_bond * D_bond
    )

    EF = torch.mm(closed_E, closed_F)
    AB = torch.mm(closed_A, closed_B)
    CD = torch.mm(closed_C, closed_D)

    return oe.contract("xy,yz,zx->", EF, CD, AB, backend="torch")


def run_test(trials=5, steps_per_trial=20):
    return run_test_configured(trials=trials, steps_per_trial=steps_per_trial)


def _fmt_re_im(z: torch.Tensor, *, re_fmt: str = "+.6e", im_fmt: str = "+.6e") -> str:
    """Format a scalar tensor as '(re, im)' with scientific notation."""
    # torch scalars -> python numbers
    zc = z.item()
    if isinstance(zc, complex):
        re = zc.real
        im = zc.imag
    else:
        re = float(zc)
        im = 0.0
    return f"({re:{re_fmt}}, {im:{im_fmt}})"


def run_test_configured(
    *,
    trials: int = 5,
    steps_per_trial: int = 20,
    print_every: int = 1,
    imag_tol: float = 1e-8,
    use_double: bool = False,
):
    """Run the normalization/imaginary-part diagnostic.

    Prints one consistent line every `print_every` steps.
    """
    if use_double:
        # Must be called before allocating core tensors (Hs, a..f, etc.).
        set_dtype(True)

    print("Running CTMRG normalization test using core_unrestricted.py code")

    D_bond = 3
    d_phys = 2
    chi = 17
    D_sq = D_bond ** 2

    Hs = [build_heisenberg_H(1.0, d_phys) for _ in range(9)]

    neg_counts = {"norm1": 0, "norm2": 0, "norm3": 0}
    imag_viol = {"E1": 0, "E2": 0, "E3": 0, "Etot": 0, "n1": 0, "n2": 0, "n3": 0}
    max_imag = {"E1": 0.0, "E2": 0.0, "E3": 0.0, "Etot": 0.0, "n1": 0.0, "n2": 0.0, "n3": 0.0}
    total = 0

    for trial in range(trials):
        if print_every > 0:
            print(
                f"\ntrial {trial+1}/{trials} (D={D_bond}, chi={chi}, steps={steps_per_trial}, "
                f"print_every={print_every}, imag_tol={imag_tol:.1e}, use_double={use_double})"
            )
            print(
                "step | Etot(re,im) | E1(re,im) E2(re,im) E3(re,im) | "
                "n1(re,im) n2(re,im) n3(re,im) | flags"
            )
        a, b, c, d, e, f = initialize_abcdef("random", D_bond, d_phys, 1e-3)
        for t in (a, b, c, d, e, f):
            t.requires_grad_(True)

        prev_total_val = None
        prev_norms = None

        for step in range(steps_per_trial):
            with torch.no_grad():
                a.data = normalize_tensor(a.data)
                b.data = normalize_tensor(b.data)
                c.data = normalize_tensor(c.data)
                d.data = normalize_tensor(d.data)
                e.data = normalize_tensor(e.data)
                f.data = normalize_tensor(f.data)
            
            with torch.no_grad():
                A, B, C, D, E, F = abcdef_to_ABCDEF(a, b, c, d, e, f, D_sq)
                all27 = CTMRG_from_init_to_stop(A, B, C, D, E, F, chi, D_sq, 70, 1e-9)

            E1 = energy_expectation_nearest_neighbor_3ebadcf_bonds(
                a, b, c, d, e, f, Hs[0], Hs[1], Hs[2], chi, D_bond, *all27[:9]
            )
            E2 = energy_expectation_nearest_neighbor_3afcbed_bonds(
                a, b, c, d, e, f, Hs[3], Hs[4], Hs[5], chi, D_bond, *all27[9:18]
            )
            E3 = energy_expectation_nearest_neighbor_other_3_bonds(
                a, b, c, d, e, f, Hs[6], Hs[7], Hs[8], chi, D_bond, *all27[18:27]
            )

            Etot = E1 + E2 + E3

            n1c = norm_env_1(a, b, c, d, e, f, chi, D_bond, *all27[:9])
            n2c = norm_env_2(a, b, c, d, e, f, chi, D_bond, *all27[9:18])
            n3c = norm_env_3(a, b, c, d, e, f, chi, D_bond, *all27[18:27])

            n1 = n1c.real.item()
            n2 = n2c.real.item()
            n3 = n3c.real.item()

            imag_vals = {
                "E1": abs(E1.imag.item()),
                "E2": abs(E2.imag.item()),
                "E3": abs(E3.imag.item()),
                "Etot": abs(Etot.imag.item()),
                "n1": abs(n1c.imag.item()),
                "n2": abs(n2c.imag.item()),
                "n3": abs(n3c.imag.item()),
            }
            for k, v in imag_vals.items():
                if v > imag_tol:
                    imag_viol[k] += 1
                if v > max_imag[k]:
                    max_imag[k] = v

            total += 1

            flags = []
            if n1 < 0:
                neg_counts["norm1"] += 1
                flags.append("neg_n1")
            if n2 < 0:
                neg_counts["norm2"] += 1
                flags.append("neg_n2")
            if n3 < 0:
                neg_counts["norm3"] += 1
                flags.append("neg_n3")

            total_val = Etot.real.item()
            if prev_total_val is not None:
                dE = abs(total_val - prev_total_val)
                dN = max(
                    abs(n1 - prev_norms[0]),
                    abs(n2 - prev_norms[1]),
                    abs(n3 - prev_norms[2]),
                )
                if dE > 1e-12 or dN > 1e-12:
                    flags.append(f"drift(dE={dE:.1e},dN={dN:.1e})")
            prev_total_val = total_val
            prev_norms = (n1, n2, n3)

            # Per-step consistent printout
            if print_every > 0 and ((step + 1) % print_every == 0):
                imag_flag = [k for k, v in imag_vals.items() if v > imag_tol]
                if imag_flag:
                    flags.append("imag>tol:" + ",".join(imag_flag))
                flag_str = "-" if not flags else ";".join(flags)
                print(
                    f"{step+1:4d} | "
                    f"Etot={_fmt_re_im(Etot, re_fmt='+.6e', im_fmt='+.6e')} | "
                    f"E1={_fmt_re_im(E1)} E2={_fmt_re_im(E2)} E3={_fmt_re_im(E3)} | "
                    f"n1={_fmt_re_im(n1c, re_fmt='+.6e', im_fmt='+.6e')} "
                    f"n2={_fmt_re_im(n2c, re_fmt='+.6e', im_fmt='+.6e')} "
                    f"n3={_fmt_re_im(n3c, re_fmt='+.6e', im_fmt='+.6e')} | "
                    f"{flag_str}"
                )

    print("\nSummary")
    print(f"Total evaluations: {total}")
    print(
        "negatives: "
        f"norm1={neg_counts['norm1']} "
        f"norm2={neg_counts['norm2']} "
        f"norm3={neg_counts['norm3']}"
    )
    print(f"imag_tol={imag_tol:.1e}")
    print(
        "imag violations: "
        f"E1={imag_viol['E1']} E2={imag_viol['E2']} E3={imag_viol['E3']} "
        f"Etot={imag_viol['Etot']} n1={imag_viol['n1']} n2={imag_viol['n2']} n3={imag_viol['n3']}"
    )
    print(
        "max |imag|: "
        f"E1={max_imag['E1']:.3e} E2={max_imag['E2']:.3e} E3={max_imag['E3']:.3e} "
        f"Etot={max_imag['Etot']:.3e} n1={max_imag['n1']:.3e} n2={max_imag['n2']:.3e} n3={max_imag['n3']:.3e}"
    )


if __name__ == "__main__":
    run_test_configured(trials=5, steps_per_trial=20, print_every=1, imag_tol=1e-8, use_double=False)
