#!/usr/bin/env python3
import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LEGACY_NEEL = Path(r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\D345678910")
NEEL_XI = ROOT / "models" / "0727core" / "neel_six_correlation_lengths" / "results"
SUMMARY = Path(r"D:\HyraiOn\ENS_Lyon\Internship\2026-EPFL\data\0713summary")
OUTPUT = ROOT / "visual_elements" / "csv_data_points"

NEEL_OUTPUT = OUTPUT / "0507coreD45678910_neel"
TWOC3_OUTPUT = OUTPUT / "0713summary_twoc3"
ORDINARY_NEEL_KEYS = (
    "env2_ordinary",
    "env1_ab_env3_ba_ordinary",
    "env3_ab_env1_ba_ordinary",
)
TWOC3_DIRECTIONS = ("env2", "env1_ab_env3_ba", "env3_ab_env1_ba")
NN_GROUPS = (
    ((1, "EB"), (1, "AD"), (1, "CF"), (3, "BE"), (3, "FC"), (3, "DA")),
    ((2, "CB"), (2, "AF"), (2, "ED"), (1, "FA"), (1, "DE"), (1, "BC")),
    ((3, "EF"), (3, "AB"), (3, "CD"), (2, "DC"), (2, "BA"), (2, "FE")),
)

FOLDER_RE = re.compile(r"^neel_symmetrized__J2_([0-9p]+)_\d{8}")
OBS_RE = re.compile(r"^D_(\d+)_chi_(\d+)_energy_magnetization_correlation\.txt$")
MAG_RE = re.compile(
    r"^mag_env(\d+)_([A-F])\s+Sx=([+-]?[\d.eE+-]+)\s+Sy=([+-]?[\d.eE+-]+)\s+Sz=([+-]?[\d.eE+-]+)",
    re.MULTILINE,
)
CORR_RE = re.compile(r"^corr_env(\d+)_([A-F]{2})\s*=\s*([+-]?[\d.eE+-]+)", re.MULTILINE)


def j2_tag(value):
    return f"{value:g}".replace(".", "p")


def spectrum_inverse_xi(spectrum):
    eigenvalues = spectrum["eigenvalues"][:2]
    magnitudes = sorted(
        (abs(complex(float(item["real"]), float(item["imag"]))) for item in eigenvalues),
        reverse=True,
    )
    return math.log(magnitudes[0] / magnitudes[1])


def central_xi(payload, keys):
    inverse_values = sorted(spectrum_inverse_xi(payload["spectra"][key]) for key in keys)
    return 1.0 / inverse_values[1]


def parse_m_neel(path):
    text = path.read_text(encoding="utf-8")
    mag = {
        (int(match.group(1)), match.group(2)): (float(match.group(3)), float(match.group(5)))
        for match in MAG_RE.finditer(text)
    }
    ace = [(env, site) for env in (1, 2, 3) for site in "ACE"]
    bdf = [(env, site) for env in (1, 2, 3) for site in "BDF"]
    sx_ace = sum(mag[key][0] for key in ace) / len(ace)
    sx_bdf = sum(mag[key][0] for key in bdf) / len(bdf)
    sz_ace = sum(mag[key][1] for key in ace) / len(ace)
    sz_bdf = sum(mag[key][1] for key in bdf) / len(bdf)
    return math.hypot(0.5 * (sx_ace - sx_bdf), 0.5 * (sz_ace - sz_bdf))


def parse_delta(path):
    text = path.read_text(encoding="utf-8")
    corr = {
        (int(match.group(1)), match.group(2)): float(match.group(3))
        for match in CORR_RE.finditer(text)
    }
    means = [sum(corr[key] for key in group) / len(group) for group in NN_GROUPS]
    return abs(max(means) - min(means))


def write_rows(directory, j2, header, rows):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / f"J2_{j2_tag(j2)}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def export_neel():
    xi_cases = {}
    for path in NEEL_XI.rglob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema") != "neel_six_correlation_lengths" or payload.get("schema_version") != 3:
                continue
            ctm = payload.get("ctm", {})
            if not (ctm.get("converged_ab_within_budget") and ctm.get("converged_ba_within_budget")):
                continue
            key = (round(float(payload["J2"]), 6), int(payload["D"]))
            rank = (int(bool(payload.get("checkpoint_sha256"))), path.stat().st_mtime_ns)
            if key not in xi_cases or rank > xi_cases[key][0]:
                xi_cases[key] = (rank, central_xi(payload, ORDINARY_NEEL_KEYS))
        except (OSError, KeyError, TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError):
            continue

    selected_folders = {}
    for path in sorted(LEGACY_NEEL.iterdir()):
        match = FOLDER_RE.match(path.name)
        if match and path.is_dir():
            j2 = round(float(match.group(1).replace("p", ".")), 6)
            selected_folders[j2] = path

    for j2, folder in selected_folders.items():
        if j2 in {0.0, 0.255}:
            continue
        observations = {}
        for path in folder.iterdir():
            match = OBS_RE.match(path.name)
            if not match:
                continue
            D, chi = int(match.group(1)), int(match.group(2))
            if D not in range(4, 11):
                continue
            if D not in observations or chi > observations[D][0]:
                observations[D] = (chi, path)
        rows = []
        for D, (_, path) in sorted(observations.items()):
            xi = xi_cases.get((j2, D))
            if xi is not None:
                rows.append((D, f"{xi[1]:.17g}", f"{parse_m_neel(path):.17g}"))
        if rows:
            write_rows(NEEL_OUTPUT, j2, ("D", "xi", "m_Neel"), rows)


def export_twoc3():
    for j2_dir in sorted(SUMMARY.glob("J2_*")):
        if not j2_dir.is_dir():
            continue
        j2 = round(float(j2_dir.name[3:].replace("p", ".")), 6)
        rows = []
        for d_dir in sorted((j2_dir / "2tensor_twoC3").glob("D_*"), key=lambda p: int(p.name[2:])):
            D = int(d_dir.name[2:])
            if D < 5:
                continue
            observable = d_dir / "energy_magnetization_correlation.txt"
            correlation = d_dir / "correlation_length.json"
            if not (observable.is_file() and correlation.is_file()):
                continue
            try:
                payload = json.loads(correlation.read_text(encoding="utf-8"))
                legacy = (
                    payload.get("schema") == "twoc3_three_ordinary_correlation_lengths"
                    and payload.get("schema_version") == 5
                    and payload.get("transfer_network_schema") == "three_geometric_straight_rows_ordinary_v5"
                )
                current = (
                    payload.get("schema") == "c3ctm_three_ordinary_correlation_lengths"
                    and payload.get("schema_version") == 6
                    and payload.get("transfer_network_schema") == "three_geometric_straight_rows_ordinary_v6"
                    and payload.get("ansatz_directory") == "2tensor_twoC3"
                )
                if not (legacy or current):
                    continue
                rows.append((D, f"{central_xi(payload, TWOC3_DIRECTIONS):.17g}", f"{parse_delta(observable):.17g}"))
            except (OSError, KeyError, TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError):
                continue
        if rows:
            write_rows(TWOC3_OUTPUT, j2, ("D", "xi", "Delta"), rows)


if __name__ == "__main__":
    export_neel()
    export_twoc3()
