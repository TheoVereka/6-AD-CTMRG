"""Coordinate-built honeycomb replica schematics; no entropy calculation.

Honeycomb bond length is one; sphere diameter .10, ket/bra center gap .19.
Scene JSON is retained for review.
Camera/rendering is provided by depth_renderer.py, not mplot3d painter sorting.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from depth_renderer import camera_frame

ROOT = Path(__file__).resolve().parent
SITE_DIAMETER = 0.10
HEX_BOND = 1.0
LAYER_GAP = 0.19
GRAY_STUB = 0.03
Z_UPPER = LAYER_GAP / 2
Z_LOWER = -Z_UPPER
BOND_RADIUS = 0.010
BLUE_RADIUS = 0.032
GRAY_RADIUS = 0.006
SQUARE_SIDE = 0.3
GHOST_ALPHA = 0.25
PROJECTION = "cabinet"
DEPTH_SCALE = 0.5
DEPTH_ANGLE = 60.
ROW_LABELS = ("cfabedcfabed", "bedcfabedcfa") * 2
PALETTE_RGB = {
    "a": (122, 191, 165), "b": (219, 140, 191),
    "c": (145, 157, 199), "d": (171, 215, 101),
    "e": (236, 145, 105), "f": (246, 219, 82),
}
PALETTE = {k: [c / 255 for c in v] for k, v in PALETTE_RGB.items()}
BLACK = [0.04, 0.04, 0.04]
BLUE = [110 / 255, 150 / 255, 255 / 255]
DARK_BLUE = [0.025, 0.12, 0.39]
RED = [0.87, 0.08, 0.12]
GREEN = [0.05, 0.59, 0.25]
GRAY = [160 / 255, 160 / 255, 160 / 255]


def point(row, col, z=0.0):
    parity = col % 2
    ys = (2 + parity / 2, 1 - parity / 2,
          -1 + parity / 2, -2 - parity / 2)
    # The whole world-x convention is reversed together with every x shift.
    # +x projects left, but increasing column still moves VISUALLY right.
    return [-(col - 5.5) * math.sqrt(3) / 2 * HEX_BOND, ys[row] * HEX_BOND, float(z)]


def rod(uid, p0, p1, color, **metadata):
    return dict(id=uid, kind="cylinder", p0=list(p0), p1=list(p1),
                radius=BOND_RADIUS, color=list(color), alpha=1.0, **metadata)


def translate(elements, offset, prefix=None, **metadata):
    out = copy.deepcopy(elements)
    for e in out:
        for key in ("center", "p0", "p1", "cut_tip"):
            if key in e:
                e[key] = [float(v + d) for v, d in zip(e[key], offset)]
        if prefix:
            e["id"] = prefix + "/" + e["id"]
        e.update(metadata)
    return out


def step1():
    elements = []
    for row in range(4):
        for col in range(12):
            letter = ROW_LABELS[row][col]
            elements.append(dict(id=f"site/{row}/{col}", kind="sphere",
                center=point(row, col), radius=SITE_DIAMETER / 2,
                round_symbol=True,
                color=PALETTE[letter], alpha=1., role="site", row=row,
                col=col, letter=letter, side="left" if col < 6 else "right"))
        for col in range(11):
            elements.append(rod(f"zigzag/{row}/{col}", point(row, col),
                point(row, col + 1), BLACK, role="zigzag", row=row, col=col,
                side="cut" if col == 5 else "left" if col < 5 else "right"))
    for row in range(3):
        for col in range(12):
            p, q = point(row, col), point(row + 1, col)
            if abs(abs(p[1] - q[1]) - HEX_BOND) < 1e-12:
                elements.append(rod(f"vertical/{row}/{col}", p, q, BLUE,
                    role="vertical", row=row, col=col,
                    side="left" if col < 6 else "right", dashed=True, line_style="dotted"))
    return elements


def step2(base):
    elements = []
    for layer, z, color in (("upper", Z_UPPER, RED), ("lower", Z_LOWER, GREEN)):
        for e in translate(base, (0, 0, z), layer, layer=layer):
            if e.get("side") != "cut":
                elements.append(e)
                continue
            mid = ((np.array(e["p0"]) + np.array(e["p1"])) / 2).tolist()
            for side, p, q in (("left", e["p0"], mid), ("right", mid, e["p1"])):
                half = copy.deepcopy(e)
                half.update(id=e["id"] + "/" + side, p0=p, p1=q, color=color,
                            side=side, role="cut_half", cut_tip=mid)
                elements.append(half)
    return elements


def gray_connectors(side, mode, prefix=""):
    elements = []
    for row in range(4):
        columns = range(6) if side == "left" else range(6, 12)
        for col in columns:
            if mode == "full":
                connector = rod(f"{prefix}gray/full/{row}/{col}",
                    point(row, col, Z_LOWER), point(row, col, Z_UPPER), GRAY,
                    role="physical_trace", side=side, row=row, col=col,
                    layer="pair")
                connector["radius"] = GRAY_RADIUS
                elements.append(connector)
            else:
                for layer, z, sign in (("upper", Z_UPPER, -1), ("lower", Z_LOWER, 1)):
                    p = point(row, col, z + sign * SITE_DIAMETER / 2)
                    q = point(row, col, p[2] + sign * GRAY_STUB)
                    connector = rod(f"{prefix}gray/{layer}/{row}/{col}", p, q, GRAY,
                        role="physical_open", side=side, row=row, col=col, layer=layer)
                    connector["radius"] = GRAY_RADIUS
                    elements.append(connector)
    return elements


def step3(base):
    return copy.deepcopy(base) + gray_connectors("left", "full") + gray_connectors("right", "open")


def bounds(elements):
    points = []
    for e in elements:
        for k in ("center", "p0", "p1"):
            if k in e:
                points.append(e[k])
    arr = np.array(points)
    return arr.min(axis=0), arr.max(axis=0)


def ellipsis(uid, center, direction):
    center, direction = np.array(center), np.array(direction)
    direction = direction / np.linalg.norm(direction)
    right, up, _, _, _ = camera_frame(PROJECTION, depth_scale=DEPTH_SCALE, depth_angle=DEPTH_ANGLE)
    screen_length = np.linalg.norm([right @ direction, up @ direction])
    world_spacing = .14 * HEX_BOND / screen_length
    return [dict(id=f"{uid}/{i}", kind="sphere", center=(center + (i - 1) * world_spacing * direction).tolist(),
                 radius=.033 * HEX_BOND, color=BLACK, alpha=1., round_symbol=True,
                 shade=False, role="ellipsis", ellipsis_id=uid)
            for i in range(3)]


def step3_bis(base):
    out = copy.deepcopy(base)
    for side, sign in (("left", 1), ("right", -1)):
        part = [e for e in base if e.get("side") == side]
        lo, hi = bounds(part)
        center = (lo + hi) / 2
        for axis, direction in ((0, sign), (1, -1), (1, 1)):
            d = np.zeros(3)
            d[axis] = direction
            # Measure the visible outline along the projected conceptual ray.
            # Using just center-coordinate bounds would hide a dot behind a
            # ket sphere at this view angle. The ray itself stays in x/y.
            distance = projected_ray_exit(part, center, d) + .5 * HEX_BOND
            p = center + distance * d
            out.extend(ellipsis(f"ellipsis/{side}/{axis}/{direction}", p, d))
    return out


def step2_bis(base):
    """The same six continuation symbols, on the untraced step-2 layers."""
    return step3_bis(base)


def projected_ray_exit(elements, center, direction):
    from scipy.spatial import ConvexHull
    right, up, _, _, _ = camera_frame(PROJECTION, depth_scale=DEPTH_SCALE, depth_angle=DEPTH_ANGLE)
    basis = np.array([right, up])
    samples = []
    for e in elements:
        radius = e.get("radius", BOND_RADIUS)
        for key in ("center", "p0", "p1"):
            if key in e:
                q = basis @ np.array(e[key])
                # Conservative circle enclosing the projected ball/rod cap.
                projected_radius = radius * np.linalg.norm(basis, ord=2)
                for angle in np.linspace(0, 2 * math.pi, 24, endpoint=False):
                    samples.append(q + projected_radius * np.array([math.cos(angle), math.sin(angle)]))
    samples = np.array(samples)
    vertices = samples[ConvexHull(samples).vertices]
    c, d = basis @ center, basis @ direction
    intersections = []
    for i, a in enumerate(vertices):
        b = vertices[(i + 1) % len(vertices)]
        matrix = np.column_stack([d, a - b])
        if abs(np.linalg.det(matrix)) < 1e-12:
            continue
        t, u = np.linalg.solve(matrix, a - c)
        if t >= 0 and -1e-10 <= u <= 1 + 1e-10:
            intersections.append(t)
    return max(intersections)


def step4(base, bottom_shift):
    left = [e for e in base if e.get("side") == "left"]
    right = [e for e in base if e.get("side") == "right"]
    right_no_stubs = [e for e in right if e.get("role") != "physical_open"]
    right_pair = right_no_stubs + gray_connectors("right", "full", "replacement/")
    out = translate(left, (0, 0, 0), "left_original", group="left_original")
    out += translate(left, (0, 0, -4 * HEX_BOND), "left_copy", group="left_copy")
    out += translate(right_pair, (-2 * HEX_BOND, 0, -2 * HEX_BOND), "right_middle", group="right_middle")
    out += translate([e for e in right if e.get("layer") == "upper"],
                     (-2 * HEX_BOND, 0, 2 * HEX_BOND), "right_top", group="right_top")
    out += translate([e for e in right if e.get("layer") == "lower"],
                     (-2 * HEX_BOND, 0, -bottom_shift * HEX_BOND), "right_bottom", group="right_bottom")
    return out


def tip(element):
    return element["p1"] if element["side"] == "left" else element["p0"]


def step5(base):
    out = copy.deepcopy(base)
    matches = (("left_original", "right_top", "upper", RED),
               ("left_copy", "right_bottom", "lower", GREEN),
               ("left_copy", "right_middle", "upper", RED),
               ("left_original", "right_middle", "lower", GREEN))
    for left_group, right_group, layer, color in matches:
        for row in range(4):
            def get(group):
                return next(e for e in base if e.get("role") == "cut_half"
                            and e.get("group") == group and e["layer"] == layer
                            and e["row"] == row)
            p, q = tip(get(left_group)), tip(get(right_group))
            out.append(rod(f"flying/{left_group}/{right_group}/{layer}/{row}", p, q, color,
                role="flying", row=row, layer=layer, source_group=left_group,
                destination_group=right_group))
    return out


def step5_bis(base):
    """Fifteen continuation symbols for the five physical lattice regions.

    Flying replica connections are retained in the picture but excluded when
    locating each region's center and extent.
    """
    out = copy.deepcopy(base)
    for group in ("left_original", "left_copy", "right_middle", "right_top", "right_bottom"):
        part = [e for e in base if e.get("group") == group and e.get("role") != "flying"]
        lo, hi = bounds(part)
        center = (lo + hi) / 2
        outward_x = 1 if group.startswith("left") else -1
        for axis, sign in ((0, outward_x), (1, -1), (1, 1)):
            direction = np.zeros(3)
            direction[axis] = sign
            distance = projected_ray_exit(part, center, direction) + .5 * HEX_BOND
            out += ellipsis(f"ellipsis/region/{group}/{axis}/{sign}", center + distance * direction, direction)
    return out


def boundary_group(side, center_z, shift_x=0., name="boundary", half=None):
    """Four CTM edge symbols and five segments of the auxiliary chi chain.

    Symbol centers coincide with the former ket/bra pair centers. Its attached
    Thin attached legs are hidden by owner associations. Auxiliary chi tubes
    and square faces use true three-dimensional ray depth against each other.
    """
    out = []
    col = 5 if side == "left" else 6
    centers = []
    for row in range(4):
        center = point(row, col, center_z)
        center[0] += shift_x
        trace_x = center[0]
        centers.append(center)
        out.append(dict(id=f"{name}/edge/{row}", kind="square", center=center,
            side=SQUARE_SIDE, color=PALETTE[ROW_LABELS[row][col]], alpha=1.,
            frame_color=BLACK, frame_radius=BOND_RADIUS * .85,
            role="edge_symbol", boundary_side=side, row=row,
            letter=ROW_LABELS[row][col], group=name, half=half,
            physical_trace_x=trace_x, symbol_id=f"{name}/edge/{row}"))
    short, long = HEX_BOND, 2 * HEX_BOND
    extension = short / 2 if side == "left" else long / 2
    spans = [(centers[i], centers[i + 1], [i, i + 1]) for i in range(3)]
    top, bottom = centers[0].copy(), centers[-1].copy()
    top[1] += extension
    bottom[1] -= extension
    spans += [(top, centers[0], [0]), (centers[-1], bottom, [3])]
    for i, (p, q, rows) in enumerate(spans):
        e = rod(f"{name}/chi/{i}", p, q, BLUE, role="chi_chain",
                group=name, boundary_side=side, half=half,
                depth_policy="true_ray_depth_against_tensor_symbols")
        e.update(radius=BLUE_RADIUS, outline_color=DARK_BLUE, outline_width=.003)
        out.append(e)
    return out


def step6(base):
    out = []
    for e in base:
        if e.get("role") == "cut_half":
            half = copy.deepcopy(e)
            half["symbol_owners"] = [f"{e['side']}_boundary/edge/{e['row']}"]
            out.append(half)
        elif e.get("side") == "left" and not (
            e.get("col") == 5 and e.get("role") in ("site", "physical_trace")):
            ghost = copy.deepcopy(e)
            ghost["alpha"] = GHOST_ALPHA
            ghost["context_background"] = True
            if e.get("role") == "zigzag" and e.get("col") == 4:
                ghost["symbol_owners"] = [f"left_boundary/edge/{e['row']}"]
            elif e.get("role") == "vertical" and e.get("col") == 5:
                ghost["symbol_owners"] = [f"left_boundary/edge/{r}" for r in (e["row"], e["row"] + 1)]
            out.append(ghost)
    out += boundary_group("left", 0., name="left_boundary")
    out += boundary_group("right", 0., name="right_boundary")
    # Two ellipses total, on a shared centerline between the two equal spans.
    for sign in (-1, 1):
        out += ellipsis(f"ellipsis/chi/{sign}", [0., sign * 3.5 * HEX_BOND, 0.], [0, 1, 0])
    return out


def step6_bis(base):
    """Preserve the two y continuations and extend the faded lattice to left."""
    out = copy.deepcopy(base)
    ghost = [e for e in base if e.get("context_background")]
    lo, hi = bounds(ghost)
    center = (lo + hi) / 2
    direction = np.array([1., 0., 0.])
    distance = projected_ray_exit(ghost, center, direction) + .5 * HEX_BOND
    extra = ellipsis("ellipsis/context_left", center + distance * direction, direction)
    for dot in extra:
        dot.update(alpha=GHOST_ALPHA, context_background=True)
    return out + extra


def step7(base, bottom_shift):
    out = [copy.deepcopy(e) for e in base if e.get("role") in ("cut_half", "flying")]
    for e in out:
        if e["role"] == "cut_half":
            e["symbol_owners"] = [f"{e['group']}/edge/{e['row']}"]
        else:
            e["symbol_owners"] = [f"{g}/edge/{e['row']}" for g in (e["source_group"], e["destination_group"])]
    out += boundary_group("left", 0., name="left_original")
    out += boundary_group("left", -4 * HEX_BOND, name="left_copy")
    out += boundary_group("right", -2 * HEX_BOND, -2 * HEX_BOND, name="right_middle")
    out += boundary_group("right", 2 * HEX_BOND, -2 * HEX_BOND, name="right_top", half="upper")
    out += boundary_group("right", -bottom_shift * HEX_BOND, -2 * HEX_BOND, name="right_bottom", half="lower")
    return out


def step7_bis(base):
    """Two y continuations for each of the five full or half chi chains."""
    out = copy.deepcopy(base)
    for group in ("left_original", "left_copy", "right_middle", "right_top", "right_bottom"):
        chain = [e for e in base if e.get("role") == "chi_chain" and e["group"] == group]
        lo, hi = bounds(chain)
        center = (lo + hi) / 2
        for sign in (-1, 1):
            direction = np.array([0., float(sign), 0.])
            distance = projected_ray_exit(chain, center, direction) + .5 * HEX_BOND
            out += ellipsis(f"ellipsis/replica_chain/{group}/{sign}", center + distance * direction, direction)
    return out


def expand_dashes(elements):
    """Expand legacy styled-bond records into equally spaced round dots."""
    out = []
    for e in elements:
        if not e.get("dashed"):
            out.append(e)
            continue
        p, q = np.array(e["p0"]), np.array(e["p1"])
        length = np.linalg.norm(q - p)
        intervals = max(1, round(length / (.10 * HEX_BOND)))
        for i, fraction in enumerate(np.linspace(0., 1., intervals + 1)):
            dot = copy.deepcopy(e)
            dot.pop("p0")
            dot.pop("p1")
            dot.update(id=e["id"] + f"/dot/{i}", kind="sphere", dotted_mark=True,
                       center=(p + fraction * (q - p)).tolist(),
                       radius=BOND_RADIUS, round_symbol=True, shade=False)
            out.append(dot)
    return out


def validate(scenes, bottom_shift):
    report = {}
    s1 = scenes["01"]
    sites = [e for e in s1 if e["role"] == "site"]
    edges = [e for e in s1 if e["kind"] == "cylinder"]
    assert len(sites) == 48 and len(edges) == 62
    degree = Counter()
    for e in edges:
        assert abs(np.linalg.norm(np.array(e["p1"]) - e["p0"]) - HEX_BOND) < 1e-12
        for p in (e["p0"], e["p1"]):
            node = next(s for s in sites if np.linalg.norm(np.array(s["center"]) - p) < 1e-12)
            degree[(node["row"], node["col"])] += 1
    assert degree[(0, 0)] == 2
    report["boundary_degree_counts"] = dict(Counter(degree.values()))
    report["degree_one_nodes_1_based"] = [[r + 1, c + 1] for (r, c), d in degree.items() if d == 1]
    for name, scene in scenes.items():
        ids = [e["id"] for e in scene]
        assert len(ids) == len(set(ids)), f"Duplicate IDs in {name}"
        report[name] = {"elements": len(scene), "by_kind": dict(Counter(e["kind"] for e in scene)),
                        "by_role": dict(Counter(e.get("role") for e in scene))}
        for e in scene:
            if e.get("role") == "chi_chain":
                assert not e.get("symbol_owners"), "Chi tubes must use actual ray depth against squares"
    for branch, count in (("02_bis", 6), ("03_bis", 6), ("05_bis", 15), ("06_bis", 3), ("07_bis", 10)):
        if branch in scenes:
            assert len({e["ellipsis_id"] for e in scenes[branch] if e.get("role") == "ellipsis"}) == count
    cuts = [e for e in scenes["02"] if e.get("role") == "cut_half"]
    assert len(cuts) == 16
    assert all(abs(np.linalg.norm(np.array(e["p1"]) - e["p0"]) - .5 * HEX_BOND) < 1e-12 for e in cuts)
    assert sum(e.get("role") == "physical_trace" for e in scenes["03"]) == 24
    assert sum(e.get("role") == "physical_open" for e in scenes["03"]) == 48
    assert sum(e.get("role") == "physical_trace" for e in scenes["03_bis"]) == 24
    assert sum(e.get("role") == "physical_open" for e in scenes["03_bis"]) == 48
    s6 = scenes["06"]
    assert sum(e.get("role") == "edge_symbol" for e in s6) == 8
    assert sum(e.get("role") == "site" for e in s6) == 40
    assert sum(e.get("role") == "physical_trace" for e in s6) == 20
    assert all(e.get("context_background") and np.isclose(e["alpha"], GHOST_ALPHA)
               for e in s6 if e.get("role") in ("site", "physical_trace"))
    for square in (e for scene in scenes.values() for e in scene if e.get("role") == "edge_symbol"):
        assert np.isclose(square["center"][0], square["physical_trace_x"]), "Square center must lie on the former trace"
    for side, spacings in (("left", [2, 1, 2]), ("right", [1, 2, 1])):
        symbols = [e for e in s6 if e.get("role") == "edge_symbol" and e["boundary_side"] == side]
        assert np.allclose(-np.diff([e["center"][1] for e in symbols]), np.array(spacings) * HEX_BOND)
        chains = [e for e in s6 if e.get("role") == "chi_chain" and e["boundary_side"] == side]
        assert math.isclose(sum(np.linalg.norm(np.array(e["p1"]) - e["p0"]) for e in chains), 6 * HEX_BOND)
    if "05" in scenes:
        flies = [e for e in scenes["05"] if e.get("role") == "flying"]
        assert len(flies) == 16
        for key in {(e["source_group"], e["destination_group"], e["layer"]) for e in flies}:
            bundle = [e for e in flies if (e["source_group"], e["destination_group"], e["layer"]) == key]
            vectors = [np.array(e["p1"]) - e["p0"] for e in bundle]
            assert all(np.allclose(v, vectors[0]) for v in vectors)
        s7 = scenes["07"]
        assert sum(e.get("role") == "edge_symbol" and e.get("half") is None for e in s7) == 12
        assert sum(e.get("role") == "edge_symbol" and e.get("half") is not None for e in s7) == 8
        assert sum(e.get("role") == "chi_chain" and e.get("half") is not None for e in s7) == 10
        if bottom_shift == 6:
            moved_sites = [e for e in scenes["04"] if e.get("role") == "site"]
            positions = [tuple(np.round(e["center"], 12)) for e in moved_sites]
            assert len(positions) == len(set(positions)), "The separated layers must not overlap"
            bottom_z = [e["center"][2] for e in moved_sites if e["group"] == "right_bottom"]
            assert np.allclose(bottom_z, -6 * HEX_BOND - LAYER_GAP / 2)
    right, up, ray_right, ray_up, eye = camera_frame(PROJECTION, depth_scale=DEPTH_SCALE, depth_angle=DEPTH_ANGLE)
    projection_matrix = np.array([right, up])
    assert np.allclose(projection_matrix @ [1., 0., 0.], [-1., 0.])
    assert np.allclose(projection_matrix @ [0., 0., 1.], [0., 1.])
    assert np.all(projection_matrix @ [0., 1., 0.] > 0)
    assert np.allclose(projection_matrix @ np.column_stack([ray_right, ray_up]), np.eye(2))
    assert np.allclose(projection_matrix @ eye, 0)
    assert eye[2] > 0 and eye[1] < 0, "Camera must view from above/front"
    first_black = projection_matrix @ (np.array(point(0, 1)) - point(0, 0))
    assert np.all(first_black > 0), "The first black bond must go visually right/up"
    assert np.isclose(LAYER_GAP, .19)
    assert np.isclose(HEX_BOND, 1.) and np.isclose(SITE_DIAMETER, .10)
    assert np.isclose(GRAY_STUB, .03)
    assert np.isclose(LAYER_GAP - SITE_DIAMETER - 2 * GRAY_STUB, GRAY_STUB)
    if "05" in scenes:
        bundle_gaps = []
        for key in {(e["source_group"], e["destination_group"], e["layer"]) for e in flies}:
            bundle = [e for e in flies if (e["source_group"], e["destination_group"], e["layer"]) == key]
            direction = projection_matrix @ (np.array(bundle[0]["p1"]) - bundle[0]["p0"])
            normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
            offsets = sorted(normal @ projection_matrix @ np.array(e["p0"]) for e in bundle)
            minimum_gap = float(np.min(np.diff(offsets)))
            assert minimum_gap > 2 * BOND_RADIUS * np.linalg.norm(projection_matrix, ord=2), "Flying lines collapse in projection"
            bundle_gaps.append(dict(source=key[0], destination=key[1], layer=key[2], min_screen_gap=minimum_gap))
        report["flying_bundle_projection_gaps"] = bundle_gaps
    report["projection_checks"] = dict(x_points_left=True, z_points_up=True,
        y_points_up_right=True, xz_faces_are_true_squares=True,
        logical_left_right_preserved=True, camera_is_above_front=True,
        screen_projection_matrix=projection_matrix.tolist())
    report["parameters"] = dict(site_diameter=SITE_DIAMETER, layer_gap=LAYER_GAP,
        hex_bond=HEX_BOND, gray_stub=GRAY_STUB, square_side=SQUARE_SIDE,
        bottom_shift=bottom_shift, edge_join_length=0., gray_radius=GRAY_RADIUS,
        bulk_blue_rgb=[110, 150, 255], gray_rgb=[160, 160, 160],
        vertical_line_style="dotted_round_marks", ghost_opacity=GHOST_ALPHA,
        ghost_gray_trace_count=20, square_anchor="center_on_original_ket_bra_pair_center",
        square_precedence="exact_symbol_silhouette_masks_only_attached_objects",
        chi_square_occlusion="true_ray_depth_without_symbol_owner_mask",
        palette_rgb=PALETTE_RGB, camera=dict(projection=PROJECTION, depth_scale=DEPTH_SCALE, depth_angle=DEPTH_ANGLE))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", nargs="+", default=["01", "02", "02_bis", "03", "03_bis", "04", "05", "05_bis", "06", "06_bis", "07", "07_bis"])
    parser.add_argument("--bottom-shift", type=float, choices=[2., 6.], default=6.,
                        help="6 is the separated placement in the current 4/2/6 layout.")
    parser.add_argument("--width", type=int, default=1900)
    parser.add_argument("--supersample", type=int, default=2)
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args()
    base = step1()
    double = step2(base)
    connected = step3(double)
    scenes = {"01": base, "02": double, "02_bis": step2_bis(double),
              "03": connected, "03_bis": step3_bis(connected)}
    if args.bottom_shift is not None:
        moved = step4(connected, args.bottom_shift)
        flown = step5(moved)
        scenes.update({"04": moved, "05": flown, "05_bis": step5_bis(flown)})
    scenes["06"] = step6(connected)
    scenes["06_bis"] = step6_bis(scenes["06"])
    if args.bottom_shift is not None:
        scenes["07"] = step7(flown, args.bottom_shift)
        scenes["07_bis"] = step7_bis(scenes["07"])
    if any(name not in scenes for name in args.steps):
        parser.error("Steps 04/05/07 need the explicit --bottom-shift decision.")
    report = validate(scenes, args.bottom_shift)
    (ROOT / "geometry_checks.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    figures, coordinates = ROOT / "figures", ROOT / "coordinates"
    figures.mkdir(exist_ok=True)
    coordinates.mkdir(exist_ok=True)
    for name, scene in scenes.items():
        (coordinates / f"step_{name}.json").write_text(json.dumps(scene, indent=2), encoding="utf-8")
    if not args.json_only:
        from depth_renderer import render_scene
        for name in args.steps:
            meta = render_scene(expand_dashes(scenes[name]), figures / f"step_{name}.png",
                                width=args.width, supersample=args.supersample, projection=PROJECTION,
                                depth_scale=DEPTH_SCALE, depth_angle=DEPTH_ANGLE)
            (coordinates / f"camera_{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(f"Rendered {name}: {figures / ('step_' + name + '.png')}", flush=True)
    print("Coordinate, count, length, parallelism and chain-span checks passed.", flush=True)


if __name__ == "__main__":
    main()
