"""Parallel schematic rendering with per-pixel analytic depth tests.

The camera agrees with Matplotlib's default view_init(elev=30, azim=-60),
using equal world-unit scaling in x/y/z and orthographic projection. The
optional cabinet projection keeps x-z faces square, projects +x to the left,
+z upward, and +y diagonally up/right with configurable depth shortening. This
module draws no axes, labels, text, or background decorations. It uses NumPy
and Pillow only; it does not depend on Matplotlib's 3D painter ordering.

Public API::

    meta = render_scene(elements, "figure.png", width=1800, height=None,
                        supersample=2, elev=30, azim=-60)

Supported elements:
  sphere: center, radius, color, alpha=1, round_symbol=False
  cylinder: p0, p1, radius, color, alpha=1,
            outline_color=None, outline_width=0, half=None|upper|lower
  square: center, side, color, alpha=1, frame_color=None,
          frame_radius=0.01, half=None|upper|lower

Colors are RGB floats in [0,1]; Matplotlib color strings also work. Squares
lie in the x-z plane (normal y). Half objects are clipped at their original
midpoint z. A half square keeps its original outer frame edges and NEVER
receives a new waist frame. Clipped cylinders preserve their curved sides
and original end caps; a plain same-color cut face closes the solid without
an added waist outline. Optional cylinder outlines mark silhouettes and
original cap rims, not clipping boundaries.

All opaque hits share one z buffer. Transparent object fragments in front of
the nearest opaque hit are sorted individually per pixel and alpha composited.
A transparent solid contributes its nearest surviving surface once, so alpha
means object opacity rather than two independent front/back surface coatings.
There are no cast shadows, reflections, refraction, or volumetric absorption.

An optional ``round_symbol`` sphere is a schematic glyph with an exactly
circular screen footprint even under cabinet projection. Its surface is
defined in the camera's ray-coordinate frame, so it remains a volumetric
symbol with a well-defined nearest-hit depth shared with all other objects.
It is not the oblique image of a physical world-coordinate sphere. Ordinary
spheres retain their original geometry when this option is absent.

An element with ``context_background=True`` belongs to a separate schematic
underlay: it is composited before all foreground geometry, irrespective of
its world depth. Objects within either layer still use analytic ray depth,
and transparent fragments are sorted per pixel. This explicit convention
keeps the faded, already-contracted lattice behind the replacement tensors;
it does not pretend that the underlay is foreground physical geometry.

Square tensor symbols may have a ``symbol_id``. A connected object names its
owners in ``symbol_owners``; its hits inside those symbols' exact projected
silhouettes are suppressed, including the frame. This is an explicit diagram
convention making a tensor hide its attached legs. Unrelated foreground
geometry keeps true ray-depth ordering and can pass in front of a symbol.
The auxiliary chi tubes deliberately have no symbol owner mask: tubes and
square faces occlude each other according to their actual analytic ray hits.

Run this file with Python -B for a smoke image and numerical occlusion tests.
All smoke artifacts are written beside this module.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

sys.dont_write_bytecode = True

import numpy as np
from PIL import Image


def _vector(value: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (3,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain three finite coordinates")
    return result


def _color(value: Any) -> np.ndarray:
    if isinstance(value, str):
        from matplotlib.colors import to_rgb
        value = to_rgb(value)
    result = _vector(value, "color")
    if np.any(result < 0) or np.any(result > 1):
        raise ValueError("RGB color channels must be in [0,1]")
    return result


def _positive(value: Any, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return value


def camera_basis(elev: float = 30, azim: float = -60) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return screen-right, screen-up, and toward-eye unit vectors."""
    elevation, azimuth = math.radians(float(elev)), math.radians(float(azim))
    if not math.isfinite(elevation) or not math.isfinite(azimuth):
        raise ValueError("Camera angles must be finite")
    eye = np.array([math.cos(elevation) * math.cos(azimuth),
                    math.cos(elevation) * math.sin(azimuth), math.sin(elevation)])
    right = np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])
    up = np.cross(eye, right)
    return right, up, eye


def camera_frame(projection: str = "orthographic", elev: float = 30, azim: float = -60,
                 depth_scale: float = 0.5, depth_angle: float = 45):
    """Projection covectors, ray-origin vectors, and unit toward-eye vector.

    A cabinet ray is not perpendicular to the image plane. Separating the
    projection from its right-inverse avoids treating a sheared image as an
    ordinary rotated camera, which would give incorrect occlusion.
    """
    if projection == "orthographic":
        right, up, eye = camera_basis(elev, azim)
        return right, up, right, up, eye
    if projection != "cabinet":
        raise ValueError("projection must be orthographic or cabinet")
    depth_scale = _positive(depth_scale, "depth_scale")
    angle = math.radians(float(depth_angle))
    if not math.isfinite(angle):
        raise ValueError("depth_angle must be finite")
    dx, dy = depth_scale * math.cos(angle), depth_scale * math.sin(angle)
    right, up = np.array([-1., dx, 0.]), np.array([0., dy, 1.])
    ray_right, ray_up = np.array([-1., 0., 0.]), np.array([0., 0., 1.])
    # Select the ray branch ABOVE the lattice, looking from negative y.
    # The other kernel direction sees the lattice from underneath and reverses
    # front/back even though the screen coordinates are identical.
    eye = np.array([-dx, -1., dy])
    eye /= np.linalg.norm(eye)
    return right, up, ray_right, ray_up, eye


def _expand_elements(elements: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    primitives: list[dict[str, Any]] = []
    for element in elements:
        kind = element.get("kind")
        if kind not in ("sphere", "cylinder", "square"):
            raise ValueError(f"Unknown element kind: {kind!r}")
        alpha = float(element.get("alpha", 1.0))
        if not math.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError("alpha must be finite and in [0,1]")
        if alpha == 0:
            continue
        primitive = {"kind": kind, "color": _color(element.get("color", [0.2, 0.4, 0.8])),
                     "alpha": alpha, "shade": bool(element.get("shade", kind != "square")),
                     "context_background": bool(element.get("context_background", False)),
                     "symbol_owners": tuple(element.get("symbol_owners", ())),
                     "role": element.get("role")}
        half = element.get("half")
        if half not in (None, "upper", "lower"):
            raise ValueError("half must be None, upper, or lower")
        primitive["clip_mode"] = 0 if half is None else (1 if half == "upper" else -1)
        if kind == "sphere":
            primitive.update(center=_vector(element["center"], "center"),
                             radius=_positive(element["radius"], "radius"),
                             round_symbol=bool(element.get("round_symbol", False)))
            if half is not None:
                raise ValueError("Half clipping is supported for cylinders and squares, not spheres")
        elif kind == "cylinder":
            p0, p1 = _vector(element["p0"], "p0"), _vector(element["p1"], "p1")
            axis = p1 - p0
            length = _positive(np.linalg.norm(axis), "cylinder length")
            outline_width = float(element.get("outline_width", 0.0))
            if not math.isfinite(outline_width) or outline_width < 0:
                raise ValueError("outline_width must be finite and nonnegative")
            primitive.update(p0=p0, p1=p1, axis=axis / length, length=length,
                             radius=_positive(element["radius"], "radius"),
                             clip_z=0.5 * (p0[2] + p1[2]), outline_width=outline_width,
                             outline_color=_color(element["outline_color"]) if element.get("outline_color") is not None else None)
        else:
            center = _vector(element["center"], "center")
            side = _positive(element["side"], "side")
            primitive.update(center=center, side=side, clip_z=center[2])
            primitive["symbol_id"] = element.get("symbol_id", element.get("id"))
            if element.get("frame_color") is not None:
                radius = _positive(element.get("frame_radius", 0.01), "frame_radius")
                frame_color = _color(element["frame_color"])
                x0, x1 = center[0] - side / 2, center[0] + side / 2
                z0, z1 = center[2] - side / 2, center[2] + side / 2
                y = center[1]
                edges = [([x0, y, z0], [x0, y, z1]), ([x1, y, z0], [x1, y, z1])]
                if half != "upper":
                    edges.append(([x0, y, z0], [x1, y, z0]))
                if half != "lower":
                    edges.append(([x0, y, z1], [x1, y, z1]))
                for start, end in edges:
                    start, end = np.asarray(start), np.asarray(end)
                    axis = end - start
                    primitives.append({
                        "kind": "cylinder", "p0": start, "p1": end,
                        "axis": axis / np.linalg.norm(axis), "length": float(np.linalg.norm(axis)),
                        "radius": radius, "color": frame_color, "alpha": alpha,
                        "clip_mode": primitive["clip_mode"], "clip_z": center[2],
                        "outline_width": 0.0, "outline_color": None, "shade": False,
                        "context_background": primitive["context_background"],
                        "symbol_owners": primitive["symbol_owners"],
                        "symbol_id": primitive["symbol_id"],
                    })
        primitives.append(primitive)
    if not primitives:
        raise ValueError("At least one element with alpha > 0 is required")
    return primitives


def _bounds(primitive: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    kind = primitive["kind"]
    if kind == "sphere":
        extent = primitive["radius"]
        if primitive.get("round_symbol") and "_symbol_frame" in primitive:
            # The glyph is an affine image of a unit sphere in ray coordinates.
            extent = extent * np.linalg.norm(primitive["_symbol_frame"], axis=1)
        return primitive["center"] - extent, primitive["center"] + extent
    if kind == "cylinder":
        minimum = np.minimum(primitive["p0"], primitive["p1"]) - primitive["radius"]
        maximum = np.maximum(primitive["p0"], primitive["p1"]) + primitive["radius"]
    else:
        center, half_side = primitive["center"], primitive["side"] / 2
        minimum = center - np.array([half_side, 0.0, half_side])
        maximum = center + np.array([half_side, 0.0, half_side])
    if primitive["clip_mode"] == 1:
        minimum[2] = max(minimum[2], primitive["clip_z"])
    elif primitive["clip_mode"] == -1:
        maximum[2] = min(maximum[2], primitive["clip_z"])
    return minimum, maximum


def _project_bounds(primitive: dict[str, Any], right: np.ndarray, up: np.ndarray) -> tuple[float, float, float, float]:
    # Conservative projected bounds; actual visibility is always ray tested.
    if primitive["kind"] == "sphere":
        center, radius = primitive["center"], primitive["radius"]
        x, y = center @ right, center @ up
        if primitive.get("round_symbol"):
            rx = ry = radius
        else:
            rx, ry = radius * np.linalg.norm(right), radius * np.linalg.norm(up)
        return x - rx, x + rx, y - ry, y + ry
    if primitive["kind"] == "cylinder":
        axis, radius = primitive["axis"], primitive["radius"]
        mode = primitive["clip_mode"]
        axis_is_y = abs(axis[0]) < 1e-12 and abs(axis[2]) < 1e-12
        if mode == 0 or axis_is_y:
            # Exact finite-cylinder support avoids excess white borders from
            # projecting a world-axis bounding box. The only clipped tubes in
            # this schematic are y-axis half cylinders, whose disk support is
            # also analytic (no artificial bounding-box padding).
            def interval(covector):
                ends = [primitive["p0"] @ covector, primitive["p1"] @ covector]
                if mode == 0:
                    support = radius * math.sqrt(max(0., covector @ covector - (covector @ axis) ** 2))
                    return min(ends) - support, max(ends) + support
                def support(sign):
                    cx, cz = sign * covector[0], sign * covector[2]
                    return radius * (math.hypot(cx, cz) if mode * cz >= 0 else abs(cx))
                return min(ends) - support(-1), max(ends) + support(1)
            xmin, xmax = interval(right)
            ymin, ymax = interval(up)
            return xmin, xmax, ymin, ymax
    minimum, maximum = _bounds(primitive)
    corners = np.array([[x, y, z] for x in (minimum[0], maximum[0])
                        for y in (minimum[1], maximum[1]) for z in (minimum[2], maximum[2])])
    xs, ys = corners @ right, corners @ up
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())


def _intersect(primitive: dict[str, Any], sx: np.ndarray, sy: np.ndarray,
               right: np.ndarray, up: np.ndarray, eye: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest-hit toward-eye depth and shaded RGB for a ray grid."""
    # sx is (1,w), sy is (h,1); right/up map pixels to a ray-origin plane.
    # For oblique projection that plane is intentionally not eye-orthogonal.
    ox, oy, oz = (right[index] * sx + up[index] * sy for index in range(3))
    shape = np.broadcast_shapes(sx.shape, sy.shape)
    depth = np.full(shape, -np.inf, dtype=np.float64)
    intensity = np.ones(shape, dtype=np.float32)
    outline = np.zeros(shape, dtype=bool)
    kind = primitive["kind"]
    lamp = 0.75 * eye + 0.25 * up + 0.15 * right
    lamp /= np.linalg.norm(lamp)
    epsilon = 1.0e-10

    def clipping(t: np.ndarray) -> np.ndarray:
        mode = primitive["clip_mode"]
        if mode == 0:
            return np.ones(shape, dtype=bool)
        return mode * (oz + eye[2] * t - primitive["clip_z"]) >= -epsilon

    def update(t: np.ndarray, valid: np.ndarray, nx: Any, ny: Any, nz: Any,
               outlined: Any = False) -> None:
        accepted = valid & (t > depth)
        depth[accepted] = t[accepted]
        if primitive["shade"]:
            diffuse = np.maximum(0.0, nx * lamp[0] + ny * lamp[1] + nz * lamp[2])
            shade = 0.70 + 0.30 * diffuse
            if np.ndim(shade) == 0:
                intensity[accepted] = shade
            else:
                intensity[accepted] = shade[accepted]
        if np.ndim(outlined) == 0:
            outline[accepted] = outlined
        else:
            outline[accepted] = outlined[accepted]

    if kind == "sphere" and primitive.get("round_symbol"):
        # A world point is ray_right*u + ray_up*v + eye*w. Since the
        # screen projection maps these basis columns to (1,0), (0,1), (0,0),
        # a local sphere has a circular footprint while retaining exact depth.
        frame = np.column_stack([right, up, eye])
        center = np.linalg.solve(frame, primitive["center"])
        radius = primitive["radius"]
        du, dv = sx - center[0], sy - center[1]
        # Sum the squared screen offsets first to preserve x/y symmetry even
        # for samples exactly on the circular rim in floating-point arithmetic.
        discriminant = radius * radius - (du * du + dv * dv)
        valid = discriminant >= 0
        w = np.sqrt(np.maximum(discriminant, 0.0))
        t = center[2] + w
        depth[valid] = t[valid]
        if primitive["shade"]:
            # Fixed screen-space lighting keeps the round symbol's shading
            # spherical; physical oblique normals would distort that cue.
            symbol_lamp = np.array([-0.35, 0.45, 0.82])
            symbol_lamp /= np.linalg.norm(symbol_lamp)
            diffuse = np.maximum(0.0, (du * symbol_lamp[0] +
                dv * symbol_lamp[1] + w * symbol_lamp[2]) / radius)
            intensity[valid] = (0.70 + 0.30 * diffuse)[valid]
    elif kind == "sphere":
        center, radius = primitive["center"], primitive["radius"]
        dx, dy, dz = ox - center[0], oy - center[1], oz - center[2]
        along = dx * eye[0] + dy * eye[1] + dz * eye[2]
        perpendicular_squared = dx * dx + dy * dy + dz * dz - along * along
        discriminant = radius * radius - perpendicular_squared
        valid = discriminant >= 0
        t = -along + np.sqrt(np.maximum(discriminant, 0.0))
        update(t, valid, (dx + eye[0] * t) / radius,
               (dy + eye[1] * t) / radius, (dz + eye[2] * t) / radius)
    elif kind == "square":
        if abs(eye[1]) > 1.0e-12:
            center, side = primitive["center"], primitive["side"]
            t = (center[1] - oy) / eye[1]
            x, z = ox + eye[0] * t, oz + eye[2] * t
            valid = (np.abs(x - center[0]) <= side / 2) & (np.abs(z - center[2]) <= side / 2) & clipping(t)
            update(t, valid, 0.0, -1.0 if eye[1] < 0 else 1.0, 0.0)
    else:
        p0, axis, length, radius = (primitive[key] for key in ("p0", "axis", "length", "radius"))
        dx, dy, dz = ox - p0[0], oy - p0[1], oz - p0[2]
        h0 = dx * axis[0] + dy * axis[1] + dz * axis[2]
        he = float(eye @ axis)
        px, py, pz = dx - h0 * axis[0], dy - h0 * axis[1], dz - h0 * axis[2]
        ex, ey, ez = eye - he * axis
        aa = ex * ex + ey * ey + ez * ez
        bb = 2 * (px * ex + py * ey + pz * ez)
        cc = px * px + py * py + pz * pz - radius * radius
        width = min(primitive["outline_width"], radius)
        outlined = primitive["outline_color"] is not None and width > 0
        if aa > 1.0e-14:
            discriminant = bb * bb - 4 * aa * cc
            root = np.sqrt(np.maximum(discriminant, 0.0))
            # Perpendicular distance to the projected cylinder axis determines
            # the silhouette band; original end rims add separate dark rings.
            ray_distance_squared = np.maximum(0.0, cc + radius * radius - bb * bb / (4 * aa))
            silhouette = ray_distance_squared >= max(0.0, radius - width) ** 2
            for sign in (1.0, -1.0):
                t = (-bb + sign * root) / (2 * aa)
                h = h0 + he * t
                valid = (discriminant >= 0) & (h >= -epsilon) & (h <= length + epsilon) & clipping(t)
                rim = (h <= width) | (h >= length - width)
                update(t, valid, (px + ex * t) / radius, (py + ey * t) / radius,
                       (pz + ez * t) / radius, (silhouette | rim) if outlined else False)
        if abs(he) > 1.0e-12:
            for h, sign in ((0.0, -1.0), (length, 1.0)):
                t = (h - h0) / he
                rx, ry, rz = px + ex * t, py + ey * t, pz + ez * t
                radial_squared = rx * rx + ry * ry + rz * rz
                valid = (radial_squared <= radius * radius + epsilon) & clipping(t)
                update(t, valid, sign * axis[0], sign * axis[1], sign * axis[2],
                       radial_squared >= max(0.0, radius - width) ** 2 if outlined else False)
        if primitive["clip_mode"] and abs(eye[2]) > 1.0e-12:
            # Closing the clipped solid adds a same-color plane, no rim stroke.
            t = (primitive["clip_z"] - oz) / eye[2]
            h = h0 + he * t
            rx, ry, rz = px + ex * t, py + ey * t, pz + ez * t
            valid = (h >= -epsilon) & (h <= length + epsilon) & (rx * rx + ry * ry + rz * rz <= radius * radius + epsilon)
            update(t, valid, 0.0, 0.0, -float(primitive["clip_mode"]), False)
    rgb = intensity[..., None] * primitive["color"].astype(np.float32)
    if primitive.get("outline_color") is not None:
        rgb[outline] = primitive["outline_color"]
    return depth, rgb


def _render_arrays(primitives: list[dict[str, Any]], width: int, height: int | None,
                   supersample: int, elev: float, azim: float,
                   projection: str = "orthographic", depth_scale: float = 0.5,
                   depth_angle: float = 45, padding_pixels: float = 2.) -> tuple[np.ndarray, dict[str, Any]]:
    right, up, ray_right, ray_up, eye = camera_frame(projection, elev, azim, depth_scale, depth_angle)
    symbol_frame = np.column_stack([ray_right, ray_up, eye])
    for primitive in primitives:
        if primitive.get("round_symbol"):
            primitive["_symbol_frame"] = symbol_frame
    projected = np.array([_project_bounds(primitive, right, up) for primitive in primitives])
    xmin, xmax = projected[:, 0].min(), projected[:, 1].max()
    ymin, ymax = projected[:, 2].min(), projected[:, 3].max()
    scene_width, scene_height = float(xmax - xmin), float(ymax - ymin)
    padding_pixels = float(padding_pixels)
    if not math.isfinite(padding_pixels) or padding_pixels < 0 or 2 * padding_pixels >= width:
        raise ValueError("padding_pixels must be finite, nonnegative, and less than half the width")
    if height is None:
        world_per_pixel = scene_width / (width - 2 * padding_pixels)
        height = max(1, int(math.ceil(scene_height / world_per_pixel + 2 * padding_pixels)))
    else:
        if 2 * padding_pixels >= height:
            raise ValueError("padding_pixels must be less than half the height")
        world_per_pixel = max(scene_width / (width - 2 * padding_pixels),
                              scene_height / (height - 2 * padding_pixels))
    xcenter, ycenter = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    viewport_width = width * world_per_pixel
    viewport_height = height * world_per_pixel
    xmin, xmax = xcenter - viewport_width / 2, xcenter + viewport_width / 2
    ymin, ymax = ycenter - viewport_height / 2, ycenter + viewport_height / 2
    high_width, high_height = width * supersample, height * supersample
    pixel = viewport_width / high_width
    xs = xmin + (np.arange(high_width) + 0.5) * pixel
    ys = ymax - (np.arange(high_height) + 0.5) * pixel
    opaque_rgb = np.ones((high_height, high_width, 3), dtype=np.float32)

    def region(bounds: np.ndarray) -> tuple[int, int, int, int]:
        bx0, bx1, by0, by1 = bounds
        ix0 = max(0, int(math.floor((bx0 - xmin) / pixel)) - 1)
        ix1 = min(high_width, int(math.ceil((bx1 - xmin) / pixel)) + 1)
        iy0 = max(0, int(math.floor((ymax - by1) / pixel)) - 1)
        iy1 = min(high_height, int(math.ceil((ymax - by0) / pixel)) + 1)
        return ix0, ix1, iy0, iy1

    symbol_parts = {}
    for primitive, primitive_bounds in zip(primitives, projected):
        if primitive.get("symbol_id") is not None:
            symbol_parts.setdefault(primitive["symbol_id"], []).append((primitive, primitive_bounds))
    symbol_masks = {}
    for symbol_id, parts in symbol_parts.items():
        part_bounds = np.array([bound for _, bound in parts])
        union_bounds = [part_bounds[:, 0].min(), part_bounds[:, 1].max(),
                        part_bounds[:, 2].min(), part_bounds[:, 3].max()]
        x0, x1, y0, y1 = region(union_bounds)
        mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        for part, _ in parts:
            hits, _ = _intersect(part, xs[None, x0:x1], ys[y0:y1, None], ray_right, ray_up, eye)
            mask |= np.isfinite(hits)
        symbol_masks[symbol_id] = (x0, x1, y0, y1, mask)

    def apply_symbol_precedence(primitive, depths, primitive_region):
        px0, px1, py0, py1 = primitive_region
        for owner in primitive.get("symbol_owners", ()):
            if owner not in symbol_masks:
                continue
            sx0, sx1, sy0, sy1, mask = symbol_masks[owner]
            x0, x1, y0, y1 = max(px0, sx0), min(px1, sx1), max(py0, sy0), min(py1, sy1)
            if x0 >= x1 or y0 >= y1:
                continue
            overlap = depths[y0 - py0:y1 - py0, x0 - px0:x1 - px0]
            overlap[mask[y0 - sy0:y1 - sy0, x0 - sx0:x1 - sx0]] = -np.inf

    def paint_layer(items):
        # A separate depth buffer makes the explicit background context an
        # underlay. Within this layer, drawing order never controls occlusion.
        opaque_depth = np.full((high_height, high_width), -np.inf, dtype=np.float64)
        for primitive, bounds in items:
            if primitive["alpha"] < 1:
                continue
            ix0, ix1, iy0, iy1 = region(bounds)
            depth, rgb = _intersect(primitive, xs[None, ix0:ix1], ys[iy0:iy1, None], ray_right, ray_up, eye)
            apply_symbol_precedence(primitive, depth, (ix0, ix1, iy0, iy1))
            current_depth = opaque_depth[iy0:iy1, ix0:ix1]
            update = depth > current_depth
            current_depth[update] = depth[update]
            opaque_rgb[iy0:iy1, ix0:ix1][update] = rgb[update]

        fragment_pixels, fragment_depths, fragment_colors, fragment_alphas = [], [], [], []
        for primitive, bounds in items:
            if primitive["alpha"] >= 1:
                continue
            ix0, ix1, iy0, iy1 = region(bounds)
            depth, rgb = _intersect(primitive, xs[None, ix0:ix1], ys[iy0:iy1, None], ray_right, ray_up, eye)
            apply_symbol_precedence(primitive, depth, (ix0, ix1, iy0, iy1))
            visible = np.isfinite(depth) & (depth > opaque_depth[iy0:iy1, ix0:ix1] + 1.0e-9)
            rows, columns = np.nonzero(visible)
            if rows.size:
                fragment_pixels.append((rows + iy0) * high_width + columns + ix0)
                fragment_depths.append(depth[visible])
                fragment_colors.append(rgb[visible])
                fragment_alphas.append(np.full(rows.size, primitive["alpha"], dtype=np.float32))
        fragment_count = sum(fragment.size for fragment in fragment_pixels)
        if fragment_count:
            indices = np.concatenate(fragment_pixels)
            depths = np.concatenate(fragment_depths)
            colors = np.concatenate(fragment_colors)
            alphas = np.concatenate(fragment_alphas).astype(np.float64)
            order = np.lexsort((-depths, indices))
            indices, colors, alphas = indices[order], colors[order], alphas[order]
            starts = np.r_[0, np.flatnonzero(indices[1:] != indices[:-1]) + 1]
            counts = np.diff(np.r_[starts, len(indices)])
            log_transmission = np.log1p(-alphas)
            cumulative = np.cumsum(log_transmission, dtype=np.float64)
            before = cumulative - log_transmission
            group_offsets = before[starts]
            transmission_before = np.exp(before - np.repeat(group_offsets, counts))
            weights = alphas * transmission_before
            foreground = np.add.reduceat(colors * weights[:, None], starts, axis=0)
            total_log_transmission = np.add.reduceat(log_transmission, starts)
            unique_pixels = indices[starts]
            flattened = opaque_rgb.reshape(-1, 3)
            flattened[unique_pixels] = foreground + flattened[unique_pixels] * np.exp(total_log_transmission)[:, None]
        return fragment_count

    background = [(p, b) for p, b in zip(primitives, projected) if p.get("context_background")]
    foreground = [(p, b) for p, b in zip(primitives, projected) if not p.get("context_background")]
    background_fragment_count = paint_layer(background)
    fragment_count = background_fragment_count + paint_layer(foreground)

    world_bounds = [_bounds(primitive) for primitive in primitives]
    metadata = {
        "projection": "orthographic_equal_world_units" if projection == "orthographic" else "cabinet_oblique",
        "elev": float(elev) if projection == "orthographic" else None,
        "azim": float(azim) if projection == "orthographic" else None,
        "depth_scale": float(depth_scale) if projection == "cabinet" else None,
        "depth_angle": float(depth_angle) if projection == "cabinet" else None,
        "screen_projection_matrix": [right.tolist(), up.tolist()],
        "camera_right": right.tolist(), "camera_up": up.tolist(), "camera_toward_eye": eye.tolist(),
        "ray_origin_right": ray_right.tolist(), "ray_origin_up": ray_up.tolist(),
        "world_bounds": {"minimum": np.min([bound[0] for bound in world_bounds], axis=0).tolist(),
                         "maximum": np.max([bound[1] for bound in world_bounds], axis=0).tolist()},
        "screen_bounds": {"xmin": float(xmin), "xmax": float(xmax), "ymin": float(ymin), "ymax": float(ymax)},
        "width": width, "height": height, "supersample": supersample,
        "padding_output_pixels": padding_pixels,
        "world_units_per_output_pixel": float(pixel * supersample),
        "primitive_count": len(primitives), "transparent_fragment_count": fragment_count,
        "occlusion": "analytic_per_pixel_nearest_opaque_then_depth_sorted_transparent_fragments",
        "transparent_solid_model": "one_nearest_surface_per_object",
        "context_background_policy": "explicit_schematic_underlay_before_depth_tested_foreground",
        "context_background_primitive_count": len(background),
        "context_background_fragment_count": background_fragment_count,
        "symbol_foreground_policy": "owner_symbol_silhouette_hides_attached_objects_only;unrelated_foreground_uses_ray_depth",
        "symbol_silhouette_count": len(symbol_masks),
        "symbol_owned_primitive_count": sum(bool(p.get("symbol_owners")) for p in primitives),
        "chi_square_occlusion_policy": "chi_tubes_unmasked_share_actual_foreground_ray_depth_with_square_faces",
        "chi_chain_true_depth_primitive_count": sum(p.get("role") == "chi_chain" and not p.get("symbol_owners") for p in primitives),
    }
    return np.clip(opaque_rgb, 0.0, 1.0), metadata


def render_scene(elements: Sequence[dict[str, Any]], output_path: str | Path,
                 width: int = 1800, height: int | None = None, supersample: int = 2,
                 elev: float = 30, azim: float = -60, projection: str = "orthographic",
                 depth_scale: float = 0.5, depth_angle: float = 45,
                 padding_pixels: float = 2.) -> dict[str, Any]:
    """Save a white-background depth-correct image; return JSON-safe metadata.

    height=None selects a height fitting the projected scene's aspect ratio.
    A supplied height fits the full scene with white margins and preserves
    the projection's aspect ratio. Cabinet projection keeps the x-z face in
    true proportions and uses depth_scale/depth_angle for +y. Supersampling uses Pillow's Lanczos filter.
    Optional element ``shade=False`` requests the exact supplied flat RGB.
    Default padding is two output pixels; screen_bounds retain the exact mapping.
    """
    if isinstance(width, bool) or int(width) != width or width <= 0:
        raise ValueError("width must be a positive integer")
    if height is not None and (isinstance(height, bool) or int(height) != height or height <= 0):
        raise ValueError("height must be a positive integer or None")
    if isinstance(supersample, bool) or int(supersample) != supersample or supersample <= 0:
        raise ValueError("supersample must be a positive integer")
    rgb, metadata = _render_arrays(_expand_elements(elements), int(width), None if height is None else int(height),
                                   int(supersample), elev, azim, projection, depth_scale, depth_angle, padding_pixels)
    image = Image.fromarray(np.round(rgb * 255).astype(np.uint8))
    if supersample != 1:
        image = image.resize((metadata["width"], metadata["height"]), Image.Resampling.LANCZOS)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, dpi=(300, 300))
    metadata["output_path"] = str(output.resolve())
    return metadata


def _smoke_test() -> None:
    directory = Path(__file__).resolve().parent
    right, up, eye = camera_basis()
    common = {"color": [1.0, 0.0, 0.0], "shade": False}
    front = {"kind": "sphere", "center": (0.35 * eye).tolist(), "radius": 0.2, **common}
    back = {"kind": "sphere", "center": (-0.35 * eye).tolist(), "radius": 0.2,
            "color": [0.0, 0.0, 1.0], "shade": False}
    a, _ = _render_arrays(_expand_elements([front, back]), 121, 121, 1, 30, -60)
    b, _ = _render_arrays(_expand_elements([back, front]), 121, 121, 1, 30, -60)
    assert np.array_equal(a, b), "Opaque result must not depend on whole-object ordering"
    assert np.allclose(a[60, 60], [1, 0, 0]), "Nearest opaque sphere must fully hide the rear sphere"
    back["alpha"] = 0.35
    hidden, _ = _render_arrays(_expand_elements([back, front]), 121, 121, 1, 30, -60)
    assert np.allclose(hidden[60, 60], [1, 0, 0]), "Ghost behind an opaque surface must not show through"
    front["alpha"], back["alpha"] = 0.35, 1.0
    ghost, _ = _render_arrays(_expand_elements([front, back]), 121, 121, 1, 30, -60)
    assert np.allclose(ghost[60, 60], [0.35, 0, 0.65]), "A front ghost must alpha blend with the nearest opaque hit"
    clipped = _expand_elements([{"kind": "square", "center": [0, 0, 0], "side": 1,
                                 "color": [0.7, 0.8, 0.9], "frame_color": [0, 0, 0.3], "half": "upper"}])
    assert len(clipped) == 4, "Half-square should contain a plane and exactly three original outer edges"
    assert all(not (primitive["p0"][2] == primitive["p1"][2] == 0) for primitive in clipped if primitive["kind"] == "cylinder"), "No waist frame may be introduced"
    cylinder = _expand_elements([{"kind": "cylinder", "p0": [-1, 0, 0], "p1": [1, 0, 0], "radius": 0.1, "color": [0, 0, 1]}])[0]
    depth, _ = _intersect(cylinder, np.array([[0.0]]), np.array([[0.0]]), right, up, eye)
    assert abs(float(depth[0, 0]) - 0.1 / math.sqrt(1 - eye[0] ** 2)) < 1e-12
    scene = [
        {"kind": "square", "center": [0, 0.15, 0], "side": 1.3, "color": [0.91, 0.96, 1.0], "frame_color": [0.02, 0.12, 0.4], "frame_radius": 0.018},
        {"kind": "cylinder", "p0": [-0.85, -0.4, -0.35], "p1": [0.85, 0.4, 0.35], "radius": 0.035, "color": [0.12, 0.27, 0.72]},
        {"kind": "sphere", "center": [-0.46, -0.22, -0.22], "radius": 0.12, "color": [0.88, 0.20, 0.12]},
        {"kind": "sphere", "center": [0.4, 0.4, 0.2], "radius": 0.13, "color": [0.20, 0.62, 0.30]},
        {"kind": "square", "center": [-1.1, 0.1, 0], "side": 0.65, "color": [0.3, 0.55, 0.9], "frame_color": [0.02, 0.12, 0.4], "frame_radius": 0.012, "alpha": 0.35, "half": "lower"},
        {"kind": "square", "center": [-1.1, 0.1, 0], "side": 0.65, "color": [0.3, 0.55, 0.9], "frame_color": [0.02, 0.12, 0.4], "frame_radius": 0.012, "half": "upper"},
        {"kind": "cylinder", "p0": [0.8, -0.5, -0.1], "p1": [0.8, 0.45, -0.1], "radius": 0.13,
         "color": [0.3, 0.55, 0.9], "outline_color": [0.02, 0.12, 0.4], "outline_width": 0.012, "half": "upper"},
        {"kind": "cylinder", "p0": [0.8, -0.5, -0.1], "p1": [0.8, 0.45, -0.1], "radius": 0.13,
         "color": [0.3, 0.55, 0.9], "outline_color": [0.02, 0.12, 0.4], "outline_width": 0.012, "half": "lower", "alpha": 0.35},
    ]
    metadata = render_scene(scene, directory / "_renderer_smoke.png", width=1000, supersample=2)
    metadata["smoke_checks"] = ["opaque order invariance", "opaque hides rear ghost", "front ghost compositing", "no half-square waist frame", "analytic cylinder depth"]
    (directory / "_renderer_smoke_meta.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(metadata['smoke_checks'])} numerical rendering checks; {metadata['output_path']}")


if __name__ == "__main__":
    _smoke_test()
