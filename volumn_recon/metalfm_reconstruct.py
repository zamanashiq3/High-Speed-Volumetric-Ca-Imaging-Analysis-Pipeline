#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
metalfm_reconstruct.py

One-command Meta-LFM reconstruction:

    python metalfm_reconstruct.py raw_image.tif

Everything else is fixed in CONFIG below. Steps:
  1. load + normalise the raw image
  2. measure lenslet pitch / origin by FFT
  3. extract sub-aperture views, keep only those inside the pupil image
  4. shift-sum refocus, z in micrometres (parallax from the optics in CONFIG)
  5. dense shift-and-add stack
  6. multi-view Richardson-Lucy on the lenslet grid (+ dense display version)
  7. save TIFF stacks, QC figures, metrics
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import tifffile as tiff

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.ndimage import map_coordinates, shift as ndi_shift, gaussian_filter, gaussian_filter1d
from scipy.signal import find_peaks

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None



# =============================================================================
# FIXED CONFIGURATION  (edit here, not on the command line)
# =============================================================================

CONFIG = dict(
    # ---- optics (from the system table) ----
    wavelength_um=0.525,        # emission
    na=0.50,                    # objective NA
    magnification=20.0,         # objective
    pixel_size_um=6.5,          # camera pixel
    lens_pitch_um=75.0,         # metalens pitch
    lens_focal_um=1430.0,       # metalens focal length
    n_medium=1.33,               # sample immersion index (1.33 for water dipping)

    # ---- lenslet grid: measured from each image by FFT ----
    pitch=None, pitch_y=None, origin_x=None, origin_y=None,
    pitch_min=8, pitch_max=18,
    autocal=False, cal_crop_lenslets=36, cal_pitch_search=0.05,
    cal_origin_search=0.75, cal_fine_rounds=2,

    # ---- sub-aperture views ----
    lenslet_samples=9,          # 9 x 9 samples, ~1 sensor px apart
    extraction_scale=0.36,      # +/- 0.36 pitch around each lenslet centre
    aperture_threshold=0.25,    # drop views dimmer than 25% of the brightest
    no_view_normalize=False,
    remove_common_background=False, background_strength=0.65,

    # ---- depth axis, micrometres of object defocus ----
    z_min=-30.0, z_max=30.0, z_steps=61,       # 10 um planes
    depth_curve=0.0,            # linear parallax
    shift_scale=-1.0,            # multiplier on the physical parallax (-1 flips z)
    angular_scale_x=1.0, angular_scale_y=1.0,
    auto_shift_scale=False,     # only sensible on bead images
    interp_order=1,

    # ---- dense output ----
    run_highres=True, target_size=512, highres_sigma=1.0,
    run_highres_rl=True, highres_rl_iter=30, highres_init_sigma=0.8,
    save_psf_stack=False, sigma_zR=18.0,

    # ---- Richardson-Lucy ----
    run_rl=True, device="cuda",                # falls back to CPU automatically
    rl_iter=10,
    sigma0=0.5,                 # in-focus PSF sigma, lenslet px
    sigma_ang_px=None,          # None -> lambda * F# / pixel / 2.355
    damping=0.5, clip_update_low=0.5, clip_update_high=2.0, ratio_clip=6.0,
    axial_smooth_strength=0.05, xy_smooth_strength=0.10,

    skip_correction=False, save_svg=False,
)

# =============================================================================
# Basic IO
# =============================================================================

def imread_tiff_safe(path: Path) -> np.ndarray:
    if path.suffix.lower() not in (".tif", ".tiff"):
        return np.asarray(plt.imread(str(path)))        # PNG / JPG fallback
    with tiff.TiffFile(str(path)) as tif:
        arr = tif.asarray()
    arr = np.asarray(arr)
    if arr.dtype.byteorder not in ("=", "|"):
        arr = arr.view(arr.dtype.newbyteorder("="))
    return arr


def ensure_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            return arr[0]
        if arr.shape[-1] in (1, 3, 4):
            return arr[..., :3].mean(axis=-1)
        return np.max(arr, axis=0)
    raise ValueError(f"Expected 2D image-compatible TIFF. Got {arr.shape}")


def robust_norm(img: np.ndarray, p_low=0.1, p_high=99.9) -> np.ndarray:
    x = np.asarray(img, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    lo, hi = np.percentile(x, [p_low, p_high])
    if hi <= lo:
        lo, hi = float(x.min()), float(x.max())
    x = (x - lo) / (hi - lo + 1e-8)
    x = np.clip(x, 0.0, 1.0)
    return x.astype(np.float32)


def to_uint16_auto(vol: np.ndarray, p_high=99.95) -> np.ndarray:
    x = np.asarray(vol, dtype=np.float32)
    x = np.nan_to_num(x, nan=0, posinf=0, neginf=0)
    x = np.clip(x, 0, None)
    hi = np.percentile(x, p_high)
    if hi <= 0:
        hi = x.max() if x.max() > 0 else 1.0
    x = np.clip(x / hi, 0, 1)
    return (x * 65535).astype(np.uint16)


def save_img(img, out: Path, cmap="gray", title="", save_svg=False):
    plt.figure(figsize=(6, 6))
    x = np.asarray(img, dtype=np.float32)
    vmin, vmax = np.percentile(x, [0.5, 99.7])
    plt.imshow(x, cmap=cmap, vmin=vmin, vmax=vmax)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    if save_svg:
        plt.savefig(out.with_suffix(".svg"))
    plt.close()


# =============================================================================
# Geometry: ONE definition of parallax, used by every stage
# =============================================================================

def parallax_gain(pixel_size_um, magnification, lens_focal_um, lens_pitch_um, n_medium=1.0):
    """
    Parallax of a classic (unfocused) LFM, in LENSLETS per (sensor pixel of
    intra-lenslet offset) per (micrometre of object defocus).

      ray angle behind the lenslet      theta = d_px * pixel / f_lens
      image-side defocus                dz'   = dz * M^2 / n
      lateral walk at the lenslet plane theta * dz'
      in lenslet units                  theta * dz' / lens_pitch

    For 6.5 um pixels, 20x, f = 1430 um, pitch = 75 um  ->  0.0242.
    """
    return float(pixel_size_um * magnification ** 2 /
                 (max(n_medium, 1e-6) * lens_focal_um * lens_pitch_um))


def effective_z(z, depth_curve=0.0):
    """Optional soft compression of large |z|. Monotonic (the old z/(1+k z^2)
    folded back on itself beyond |z| = 1/sqrt(k), duplicating planes)."""
    z = float(z)
    if depth_curve <= 0:
        return z
    return z / math.sqrt(1.0 + depth_curve * z * z)


def compute_depth_shift(z, du_px, dv_px, kx, ky, depth_curve=0.0):
    """
    Refocus shift of one sub-aperture view, in LENSLET-GRID pixels
    (the pixel unit of the extracted views).

    du_px, dv_px : offset of this view from the lenslet centre, in sensor pixels
    kx, ky       : lenslets per (sensor px * z unit), see parallax_gain()
    """
    ze = effective_z(z, depth_curve)
    return -ze * float(du_px) * kx, -ze * float(dv_px) * ky


def estimate_pitch_1d(profile: np.ndarray, min_pitch=8, max_pitch=18):
    """Sub-pixel pitch + origin from the fundamental peak of the profile spectrum.
    (Autocorrelation lag only gives an integer pitch: 0.3 px error x 60 lenslets
    = 18 px of grid drift across the field.)"""
    p = profile.astype(np.float64)
    p = p - p.mean()
    n = 1 << 18
    spec = np.abs(np.fft.rfft(p * np.hanning(len(p)), n))
    f = np.fft.rfftfreq(n)
    band = (f > 1.0 / max_pitch) & (f < 1.0 / min_pitch)
    if not np.any(band) or spec[band].max() <= 0:
        return 11.54, 0.0
    k = int(np.argmax(np.where(band, spec, 0)))
    pitch = 1.0 / f[k]
    x = np.arange(len(p))
    ph = np.angle(np.sum(p * np.exp(-2j * np.pi * x / pitch)))
    origin = (-ph / (2 * np.pi) * pitch) % pitch
    return float(pitch), float(origin)


def estimate_pitch_xy(img: np.ndarray, min_pitch=8, max_pitch=18) -> Tuple[float, float]:
    return (
        estimate_pitch_1d(np.mean(img, axis=0), min_pitch, max_pitch)[0],
        estimate_pitch_1d(np.mean(img, axis=1), min_pitch, max_pitch)[0],
    )


def estimate_origin_from_profile(profile: np.ndarray, pitch: float) -> float:
    """Phase of the fundamental at the given pitch = centroid of the spots."""
    p = profile.astype(np.float64)
    p = p - p.mean()
    x = np.arange(len(p))
    ph = np.angle(np.sum(p * np.hanning(len(p)) * np.exp(-2j * np.pi * x / pitch)))
    return float((-ph / (2 * np.pi) * pitch) % pitch)


def estimate_origin_xy(img: np.ndarray, pitch_x: float, pitch_y: float) -> Tuple[float, float]:
    return (
        estimate_origin_from_profile(np.mean(img, axis=0), pitch_x),
        estimate_origin_from_profile(np.mean(img, axis=1), pitch_y),
    )


def grid_centers_float(length: int, origin: float, pitch: float, margin: float) -> np.ndarray:
    centers = []
    c = float(origin)
    while c < length:
        if c - margin >= 0 and c + margin < length:
            centers.append(c)
        c += pitch
    return np.asarray(centers, dtype=np.float32)


def save_grid_overlay(img: np.ndarray, out: Path, centers_x, centers_y, save_svg=False):
    plt.figure(figsize=(8, 8))
    vmin, vmax = np.percentile(img, [0.5, 99.5])
    plt.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)
    for x in centers_x:
        plt.axvline(x, color="cyan", linewidth=0.3, alpha=0.55)
    for y in centers_y:
        plt.axhline(y, color="cyan", linewidth=0.3, alpha=0.55)
    plt.title("Lenslet grid")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    if save_svg:
        plt.savefig(out.with_suffix(".svg"))
    plt.close()


# View extraction

def extract_views(
    img: np.ndarray,
    pitch_x: float,
    pitch_y: float,
    origin_x: float,
    origin_y: float,
    lenslet_samples: int = 9,
    extraction_scale: float = 0.36,
    normalize_views: bool = True,
    crop_lenslets: Optional[int] = None,
    aperture_threshold: float = 0.0,
):
    """
    Sample L x L positions inside every lenslet cell, spanning
    +/- extraction_scale * pitch around the lenslet centre.

    aperture_threshold > 0: views whose mean brightness is below that fraction
    of the brightest view lie outside the pupil image (dark gap between spots).
    They are flagged in info["view_mask"] and set to zero, instead of being
    gain-normalised up to full brightness (which turned noise into "signal").
    """
    img = img.astype(np.float32)
    H, W = img.shape

    L = int(lenslet_samples)
    if L % 2 == 0:
        raise ValueError("lenslet_samples must be odd.")

    span_x = extraction_scale * pitch_x
    span_y = extraction_scale * pitch_y
    margin = max(span_x, span_y) + 1

    centers_x = grid_centers_float(W, origin_x, pitch_x, margin)
    centers_y = grid_centers_float(H, origin_y, pitch_y, margin)

    if crop_lenslets is not None:
        crop_lenslets = int(crop_lenslets)
        if len(centers_x) > crop_lenslets:
            s = (len(centers_x) - crop_lenslets) // 2
            centers_x = centers_x[s:s+crop_lenslets]
        if len(centers_y) > crop_lenslets:
            s = (len(centers_y) - crop_lenslets) // 2
            centers_y = centers_y[s:s+crop_lenslets]

    Ny, Nx = len(centers_y), len(centers_x)

    c = (L - 1) / 2
    a = (np.arange(L) - c) / c

    ux = a * pitch_x * extraction_scale      # sensor-pixel offsets
    vy = a * pitch_y * extraction_scale

    Cx, Cy = np.meshgrid(centers_x, centers_y, indexing="xy")
    views = np.zeros((L, L, Ny, Nx), dtype=np.float32)

    for iv, dy in enumerate(vy):
        for iu, dx in enumerate(ux):
            views[iv, iu] = map_coordinates(
                img,
                [(Cy + dy).ravel(), (Cx + dx).ravel()],
                order=1,
                mode="nearest",
            ).reshape(Ny, Nx)

    means = views.mean(axis=(2, 3))
    rel = means / (means.max() + 1e-12)
    mask = rel >= float(aperture_threshold)

    if normalize_views:
        # mean (not median: the median of a sparse fluorescence view is background)
        gm = float(means[mask].mean())
        gain = gm / np.maximum(means, 1e-3 * means.max() + 1e-12)
        views = views * gain[:, :, None, None]

    views[~mask] = 0.0

    info = {
        "pitch_x": float(pitch_x),
        "pitch_y": float(pitch_y),
        "origin_x": float(origin_x),
        "origin_y": float(origin_y),
        "lenslet_samples": int(L),
        "extraction_scale": float(extraction_scale),
        "num_lenslets_x": int(Nx),
        "num_lenslets_y": int(Ny),
        "centers_x": centers_x.tolist(),
        "centers_y": centers_y.tolist(),
        "du_px": ux.tolist(),
        "dv_px": vy.tolist(),
        "view_mask": mask.tolist(),
        "view_rel_brightness": rel.round(4).tolist(),
    }

    return views.astype(np.float32), info


def make_geometry(grid_info, k_lenslet_per_px_um, shift_scale=1.0,
                  angular_scale_x=1.0, angular_scale_y=1.0, depth_curve=0.0):
    """Everything the refocus / PSF / RL stages need to agree on."""
    du = np.asarray(grid_info["du_px"], dtype=np.float64)
    dv = np.asarray(grid_info["dv_px"], dtype=np.float64)
    return {
        "du": du,
        "dv": dv,
        "mask": np.asarray(grid_info["view_mask"], dtype=bool),
        "kx": float(k_lenslet_per_px_um * shift_scale * angular_scale_x),
        "ky": float(k_lenslet_per_px_um * shift_scale * angular_scale_y),
        "depth_curve": float(depth_curve),
        "rmax": float(max(np.abs(du).max(), np.abs(dv).max(), 1e-6)),
    }


def _view_list(geom):
    """[(iv, iu, du_px, dv_px, weight)] for the views inside the aperture."""
    out = []
    for iv, dv in enumerate(geom["dv"]):
        for iu, du in enumerate(geom["du"]):
            if geom["mask"][iv, iu]:
                r2 = (du * du + dv * dv) / geom["rmax"] ** 2
                out.append((iv, iu, float(du), float(dv), 1.0 / (1.0 + 0.30 * r2)))
    return out


def remove_common_background(views: np.ndarray, strength=0.85, view_mask=None) -> np.ndarray:
    sel = views if view_mask is None else views[np.asarray(view_mask, dtype=bool)]
    common = np.median(sel.reshape(-1, *views.shape[2:]), axis=0)[None, None]
    out = np.clip(views - strength * common, 0, None)
    if view_mask is not None:
        out[~np.asarray(view_mask, dtype=bool)] = 0
    out = out / (np.mean(out) + 1e-8)
    return out.astype(np.float32)


def save_view_montage(views: np.ndarray, out: Path, save_svg=False):
    L = views.shape[0]
    fig, axs = plt.subplots(L, L, figsize=(L * 1.0, L * 1.0))
    vmax = np.percentile(views, 99.5)
    for v in range(L):
        for u in range(L):
            axs[v, u].imshow(views[v, u], cmap="gray", vmin=0, vmax=vmax)
            axs[v, u].axis("off")
    plt.tight_layout()
    plt.savefig(out, dpi=180)
    if save_svg:
        plt.savefig(out.with_suffix(".svg"))
    plt.close()


# =============================================================================
# Calibration score
# =============================================================================

def _corr(a, b):
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    return float(np.sum(a*b) / ((np.sqrt(np.sum(a*a)) + 1e-8) * (np.sqrt(np.sum(b*b)) + 1e-8)))


def _sharpness(im):
    gy, gx = np.gradient(im.astype(np.float32))
    return float(np.mean(np.sqrt(gx*gx + gy*gy)))


def calibration_score(
    img,
    px,
    py,
    ox,
    oy,
    lenslet_samples,
    extraction_scale,
    crop_lenslets,
):
    try:
        views, info = extract_views(
            img,
            px,
            py,
            ox,
            oy,
            lenslet_samples=lenslet_samples,
            extraction_scale=extraction_scale,
            crop_lenslets=crop_lenslets,
        )
    except Exception as e:
        return -1e9, {"error": str(e)}

    L = lenslet_samples
    c = L // 2
    center = views[c, c]

    neigh = [(c, c-1), (c, c+1), (c-1, c), (c+1, c)]
    neighbor_corr = np.mean([_corr(center, views[i, j]) for i, j in neigh])

    opp = [
        ((c, c-2), (c, c+2)),
        ((c-2, c), (c+2, c)),
        ((c-2, c-2), (c+2, c+2)),
        ((c-2, c+2), (c+2, c-2)),
    ]

    opp_corr = []
    for a, b in opp:
        if 0 <= a[0] < L and 0 <= a[1] < L and 0 <= b[0] < L and 0 <= b[1] < L:
            opp_corr.append(_corr(views[a], views[b]))

    opposite_sym = float(np.mean(opp_corr)) if opp_corr else 0.0
    center_sharp = _sharpness(center) / (np.std(center) + 1e-8)

    means = np.mean(views, axis=(2, 3))
    brightness_cv = float(np.std(means) / (np.mean(means) + 1e-8))

    score = 2.0 * neighbor_corr + opposite_sym + 0.35 * center_sharp - 0.8 * brightness_cv

    details = {
        "score": float(score),
        "neighbor_corr": float(neighbor_corr),
        "opposite_sym": float(opposite_sym),
        "center_sharp": float(center_sharp),
        "brightness_cv": float(brightness_cv),
        "num_lenslets_x": info["num_lenslets_x"],
        "num_lenslets_y": info["num_lenslets_y"],
    }

    return float(score), details


def auto_calibrate_grid(
    img,
    px0,
    py0,
    ox0,
    oy0,
    lenslet_samples,
    extraction_scale,
    crop_lenslets=36,
    pitch_search=0.35,
    origin_search=0.75,
    fine_rounds=2,
):
    best = {
        "pitch_x": float(px0),
        "pitch_y": float(py0),
        "origin_x": float(ox0),
        "origin_y": float(oy0),
    }

    history = []

    def eval_one(px, py, ox, oy):
        score, details = calibration_score(
            img, px, py, ox, oy,
            lenslet_samples=lenslet_samples,
            extraction_scale=extraction_scale,
            crop_lenslets=crop_lenslets,
        )
        rec = {
            "pitch_x": float(px),
            "pitch_y": float(py),
            "origin_x": float(ox),
            "origin_y": float(oy),
            "score": float(score),
            "details": details,
        }
        history.append(rec)
        return score, details

    best_score, best_details = eval_one(px0, py0, ox0, oy0)
    print(f"[CAL] initial score={best_score:.5f} params={best}")

    for dxp in np.linspace(-pitch_search, pitch_search, 5):
        for dyp in np.linspace(-pitch_search, pitch_search, 5):
            px = px0 + dxp
            py = py0 + dyp
            score, details = eval_one(px, py, best["origin_x"], best["origin_y"])
            if score > best_score:
                best_score = score
                best_details = details
                best["pitch_x"] = float(px)
                best["pitch_y"] = float(py)
                print(f"[CAL] better pitch score={score:.5f} px={px:.4f} py={py:.4f}")

    for dxo in np.linspace(-origin_search, origin_search, 7):
        for dyo in np.linspace(-origin_search, origin_search, 7):
            ox = ox0 + dxo
            oy = oy0 + dyo
            score, details = eval_one(best["pitch_x"], best["pitch_y"], ox, oy)
            if score > best_score:
                best_score = score
                best_details = details
                best["origin_x"] = float(ox)
                best["origin_y"] = float(oy)
                print(f"[CAL] better origin score={score:.5f} ox={ox:.4f} oy={oy:.4f}")

    p_radius = pitch_search / 2
    o_radius = origin_search / 2

    for r in range(fine_rounds):
        improved = True
        while improved:
            improved = False
            candidates = []
            for name, radius in [
                ("pitch_x", p_radius),
                ("pitch_y", p_radius),
                ("origin_x", o_radius),
                ("origin_y", o_radius),
            ]:
                for delta in [-radius, -radius/2, radius/2, radius]:
                    cand = best.copy()
                    cand[name] += float(delta)
                    candidates.append(cand)

            for cand in candidates:
                score, details = eval_one(
                    cand["pitch_x"], cand["pitch_y"], cand["origin_x"], cand["origin_y"]
                )
                if score > best_score:
                    best = cand
                    best_score = score
                    best_details = details
                    improved = True
                    print(f"[CAL] fine r{r+1} score={score:.5f} params={best}")

        p_radius /= 2
        o_radius /= 2

    best["score"] = float(best_score)
    best["details"] = best_details
    return best, history



# Refocus

def refocus_shift_sum(views, z_values, geom, order=1):
    """Shift-sum refocus on the lenslet grid (Z x Ny x Nx).
    Shifts are in lenslet-grid pixels, from compute_depth_shift()."""
    L, _, Ny, Nx = views.shape
    vl = _view_list(geom)
    stack = np.zeros((len(z_values), Ny, Nx), dtype=np.float32)
    valid0 = np.ones((Ny, Nx), dtype=np.float32)

    for iz, z in enumerate(z_values):
        acc = np.zeros((Ny, Nx), dtype=np.float32)
        norm = np.zeros((Ny, Nx), dtype=np.float32)
        for iv, iu, du, dv, w in vl:
            dx, dy = compute_depth_shift(z, du, dv, geom["kx"], geom["ky"], geom["depth_curve"])
            acc += w * ndi_shift(views[iv, iu], shift=(dy, dx), order=order,
                                 mode="constant", cval=0.0, prefilter=False)
            norm += w * ndi_shift(valid0, shift=(dy, dx), order=1,
                                  mode="constant", cval=0.0, prefilter=False)
        stack[iz] = acc / np.maximum(norm, 1e-6)
        stack[iz][norm < 0.25 * norm.max()] = 0.0   # too few views -> unreliable

    return np.clip(stack, 0, None).astype(np.float32)


def focus_score(stack):
    p99 = np.percentile(stack, 99, axis=(1, 2))
    p99s = gaussian_filter1d(p99, sigma=1.0)
    peak = np.max(p99s)
    base = np.percentile(p99s, 10)
    half = base + 0.5 * (peak - base)
    width = np.sum(p99s >= half)
    return float(peak / (width + 1e-6)), p99s


def auto_tune_parallax(views, z_values, geom, interp_order=1,
                       candidates=(0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0)):
    """Scan a multiplier on the physical parallax gain. Only meaningful on
    sparse samples (beads); on dense tissue keep the physical value (1.0)."""
    best = None
    print("[AUTO] tuning parallax multiplier")
    for ss in candidates:
        g = dict(geom, kx=geom["kx"] * ss, ky=geom["ky"] * ss)
        stack = refocus_shift_sum(views, z_values, g, order=interp_order)
        score, p99 = focus_score(stack)
        print(f"[AUTO] multiplier={ss:.3f}, score={score:.6g}, peak_z={np.argmax(p99)+1}")
        if best is None or score > best["score"]:
            best = {"multiplier": ss, "score": score, "stack": stack, "geom": g}
    print(f"[AUTO] BEST multiplier={best['multiplier']:.3f}")
    return best


def highres_shape(grid_info, target_size):
    """Every saved stack is target_size x target_size in x, y."""
    return int(target_size), int(target_size)


def upscale_stack(stack, hw):
    """Bicubic upscale of a lenslet-grid stack (Z x Ny x Nx) for saving/viewing.
    Interpolation only: it adds pixels, not resolution."""
    from scipy.ndimage import zoom
    z, h, w = stack.shape
    out = zoom(stack.astype(np.float32), (1, hw[0] / h, hw[1] / w), order=3, mode="nearest")
    return np.clip(out[:, :hw[0], :hw[1]], 0, None).astype(np.float32)


def save_stack_pair(out_dir, name, stack, hw):
    """Save the upscaled stack as <name>_*.tif and the native one in native_lenslet_grid/."""
    nat = out_dir / "native_lenslet_grid"
    nat.mkdir(exist_ok=True)
    tiff.imwrite(nat / f"{name}_float32.tif", stack.astype(np.float32), imagej=True)
    up = stack if stack.shape[1:] == tuple(hw) else upscale_stack(stack, hw)
    save_fullres_stack_outputs(out_dir, name, up)
    return up


def highres_refocus_backprojection(views, z_values, grid_info, geom,
                                   target_size=512, output_sigma=0.0):
    """
    Dense shift-and-add: every dense output pixel GATHERS (bilinear) from each
    view at its own sub-lenslet shifted position.

    The old version SCATTERED 60x60 samples into a 512x512 grid with a 2-pixel
    footprint. Samples are ~8.5 dense pixels apart, so most output pixels got
    no data at all: the result was a grid of dots, not an image.

    Away from z = 0 the per-view shifts are fractional lenslets, so the views
    interleave and this really samples finer than the lenslet pitch. At z = 0
    all shifts are zero and the plane is just the interpolated lenslet image.
    """
    L, _, Ny, Nx = views.shape
    Ht, Wt = highres_shape(grid_info, target_size)
    gy = np.linspace(0, Ny - 1, Ht, dtype=np.float32)
    gx = np.linspace(0, Nx - 1, Wt, dtype=np.float32)
    GY, GX = np.meshgrid(gy, gx, indexing="ij")
    vl = _view_list(geom)

    stack = np.zeros((len(z_values), Ht, Wt), dtype=np.float32)
    for iz, z in enumerate(z_values):
        acc = np.zeros((Ht, Wt), dtype=np.float32)
        norm = np.zeros((Ht, Wt), dtype=np.float32)
        for iv, iu, du, dv, w in vl:
            dx, dy = compute_depth_shift(z, du, dv, geom["kx"], geom["ky"], geom["depth_curve"])
            yy = GY - dy
            xx = GX - dx
            inside = ((yy >= 0) & (yy <= Ny - 1) & (xx >= 0) & (xx <= Nx - 1)).astype(np.float32)
            acc += w * inside * map_coordinates(views[iv, iu], [yy, xx], order=1, mode="nearest")
            norm += w * inside
        plane = acc / np.maximum(norm, 1e-6)
        plane[norm < 0.25 * norm.max()] = 0.0
        if output_sigma and output_sigma > 0:
            plane = gaussian_filter(plane, sigma=float(output_sigma))
        stack[iz] = np.clip(plane, 0, None)

    coord_info = {
        "target_h": Ht, "target_w": Wt,
        "dense_px_per_lenslet_x": (Wt - 1) / max(Nx - 1, 1),
        "dense_px_per_lenslet_y": (Ht - 1) / max(Ny - 1, 1),
    }
    return stack, coord_info


def save_fullres_stack_outputs(out_dir, name, stack, save_uint16=True):
    """Save ZYX ImageJ/FIJI-readable full-resolution stack."""
    tiff.imwrite(
        out_dir / f"{name}_float32.tif",
        stack.astype(np.float32),
        imagej=True,
        metadata={"axes": "ZYX"},
    )
    if save_uint16:
        tiff.imwrite(
            out_dir / f"{name}_uint16.tif",
            to_uint16_auto(stack),
            imagej=True,
            metadata={"axes": "ZYX"},
        )




# =============================================================================
# High-resolution RL reconstruction and PSF diagnostics
# =============================================================================

def resize_stack_torch_np(stack, target_h, target_w, mode="bilinear", device_name="cpu"):
    """Resize only for RL initialization/diagnostic output, not as final reconstruction."""
    if torch is None:
        # fallback; only used if torch unavailable
        from scipy.ndimage import zoom
        z, h, w = stack.shape
        return zoom(stack, (1, target_h / h, target_w / w), order=1).astype(np.float32)

    device = torch.device(device_name if (device_name == "cpu" or torch.cuda.is_available()) else "cpu")
    x = torch.from_numpy(stack.astype(np.float32))[:, None].to(device)
    with torch.no_grad():
        y = F.interpolate(x, size=(int(target_h), int(target_w)), mode=mode, align_corners=False)
    return y[:, 0].detach().cpu().numpy().astype(np.float32)


def make_effective_psf_stack_512(
    z_values,
    target_size,
    wavelength_um,
    na,
    pixel_size_um,
    magnification,
    sigma0_extra,
    sigma_zR,
):
    """
    Saveable effective axial PSF diagnostic stack.

    This is not the giant per-view operator. It is the central-view effective
    lateral PSF width used by the physics model, expanded over z as a 512x512 stack.
    """
    target_size = int(target_size)
    eff_pixel_um = pixel_size_um / max(magnification, 1e-8)
    airy_sigma_um = 0.21 * wavelength_um / max(na, 1e-6)
    airy_sigma_px_lowres = airy_sigma_um / max(eff_pixel_um, 1e-8)
    base_sigma_lowres = max(float(airy_sigma_px_lowres), 0.65) + float(sigma0_extra)

    # The low-res object grid is not sensor pixels. This PSF stack is diagnostic,
    # so keep a conservative visible PSF instead of an unrealistically tiny dot.
    base_sigma_hr = max(1.5, base_sigma_lowres * 2.0)

    y = np.arange(target_size, dtype=np.float32) - (target_size - 1) / 2
    x = np.arange(target_size, dtype=np.float32) - (target_size - 1) / 2
    yy, xx = np.meshgrid(y, x, indexing="ij")
    r2 = yy * yy + xx * xx

    out = np.zeros((len(z_values), target_size, target_size), dtype=np.float32)
    for iz, z in enumerate(z_values):
        defocus = math.sqrt(1.0 + (float(z) / max(float(sigma_zR), 1e-6)) ** 2)
        sigma = max(1.0, base_sigma_hr * defocus)
        psf = np.exp(-r2 / (2 * sigma * sigma))
        psf /= np.sum(psf) + 1e-8
        out[iz] = psf.astype(np.float32)

    return out


def build_physics_otf(geom, z_values, image_hw, sigma0, sigma_ang_px, device):
    """
    OTF[a, z] for every in-aperture view a and depth z, on the LENSLET grid.

    All lengths here are lenslet-grid pixels (one pixel = one lenslet =
    lens_pitch / M in the sample, 3.75 um for 75 um / 20x):

      shift  c(z, a) = +z * d_a * k            (opposite sign to the refocus shift)
      blur   sigma(z) = sqrt(sigma0^2 + (k * |z| * sigma_ang_px)^2)

    sigma0       : in-focus blur (lenslet sampling aperture), lenslet px
    sigma_ang_px : angular resolution of one view in sensor px
                   (~ lambda * F#_lenslet / pixel / 2.355); a view integrates
                   over that range of angles, which smears defocused planes.

    The objective's Airy width (0.22 um) is ~0.06 lenslet px and is negligible
    here. The previous code divided it by the *sensor* pixel (0.325 um) and
    then used the result as a blur in *lenslet* pixels: 11.5x too wide.
    """
    H, W = image_hw
    Hp, Wp = 2 * H, 2 * W

    fy = torch.fft.fftfreq(Hp, device=device)[:, None]
    fx = torch.fft.rfftfreq(Wp, device=device)[None, :]
    f2 = fy * fy + fx * fx

    ze = torch.tensor([effective_z(z, geom["depth_curve"]) for z in z_values],
                      dtype=torch.float32, device=device)
    kbar = 0.5 * (abs(geom["kx"]) + abs(geom["ky"]))
    sigma = torch.sqrt(float(sigma0) ** 2 + (kbar * ze.abs() * float(sigma_ang_px)) ** 2)
    mtf = torch.exp(-2.0 * math.pi ** 2 * sigma[:, None, None] ** 2 * f2[None])

    otfs = []
    for iv, iu, du, dv, w in _view_list(geom):
        cx = ze * du * geom["kx"]
        cy = ze * dv * geom["ky"]
        phase = torch.exp(-2j * math.pi * (fy[None] * cy[:, None, None] + fx[None] * cx[:, None, None]))
        otfs.append(mtf * phase)

    otf = torch.stack(otfs, dim=0)      # A x Z x Hp x (Wp/2+1)
    return otf, torch.conj(otf)


def forward_model(vol, otf, image_hw):
    Z, H, W = vol.shape
    Hp = otf.shape[-2]
    Wp = (otf.shape[-1] - 1) * 2

    pad = torch.zeros((Z, Hp, Wp), dtype=vol.dtype, device=vol.device)
    pad[:, :H, :W] = vol

    V = torch.fft.rfft2(pad, dim=(-2, -1))
    conv = torch.fft.irfft2(otf * V[None], s=(Hp, Wp), dim=(-2, -1))

    return torch.sum(conv, dim=1)[:, :H, :W]


def backproject_model(ratio, otf_conj, image_hw):
    A, H, W = ratio.shape
    Hp = otf_conj.shape[-2]
    Wp = (otf_conj.shape[-1] - 1) * 2

    pad = torch.zeros((A, Hp, Wp), dtype=ratio.dtype, device=ratio.device)
    pad[:, :H, :W] = ratio

    R = torch.fft.rfft2(pad, dim=(-2, -1))
    bp = torch.fft.irfft2(R[:, None] * otf_conj, s=(Hp, Wp), dim=(-2, -1))

    return torch.sum(bp, dim=0)[:, :H, :W]


def axial_smooth(vol, strength):
    if strength <= 0 or vol.shape[0] < 3:
        return vol
    out = vol.clone()
    avg = (vol[:-2] + vol[1:-1] + vol[2:]) / 3
    out[1:-1] = (1 - strength) * vol[1:-1] + strength * avg
    return out


def xy_smooth(vol, strength):
    if strength <= 0:
        return vol
    kernel = torch.tensor(
        [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
        dtype=vol.dtype,
        device=vol.device,
    ) / 16.0
    sm = F.conv2d(vol[:, None], kernel[None, None], padding=1)[:, 0]
    return (1 - strength) * vol + strength * sm


def run_physics_rl(
    views,
    geom,
    z_values,
    init_stack,
    sigma0,
    sigma_ang_px,
    n_iter,
    device_name,
    damping,
    update_clip,
    ratio_clip,
    axial_smooth_strength,
    xy_smooth_strength,
    highres_shape_hw=None,
    highres_init_sigma=0.6,
    tag="RL-physics",
):
    """
    Multi-view Richardson-Lucy on the lenslet grid.

    If highres_shape_hw is given the unknown is stored on that dense grid, but
    the forward model still goes through an area-downsample to the lenslet
    grid. That mode is a smooth *display* of the lenslet-grid solution, not
    extra resolution.
    """
    if torch is None:
        raise ImportError("PyTorch not available.")

    device = torch.device(device_name if (device_name == "cpu" or torch.cuda.is_available()) else "cpu")

    Ny, Nx = views.shape[2], views.shape[3]
    obs_np = views[geom["mask"]].astype(np.float32)       # only in-aperture views
    A = obs_np.shape[0]
    obs_np = np.clip(obs_np, 0, None)
    obs_np = obs_np / (np.mean(obs_np) + 1e-8)
    obs = torch.from_numpy(obs_np).to(device)

    hr = highres_shape_hw is not None
    print(f"[{tag}] views={A}, Z={len(z_values)}, lenslet grid={Ny}x{Nx}, "
          f"dense={highres_shape_hw if hr else None}, device={device}")

    otf, otf_conj = build_physics_otf(geom, z_values, (Ny, Nx), sigma0, sigma_ang_px, device)

    sens = backproject_model(torch.ones_like(obs), otf_conj, (Ny, Nx))
    sens_floor = float(np.percentile(sens.detach().cpu().numpy(), 1.0))
    sens = torch.clamp(sens, min=max(sens_floor, 1e-6))

    init = np.clip(init_stack.astype(np.float32), 1e-6, None)
    if hr:
        init = resize_stack_torch_np(init, highres_shape_hw[0], highres_shape_hw[1],
                                     mode="bilinear", device_name=device_name)
        if highres_init_sigma and highres_init_sigma > 0:
            init = gaussian_filter(init, sigma=(0, highres_init_sigma, highres_init_sigma))
        init = np.clip(init, 1e-6, None)
    vol = torch.from_numpy(init).to(device)

    def to_low(v):
        return F.interpolate(v[:, None], size=(Ny, Nx), mode="area")[:, 0] if hr else v

    def to_vol(c):
        if not hr:
            return c
        return F.interpolate(c[:, None], size=tuple(highres_shape_hw),
                             mode="bilinear", align_corners=False)[:, 0]

    # Scale the start so the forward projection carries the same total signal
    # as the data. Before, the volume had mean 1 on EVERY plane, so the
    # prediction was ~Z times too bright and the clipped update could never
    # catch up: the "MSE decrease" was that scale error slowly draining.
    pred0 = forward_model(to_low(vol), otf, (Ny, Nx))
    vol = vol * (obs.sum() / (pred0.sum() + 1e-8))

    best_vol = vol.clone()
    best_mse = float(torch.mean((forward_model(to_low(vol), otf, (Ny, Nx)) - obs) ** 2))
    hist = {"iter": [0], "mse": [best_mse], "mae": [], "vol_max": [],
            "sens_floor": sens_floor, "n_views": int(A),
            "mode": "dense_display" if hr else "lenslet_grid"}
    print(f"[{tag}] start mse={best_mse:.6g}")

    t0 = time.time()
    for it in range(1, int(n_iter) + 1):
        pred = torch.clamp(forward_model(to_low(vol), otf, (Ny, Nx)), min=1e-6)
        ratio = torch.clamp(obs / pred, 0.0, ratio_clip)
        corr = to_vol(backproject_model(ratio, otf_conj, (Ny, Nx)) / sens)

        update = 1.0 + damping * (corr - 1.0)
        update = torch.clamp(update, update_clip[0], update_clip[1])
        vol = torch.clamp(vol * update, min=1e-8)

        vol = axial_smooth(vol, axial_smooth_strength)
        vol = xy_smooth(vol, xy_smooth_strength)

        pred2 = forward_model(to_low(vol), otf, (Ny, Nx))
        mse = torch.mean((pred2 - obs) ** 2).item()
        mae = torch.mean(torch.abs(pred2 - obs)).item()
        vmax = torch.max(vol).item()
        hist["iter"].append(it)
        hist["mse"].append(float(mse))
        hist["mae"].append(float(mae))
        hist["vol_max"].append(float(vmax))
        print(f"[{tag}] iter {it:03d}/{n_iter} mse={mse:.6g} mae={mae:.6g} max={vmax:.5g}")

        if not np.isfinite(mse):
            print(f"[{tag}] NaN detected; reverting best.")
            break
        if mse < best_mse:
            best_mse = mse
            best_vol = vol.clone()
        elif it >= 3 and mse > best_mse * 1.02:
            print(f"[{tag}] early stop; mse increased.")
            break

    hist["elapsed_sec"] = time.time() - t0
    hist["best_mse"] = float(best_mse)

    out = best_vol.detach().cpu().numpy().astype(np.float32)
    return np.clip(np.nan_to_num(out, nan=0, posinf=0, neginf=0), 0, None), hist


# =============================================================================
# Output figures / artifact correction
# =============================================================================

def auto_detect_problem_z(vol):
    p99 = np.percentile(vol, 99, axis=(1, 2))
    mean = np.mean(vol, axis=(1, 2))
    score = p99 / (np.max(p99) + 1e-8) + 0.25 * mean / (np.max(mean) + 1e-8)
    return int(np.argmax(gaussian_filter1d(score, sigma=1.0)))


def mild_correct(vol):
    out = vol.astype(np.float32).copy()
    z0 = auto_detect_problem_z(out)

    p80 = np.asarray([np.percentile(out[k], 80) for k in range(out.shape[0])], dtype=np.float32)
    smooth = gaussian_filter1d(p80, sigma=3)
    gain = smooth / (p80 + 1e-8)
    gain = 0.85 + 0.15 * gain
    gain = np.clip(gain, 0.85, 1.15)

    for k in range(out.shape[0]):
        out[k] *= gain[k]

    return np.clip(out, 0, None), z0, {"gain": gain.tolist()}


def per_z_stats(vol):
    return {
        "mean": np.mean(vol, axis=(1, 2)),
        "p99": np.percentile(vol, 99, axis=(1, 2)),
        "sum": np.sum(vol, axis=(1, 2)),
    }


def save_stack_figures(name, vol, fig_dir, z0=None, save_svg=False):
    save_img(
        np.max(vol, axis=0),
        fig_dir / f"{name}_mip.png",
        title=f"{name} MIP",
        save_svg=save_svg,
    )

    # IMPORTANT FIX:
    # Use orthogonal MIP, not one center row/column.
    # Center-line XZ/YZ is misleading for sparse Meta-LFM samples.
    xz = np.max(vol, axis=1)   # Z × X, max over Y
    yz = np.max(vol, axis=2)   # Z × Y, max over X

    for plane, im in [("xz", xz), ("yz", yz)]:
        plt.figure(figsize=(10, 4))

        vmin, vmax = np.percentile(im, [1, 99.7])
        plt.imshow(
            im,
            cmap="magma",
            aspect="auto",
            vmin=vmin,
            vmax=vmax,
        )

        if z0 is not None:
            plt.axhline(z0, linestyle="--", linewidth=1)

        plt.title(f"{name} {plane.upper()} orthogonal MIP")
        plt.xlabel("x/y")
        plt.ylabel("z index")
        plt.tight_layout()

        out = fig_dir / f"{name}_{plane}.png"
        plt.savefig(out, dpi=220)
        if save_svg:
            plt.savefig(out.with_suffix(".svg"))
        plt.close()


def save_comparison(out_dir, shift_stack, rl_stack=None, corr_stack=None, z0=None, save_svg=False):
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    save_stack_figures("shift_sum", shift_stack, fig_dir, z0=z0, save_svg=save_svg)

    vols = [("Shift-sum", shift_stack)]

    if rl_stack is not None:
        save_stack_figures("rl_physics", rl_stack, fig_dir, z0=z0, save_svg=save_svg)
        vols.append(("Physics RL", rl_stack))

    if corr_stack is not None:
        save_stack_figures("corrected", corr_stack, fig_dir, z0=z0, save_svg=save_svg)
        vols.append(("Corrected", corr_stack))

    plt.figure(figsize=(5 * len(vols), 5))
    for i, (title, vol) in enumerate(vols):
        mip = np.max(vol, axis=0)
        vmin, vmax = np.percentile(mip, [0.5, 99.7])
        plt.subplot(1, len(vols), i + 1)
        plt.imshow(mip, cmap="gray", vmin=vmin, vmax=vmax)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    out = fig_dir / "mip_comparison_per_panel_norm.png"
    plt.savefig(out, dpi=220)
    if save_svg:
        plt.savefig(out.with_suffix(".svg"))
    plt.close()

    plt.figure(figsize=(8, 4.5))
    z_axis = np.arange(1, shift_stack.shape[0] + 1)

    for label, vol in vols:
        s = per_z_stats(vol)
        plt.plot(z_axis, s["mean"], label=f"{label} mean")
        plt.plot(z_axis, s["p99"], label=f"{label} p99")

    if z0 is not None:
        plt.axvline(z0 + 1, linestyle="--")

    plt.xlabel("z slice (1-based)")
    plt.ylabel("intensity")
    plt.title("Per-z intensity statistics")
    plt.legend(frameon=False)
    plt.tight_layout()
    out = fig_dir / "z_profile.png"
    plt.savefig(out, dpi=220)
    if save_svg:
        plt.savefig(out.with_suffix(".svg"))
    plt.close()


def save_per_z_csv(out_dir, shift_stack, rl_stack=None, corr_stack=None):
    mdir = out_dir / "metrics"
    mdir.mkdir(exist_ok=True)

    vols = [("shift", shift_stack)]
    if rl_stack is not None:
        vols.append(("rl", rl_stack))
    if corr_stack is not None:
        vols.append(("corr", corr_stack))

    stats = [(name, per_z_stats(vol)) for name, vol in vols]

    with open(mdir / "per_z_stats.csv", "w", encoding="utf-8") as f:
        header = ["z", "z_1based"]
        for name, _ in stats:
            header += [f"{name}_mean", f"{name}_p99", f"{name}_sum"]
        f.write(",".join(header) + "\n")

        for k in range(shift_stack.shape[0]):
            row = [str(k), str(k + 1)]
            for _, s in stats:
                row += [
                    f"{s['mean'][k]:.8g}",
                    f"{s['p99'][k]:.8g}",
                    f"{s['sum'][k]:.8g}",
                ]
            f.write(",".join(row) + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Meta-LFM reconstruction. All parameters are fixed in CONFIG.")
    ap.add_argument("image", help="raw Meta-LFM image (TIFF preferred; PNG/JPG accepted)")
    ap.add_argument("out", nargs="?", default=None,
                    help="output folder (default: <image name>_recon next to the image)")
    cli = ap.parse_args()
    args = SimpleNamespace(**CONFIG)
    args.image = cli.image
    args.out = cli.out or str(Path(cli.image).with_suffix("")) + "_recon"
    if torch is None:
        print("[WARN] PyTorch not installed: Richardson-Lucy steps are skipped.")
        args.run_rl = args.run_highres_rl = False
    if args.clip_update_high <= 1.0:
        raise SystemExit("--clip_update_high must be > 1, otherwise RL can only darken the volume.")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)
    (out_dir / "metrics").mkdir(exist_ok=True)

    img_raw = ensure_2d(imread_tiff_safe(Path(args.image)))
    img = robust_norm(img_raw)
    print(f"[INPUT] image shape={img.shape}, dtype={img_raw.dtype}")

    # ---------------- grid ----------------
    if args.pitch is None:
        px0, py0 = estimate_pitch_xy(img, args.pitch_min, args.pitch_max)
        print(f"[GRID] FFT pitch_x={px0:.4f}, pitch_y={py0:.4f} px")
    else:
        px0 = float(args.pitch)
        py0 = float(args.pitch_y) if args.pitch_y is not None else px0
        print(f"[GRID] user pitch_x={px0:.4f}, pitch_y={py0:.4f} px")

    nominal = args.lens_pitch_um / args.pixel_size_um
    if abs(px0 / nominal - 1) > 0.05:
        print(f"[GRID] WARNING: pitch {px0:.2f} px differs >5% from nominal {nominal:.2f} px "
              f"(lens_pitch/pixel). Check --pitch or the optics arguments.")

    ox0, oy0 = estimate_origin_xy(img, px0, py0)
    if args.origin_x is not None:
        ox0 = float(args.origin_x)
    if args.origin_y is not None:
        oy0 = float(args.origin_y)
    print(f"[GRID] origin_x={ox0:.4f}, origin_y={oy0:.4f}")

    ex = dict(lenslet_samples=args.lenslet_samples, extraction_scale=args.extraction_scale)

    save_img(img, out_dir / "00_input.png", title="Input MetaLFM", save_svg=args.save_svg)

    if args.autocal:
        best, cal_history = auto_calibrate_grid(
            img, px0, py0, ox0, oy0,
            crop_lenslets=args.cal_crop_lenslets,
            pitch_search=args.cal_pitch_search,
            origin_search=args.cal_origin_search,
            fine_rounds=args.cal_fine_rounds,
            **ex,
        )
        print("[CAL] BEST:", best)
    else:
        best = {"pitch_x": px0, "pitch_y": py0, "origin_x": ox0, "origin_y": oy0,
                "score": None, "details": {}}
        cal_history = []

    px, py = float(best["pitch_x"]), float(best["pitch_y"])
    ox, oy = float(best["origin_x"]), float(best["origin_y"])

    views, grid_info = extract_views(
        img, px, py, ox, oy,
        normalize_views=not args.no_view_normalize,
        aperture_threshold=args.aperture_threshold,
        **ex,
    )
    mask = np.asarray(grid_info["view_mask"], dtype=bool)
    print(f"[VIEWS] shape={views.shape} [V,U,Ny,Nx]; "
          f"{int(mask.sum())}/{mask.size} views inside the aperture")
    if mask.sum() < 9:
        raise SystemExit("Fewer than 9 usable views: check the grid overlay / --aperture_threshold.")
    edge = np.asarray(grid_info["view_rel_brightness"])[[0, -1], :].max()
    if edge > 0.6:
        print("[VIEWS] note: outermost views are still bright; "
              "--extraction_scale can be increased to use more of the aperture.")

    save_grid_overlay(img, out_dir / "02_grid_overlay_refined.png",
                      np.asarray(grid_info["centers_x"]), np.asarray(grid_info["centers_y"]),
                      save_svg=args.save_svg)
    save_view_montage(views, out_dir / "03_view_montage_refined.png", save_svg=args.save_svg)
    c = args.lenslet_samples // 2
    save_img(views[c, c], out_dir / "04_center_view_refined.png",
             title="Center sub-aperture view", save_svg=args.save_svg)

    if args.remove_common_background:
        views_proc = remove_common_background(views, strength=args.background_strength, view_mask=mask)
        print("[VIEWS] common background removed")
    else:
        views_proc = np.clip(views.astype(np.float32), 0, None)
        views_proc = views_proc / (np.mean(views_proc[mask]) + 1e-8)

    # ---------------- geometry ----------------
    k = parallax_gain(args.pixel_size_um, args.magnification,
                      args.lens_focal_um, args.lens_pitch_um, args.n_medium)
    geom = make_geometry(grid_info, k, args.shift_scale,
                         args.angular_scale_x, args.angular_scale_y, args.depth_curve)

    z_values = np.linspace(args.z_min, args.z_max, args.z_steps).astype(np.float32)
    r_ap = max(math.hypot(du, dv) for _, _, du, dv, _ in _view_list(geom))
    max_shift = abs(geom["kx"]) * r_ap * max(abs(args.z_min), abs(args.z_max))
    print(f"[GEOM] parallax gain = {k:.5f} lenslets / (px * um); aperture radius = {r_ap:.2f} px; "
          f"max shift at |z|max = {max_shift:.2f} lenslets")
    if max_shift > 0.25 * min(views.shape[2:]):
        print("[GEOM] WARNING: shifts exceed 1/4 of the field; reduce the z range.")

    f_number = args.lens_focal_um / args.lens_pitch_um
    sigma_ang_px = args.sigma_ang_px
    if sigma_ang_px is None:
        sigma_ang_px = max(args.wavelength_um * f_number / args.pixel_size_um / 2.355,
                           0.5 * abs(geom["du"][1] - geom["du"][0]))
    print(f"[GEOM] lenslet F# = {f_number:.1f}; angular blur sigma = {sigma_ang_px:.2f} px; "
          f"lenslet pixel in sample = {args.lens_pitch_um / args.magnification:.2f} um")

    if args.save_psf_stack:
        psf_stack = make_effective_psf_stack_512(
            z_values=z_values, target_size=args.target_size,
            wavelength_um=args.wavelength_um, na=args.na,
            pixel_size_um=args.pixel_size_um, magnification=args.magnification,
            sigma0_extra=args.sigma0, sigma_zR=args.sigma_zR,
        )
        save_fullres_stack_outputs(out_dir, f"06_effective_psf_{args.target_size}x{args.target_size}", psf_stack)
        save_stack_figures("effective_psf", psf_stack, out_dir / "figures",
                           z0=len(z_values)//2, save_svg=args.save_svg)

    # ---------------- shift-sum ----------------
    if args.auto_shift_scale:
        tuned = auto_tune_parallax(views_proc, z_values, geom, args.interp_order)
        geom, shift_stack = tuned["geom"], tuned["stack"]
    else:
        shift_stack = refocus_shift_sum(views_proc, z_values, geom, order=args.interp_order)

    hr_hw = highres_shape(grid_info, args.target_size)
    shift_up = save_stack_pair(out_dir, "05_shift_sum_refocus", shift_stack, hr_hw)
    highres_shift_stack = None
    if args.run_highres:
        print(f"[HIGHRES] dense shift-and-add to {hr_hw[0]}x{hr_hw[1]}")
        highres_shift_stack, highres_coord_info = highres_refocus_backprojection(
            views_proc, z_values, grid_info, geom,
            target_size=args.target_size, output_sigma=args.highres_sigma,
        )
        save_fullres_stack_outputs(out_dir, "05_highres_direct_refocus", highres_shift_stack)
        save_stack_figures("highres_direct_refocus", highres_shift_stack, out_dir / "figures",
                           z0=auto_detect_problem_z(highres_shift_stack), save_svg=args.save_svg)
        with open(out_dir / "metrics" / "highres_geometry.json", "w", encoding="utf-8") as f:
            json.dump(highres_coord_info, f, indent=2)

    # ---------------- RL ----------------
    rl_stack = None
    corr_stack = None
    rl_up = None
    corr_up = None
    z0 = auto_detect_problem_z(shift_stack)

    rl_kw = dict(
        views=views_proc, geom=geom, z_values=z_values, init_stack=shift_stack,
        sigma0=args.sigma0, sigma_ang_px=sigma_ang_px, device_name=args.device,
        damping=args.damping, update_clip=(args.clip_update_low, args.clip_update_high),
        ratio_clip=args.ratio_clip,
        axial_smooth_strength=args.axial_smooth_strength,
        xy_smooth_strength=args.xy_smooth_strength,
    )

    if args.run_rl:
        rl_stack, hist = run_physics_rl(n_iter=args.rl_iter, **rl_kw)
        rl_up = save_stack_pair(out_dir, "06_physics_rl_best", rl_stack, hr_hw)
        with open(out_dir / "metrics" / "rl_history.json", "w", encoding="utf-8") as f:
            json.dump(hist, f, indent=2)

        if not args.skip_correction:
            corr_stack, z0, art_info = mild_correct(rl_stack)
            corr_up = save_stack_pair(out_dir, "07_physics_rl_corrected", corr_stack, hr_hw)
            with open(out_dir / "metrics" / "artifact_info.json", "w", encoding="utf-8") as f:
                json.dump({"problem_z_0based": z0, "problem_z_1based": z0 + 1,
                           "correction": art_info}, f, indent=2)

    if args.run_highres_rl:
        highres_rl_stack, highres_hist = run_physics_rl(
            n_iter=args.highres_rl_iter, highres_shape_hw=hr_hw,
            highres_init_sigma=args.highres_init_sigma, tag="RL-dense", **rl_kw)
        save_fullres_stack_outputs(out_dir, "08_highres_physics_rl", highres_rl_stack)
        save_stack_figures("highres_physics_rl", highres_rl_stack, out_dir / "figures",
                           z0=auto_detect_problem_z(highres_rl_stack), save_svg=args.save_svg)
        with open(out_dir / "metrics" / "highres_rl_history.json", "w", encoding="utf-8") as f:
            json.dump(highres_hist, f, indent=2)

    save_comparison(out_dir, shift_stack=shift_up, rl_stack=rl_up,
                    corr_stack=corr_up, z0=z0, save_svg=args.save_svg)
    save_per_z_csv(out_dir, shift_stack, rl_stack=rl_stack, corr_stack=corr_stack)

    with open(out_dir / "metrics" / "grid_params_refined.json", "w", encoding="utf-8") as f:
        json.dump(grid_info, f, indent=2)
    with open(out_dir / "metrics" / "calibration_report.json", "w", encoding="utf-8") as f:
        json.dump({"best": best, "history": cal_history}, f, indent=2)
    with open(out_dir / "metrics" / "run_params.json", "w", encoding="utf-8") as f:
        json.dump(dict(vars(args), parallax_gain=k, kx=geom["kx"], ky=geom["ky"],
                       sigma_ang_px=sigma_ang_px, z_units="um",
                       z_values=z_values.tolist()), f, indent=2)

    print(f"[DONE] outputs in {out_dir}")


if __name__ == "__main__":
    main()
