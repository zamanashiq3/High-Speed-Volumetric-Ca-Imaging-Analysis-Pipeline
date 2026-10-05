#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metalfm_calcium.py

Dynamic Meta-LFM calcium pipeline: raw light-field movie -> 4-D volume -> metrics.

    python metalfm_calcium.py movie.tif                 # draw ROIs by hand
    python metalfm_calcium.py movie.tif --auto          # automatic ROIs
    python metalfm_calcium.py movie.tif --rois rois.csv # reuse saved ROIs
    python metalfm_calcium.py folder_of_movies --auto   # every .tif in a folder

The reconstruction is the FROZEN metalfm_reconstruct.py (imported, never edited;
its SHA-256 is checked and written to the run metadata). This file only adds
what a movie needs:

  * grid, aperture mask, view gains and intensity scale are measured ONCE on
    the time-averaged frame and then held fixed, so every frame goes through
    exactly the same operator (per-frame normalisation would erase dF/F);
  * every frame is reconstructed with the same fixed number of RL iterations;
  * ROI traces, dF/F, peak detection and metrics follow
    processing_image_cortical.py, applied at every depth plane.

Three signal sources are analysed side by side ("method" column):
    lenslet2d  - sum over the aperture, no depth (wide-field equivalent)
    shift_sum  - linear refocus stack
    rl         - multi-view Richardson-Lucy stack
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile as tiff
from scipy import sparse
from scipy.ndimage import maximum_filter, percentile_filter, zoom
from scipy.signal import butter, filtfilt, find_peaks, peak_widths

import metalfm_reconstruct as R          # FROZEN reconstruction (sets matplotlib Agg)
import matplotlib.pyplot as plt

torch = R.torch

FROZEN_SHA256 = "df4210a3f8c0eda4776d465025f968e90795943d2d55d4b2950fdffe37b8bb98"

# =============================================================================
# FIXED CONFIGURATION for dynamics (optics / views / RL come from R.CONFIG)
# =============================================================================

DYN = dict(
    fps=20.0,                   # volume rate, Hz
    # depth planes for the movie: 5 um steps keep a long movie tractable
    z_min=-30.0, z_max=30.0, z_steps=13,
    rl_iter=R.CONFIG["rl_iter"],            # same count for every frame, no early stop
    methods=("lenslet2d", "shift_sum", "rl"),

    # dF/F  (processing_image_cortical.py)
    f0_win_s=10.0, f0_percentile=10,
    f0_floor_frac=0.05,         # F0 floored at 5% of the median ROI baseline
    # peaks (processing_image_cortical.py)
    band_lo_hz=0.1, band_hi_hz=5.0, band_order=2,
    # An event must rise >= event_thresh_sd noise SDs above the band-passed
    # baseline, in both height and prominence. Noise SD comes from frame-to-frame
    # differences, so ongoing activity does not inflate it.
    # (The original rule, prominence >= 3 MAD of the filtered trace, reported
    #  10-19 "events" per 12 s on ROIs containing only noise.)
    event_thresh_sd=3.5, min_peak_distance_s=0.25,

    # automatic ROIs
    auto_roi_size=3,            # box side in lenslets (3 = 11 um in the sample)
    auto_roi_max=40,
    auto_roi_min_sep=4,         # lenslets
    auto_roi_thresh=12.0,       # transient z-score a lenslet must reach
    auto_roi_border=2,          # lenslets ignored at the field edge

    display_size=512,           # every image written for viewing is 512 x 512
    save_4d_stacks=True,
    save_mip_movie=True,
)

plt.rcParams.update({
    "savefig.dpi": 200, "font.size": 9, "axes.linewidth": 1,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


# =============================================================================
# IO
# =============================================================================

def check_frozen():
    h = hashlib.sha256(Path(R.__file__).read_bytes()).hexdigest()
    if h != FROZEN_SHA256:
        print("[WARN] metalfm_reconstruct.py differs from the frozen version "
              f"(sha256 {h[:12]}... != {FROZEN_SHA256[:12]}...). Results are not comparable.")
    return h


def load_movie(path: Path) -> np.ndarray:
    """Raw light-field movie as (T, H, W). TIFF is memory-mapped when possible."""
    if path.suffix.lower() in (".tif", ".tiff"):
        try:
            arr = tiff.memmap(str(path))
        except Exception:
            arr = tiff.imread(str(path))
    else:
        import imageio.v2 as imageio
        arr = np.stack(imageio.mimread(str(path), memtest=False))
    if arr.ndim == 4:                       # colour movie
        arr = arr[..., :3].mean(axis=-1)
    if arr.ndim != 3 or arr.shape[0] < 2:
        raise ValueError(f"{path.name}: need a multi-frame movie, got shape {arr.shape}")
    return arr


# =============================================================================
# Fixed operator, calibrated once on the time-averaged frame
# =============================================================================

def calibrate(movie):
    C = R.CONFIG
    T = movie.shape[0]
    mean = np.zeros(movie.shape[1:], dtype=np.float64)
    for t in range(T):
        mean += movie[t]
    mean = (mean / T).astype(np.float32)

    lo, hi = np.percentile(mean, [0.1, 99.9])
    scale = float(hi - lo + 1e-8)
    img = np.clip((mean - lo) / scale, 0, None)      # no upper clip: transients must not saturate

    px, py = R.estimate_pitch_xy(img, C["pitch_min"], C["pitch_max"])
    ox, oy = R.estimate_origin_xy(img, px, py)
    views, info = R.extract_views(
        img, px, py, ox, oy,
        lenslet_samples=C["lenslet_samples"], extraction_scale=C["extraction_scale"],
        normalize_views=False, aperture_threshold=C["aperture_threshold"],
    )
    mask = np.asarray(info["view_mask"], dtype=bool)
    if mask.sum() < 9:
        raise SystemExit("Fewer than 9 usable views: this does not look like a Meta-LFM movie.")

    # view gains exactly as the frozen extract_views(normalize_views=True)
    means = views.mean(axis=(2, 3))
    gm = float(means[mask].mean())
    gain = np.zeros_like(means)
    gain[mask] = gm / means[mask]
    norm = float((views * gain[:, :, None, None])[mask].mean()) + 1e-8
    gain = gain / norm                                # mean of valid views = 1 on the mean frame

    # sparse bilinear sampler: frame.ravel() -> (A, Ny, Nx), in-aperture views only
    H, W = mean.shape
    cx = np.asarray(info["centers_x"], dtype=np.float64)
    cy = np.asarray(info["centers_y"], dtype=np.float64)
    Ny, Nx = len(cy), len(cx)
    CX, CY = np.meshgrid(cx, cy, indexing="xy")
    rows, cols, vals = [], [], []
    a = 0
    for iv, dv in enumerate(info["dv_px"]):
        for iu, du in enumerate(info["du_px"]):
            if not mask[iv, iu]:
                continue
            y = np.clip((CY + dv).ravel(), 0, H - 1)
            x = np.clip((CX + du).ravel(), 0, W - 1)
            y0 = np.minimum(np.floor(y).astype(int), H - 2)
            x0 = np.minimum(np.floor(x).astype(int), W - 2)
            wy, wx = y - y0, x - x0
            r = a * Ny * Nx + np.arange(Ny * Nx)
            g = gain[iv, iu]
            for yy, xx, w in ((y0, x0, (1 - wy) * (1 - wx)), (y0, x0 + 1, (1 - wy) * wx),
                              (y0 + 1, x0, wy * (1 - wx)), (y0 + 1, x0 + 1, wy * wx)):
                rows.append(r); cols.append(yy * W + xx); vals.append(w * g)
            a += 1
    S = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                          shape=(a * Ny * Nx, H * W), dtype=np.float32)

    k = R.parallax_gain(C["pixel_size_um"], C["magnification"],
                        C["lens_focal_um"], C["lens_pitch_um"], C["n_medium"])
    geom = R.make_geometry(info, k, C["shift_scale"], C["angular_scale_x"],
                           C["angular_scale_y"], C["depth_curve"])
    sigma_ang = C["sigma_ang_px"]
    if sigma_ang is None:
        f_number = C["lens_focal_um"] / C["lens_pitch_um"]
        sigma_ang = max(C["wavelength_um"] * f_number / C["pixel_size_um"] / 2.355,
                        0.5 * abs(geom["du"][1] - geom["du"][0]))

    print(f"[CAL] pitch=({px:.4f},{py:.4f}) px, origin=({ox:.2f},{oy:.2f}), "
          f"lenslets={Ny}x{Nx}, views={int(mask.sum())}/{mask.size}")
    return dict(lo=float(lo), scale=scale, S=S, A=a, Ny=Ny, Nx=Nx, mask=mask, info=info,
                geom=geom, sigma_ang=float(sigma_ang), mean_frame=mean, k=k)


def frame_to_views(frame, cal):
    f = np.clip((np.asarray(frame, dtype=np.float32) - cal["lo"]) / cal["scale"], 0, None)
    return (cal["S"] @ f.ravel()).reshape(cal["A"], cal["Ny"], cal["Nx"])


def views_full(obs, cal):
    L = cal["mask"].shape[0]
    v = np.zeros((L, L, cal["Ny"], cal["Nx"]), dtype=np.float32)
    v[cal["mask"]] = obs
    return v


class FixedRL:
    """Same update as the frozen run_physics_rl, with the operator built once and
    a fixed iteration count (no MSE early stop) so all frames are treated alike."""

    def __init__(self, cal, z_values, n_iter):
        C = R.CONFIG
        dev = C["device"]
        self.device = torch.device(dev if (dev == "cpu" or torch.cuda.is_available()) else "cpu")
        self.hw = (cal["Ny"], cal["Nx"])
        self.otf, self.otf_conj = R.build_physics_otf(
            cal["geom"], z_values, self.hw, C["sigma0"], cal["sigma_ang"], self.device)
        ones = torch.ones((cal["A"],) + self.hw, device=self.device)
        sens = R.backproject_model(ones, self.otf_conj, self.hw)
        floor = float(np.percentile(sens.cpu().numpy(), 1.0))
        self.sens = torch.clamp(sens, min=max(floor, 1e-6))
        self.n_iter = int(n_iter)
        self.C = C

    @torch.no_grad()
    def __call__(self, obs_np, init_np):
        C = self.C
        obs = torch.from_numpy(np.ascontiguousarray(obs_np, dtype=np.float32)).to(self.device)
        vol = torch.from_numpy(np.clip(init_np, 1e-6, None).astype(np.float32)).to(self.device)
        pred0 = R.forward_model(vol, self.otf, self.hw)
        vol = vol * (obs.sum() / (pred0.sum() + 1e-8))
        for _ in range(self.n_iter):
            pred = torch.clamp(R.forward_model(vol, self.otf, self.hw), min=1e-6)
            ratio = torch.clamp(obs / pred, 0.0, C["ratio_clip"])
            corr = R.backproject_model(ratio, self.otf_conj, self.hw) / self.sens
            upd = torch.clamp(1.0 + C["damping"] * (corr - 1.0),
                              C["clip_update_low"], C["clip_update_high"])
            vol = torch.clamp(vol * upd, min=1e-8)
            vol = R.axial_smooth(vol, C["axial_smooth_strength"])
            vol = R.xy_smooth(vol, C["xy_smooth_strength"])
        return vol.cpu().numpy()


def reconstruct_movie(movie, cal, z_values, methods):
    T = movie.shape[0]
    Ny, Nx, Z = cal["Ny"], cal["Nx"], len(z_values)
    out = {}
    if "lenslet2d" in methods:
        out["lenslet2d"] = np.zeros((T, 1, Ny, Nx), np.float32)
    if "shift_sum" in methods:
        out["shift_sum"] = np.zeros((T, Z, Ny, Nx), np.float32)
    rl = None
    if "rl" in methods:
        if torch is None:
            print("[WARN] PyTorch not installed: 'rl' method skipped.")
        else:
            out["rl"] = np.zeros((T, Z, Ny, Nx), np.float32)
            rl = FixedRL(cal, z_values, DYN["rl_iter"])
            print(f"[RL] device={rl.device}, {DYN['rl_iter']} iterations per frame")

    t0 = time.time()
    for t in range(T):
        obs = frame_to_views(movie[t], cal)
        if "lenslet2d" in out:
            out["lenslet2d"][t, 0] = obs.mean(axis=0)
        if "shift_sum" in out or rl is not None:
            ss = R.refocus_shift_sum(views_full(obs, cal), z_values, cal["geom"],
                                     order=R.CONFIG["interp_order"])
            if "shift_sum" in out:
                out["shift_sum"][t] = ss
            if rl is not None:
                out["rl"][t] = rl(obs, ss)
        if t == 0 or (t + 1) % max(1, T // 20) == 0 or t == T - 1:
            el = time.time() - t0
            print(f"[RECON] frame {t+1}/{T}  {el:.0f}s elapsed, "
                  f"~{el / (t + 1) * (T - t - 1):.0f}s left", flush=True)
    return out


# =============================================================================
# ROIs  (x0, y0, w, h) -- stored in display (512) pixels and in lenslet pixels
# =============================================================================

def to_display(img2d, size):
    up = zoom(img2d.astype(np.float32), (size / img2d.shape[0], size / img2d.shape[1]),
              order=3, mode="nearest")
    return np.clip(up[:size, :size], 0, None)


def rois_display_to_native(rois, size, Ny, Nx):
    out = []
    for x0, y0, w, h in rois:
        ax = int(np.clip(np.floor(x0 * Nx / size), 0, Nx - 1))
        ay = int(np.clip(np.floor(y0 * Ny / size), 0, Ny - 1))
        bx = int(np.clip(np.ceil((x0 + w) * Nx / size), ax + 1, Nx))
        by = int(np.clip(np.ceil((y0 + h) * Ny / size), ay + 1, Ny))
        out.append((ax, ay, bx - ax, by - ay))
    return out


def rois_native_to_display(rois, size, Ny, Nx):
    return [(x0 * size / Nx, y0 * size / Ny, w * size / Nx, h * size / Ny) for x0, y0, w, h in rois]


def pick_rois_interactive(img):
    """Box picker from processing_image_cortical.py (drag boxes, Enter to finish)."""
    from matplotlib.widgets import RectangleSelector
    ok = False
    for b in ("QtAgg", "TkAgg", "MacOSX", "GTK3Agg", "WXAgg"):
        try:
            plt.switch_backend(b); ok = True; break
        except Exception:
            continue
    if not ok:
        raise SystemExit("No interactive display available. Use --auto or --rois <csv>.")
    rois = []
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap="gray", vmin=0, vmax=np.percentile(img, 99.7))
    ax.set_title("Draw ROI boxes, press Enter when done")

    def onselect(e0, e1):
        x0, y0, x1, y1 = e0.xdata, e0.ydata, e1.xdata, e1.ydata
        w, h = abs(x1 - x0), abs(y1 - y0)
        if w < 1 or h < 1:
            return
        rois.append((min(x0, x1), min(y0, y1), w, h))
        ax.add_patch(plt.Rectangle((min(x0, x1), min(y0, y1)), w, h, fill=False, ec="r", lw=1.5))
        ax.text(min(x0, x1), min(y0, y1) - 3, str(len(rois)), color="y", fontsize=8)
        fig.canvas.draw_idle()

    rs = RectangleSelector(ax, onselect, useblit=True, interactive=True)  # noqa: F841
    fig.canvas.mpl_connect("key_press_event", lambda ev: plt.close(fig) if ev.key == "enter" else None)
    plt.show()
    plt.switch_backend("Agg")
    return rois


def auto_rois(vol4d, Ny, Nx):
    """Boxes on local maxima of a transient-detection map: for every lenslet,
    (max - median) / robust SD of the 3-frame-smoothed z-MIP time course.
    Pure noise gives ~5-8; a calcium transient gives far more."""
    from scipy.ndimage import uniform_filter, uniform_filter1d
    mip = uniform_filter1d(vol4d.max(axis=1), 3, axis=0)        # T, Ny, Nx
    mip = uniform_filter(mip, size=(1, 3, 3))                   # ROI-sized spatial average
    med = np.median(mip, axis=0)
    d = np.diff(mip, axis=0)
    sd = 1.4826 * np.median(np.abs(d - np.median(d, axis=0)), axis=0) / np.sqrt(2) + 1e-9
    act = (mip.max(axis=0) - med) / sd
    b = DYN["auto_roi_border"]
    act[:b] = act[-b:] = 0; act[:, :b] = act[:, -b:] = 0
    sep = DYN["auto_roi_min_sep"]
    peaks = (act == maximum_filter(act, size=2 * sep + 1)) & (act > DYN["auto_roi_thresh"])
    ys, xs = np.nonzero(peaks)
    order = np.argsort(act[ys, xs])[::-1][:DYN["auto_roi_max"]]
    s = DYN["auto_roi_size"]
    rois = []
    for y, x in zip(ys[order], xs[order]):
        x0 = int(np.clip(x - s // 2, 0, Nx - s)); y0 = int(np.clip(y - s // 2, 0, Ny - s))
        rois.append((x0, y0, s, s))
    return rois, act


# =============================================================================
# Traces, dF/F, peaks  (processing_image_cortical.py, applied per depth plane)
# =============================================================================

def extract_traces(vol4d, rois):
    """-> (nROI, Z, T): mean intensity in each ROI box at every depth plane."""
    return np.stack([vol4d[:, :, y0:y0 + h, x0:x0 + w].mean(axis=(2, 3)).T
                     for x0, y0, w, h in rois])


def dff_robust(traces, fs, floor):
    """traces (..., T). Rolling-percentile F0; F0 floored so near-empty voxels
    cannot produce huge dF/F. Returns dff, F0, low_baseline flag."""
    win = max(3, int(round(DYN["f0_win_s"] * fs)))
    flat = traces.reshape(-1, traces.shape[-1])
    dff = np.zeros_like(flat); f0s = np.zeros_like(flat); low = np.zeros(len(flat), bool)
    for i, y in enumerate(flat):
        f0 = percentile_filter(y, size=min(win, len(y)), percentile=DYN["f0_percentile"], mode="nearest")
        low[i] = np.median(f0) < floor
        f0 = np.clip(f0, max(floor, 1e-9), None)
        dff[i] = (y - f0) / f0; f0s[i] = f0
    return dff.reshape(traces.shape), f0s.reshape(traces.shape), low.reshape(traces.shape[:-1])


def bandpass(x, fs):
    hi = min(DYN["band_hi_hz"], 0.45 * fs)
    b, a = butter(DYN["band_order"], [DYN["band_lo_hz"], hi], btype="band", fs=fs)
    if len(x) <= 3 * max(len(a), len(b)):
        return x - np.median(x)
    return filtfilt(b, a, x)


def analyse_trace(y, fs):
    """Peak detection as in find_peaks_ca, plus the full metric set."""
    T = len(y)
    yf = bandpass(y, fs)
    d = np.diff(y)
    noise = 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(2) + 1e-9
    prom_thr = DYN["event_thresh_sd"] * noise
    peaks, props = find_peaks(yf, prominence=prom_thr, height=prom_thr,
                              distance=max(1, int(DYN["min_peak_distance_s"] * fs)))
    n = len(peaks)
    dur = T / fs
    amp = y[peaks] if n else np.array([])
    isi = np.diff(peaks) / fs if n > 1 else np.array([])
    fwhm = peak_widths(yf, peaks, rel_height=0.5)[0] / fs if n else np.array([])
    nan = np.nan
    m = {
        "n_peaks": int(n),
        "event_rate_per_min": 60.0 * n / dur,
        "Ca2+_frequency_Hz": float(1.0 / isi.mean()) if n > 1 else nan,   # 1 / mean inter-event interval
        "mean_amp": float(amp.mean()) if n else nan,
        "max_amp": float(amp.max()) if n else nan,
        "sd_amp": float(amp.std()) if n > 1 else nan,
        "mean_prominence": float(props["prominences"].mean()) if n else nan,
        "mean_fwhm_s": float(fwhm.mean()) if n else nan,
        "mean_isi_s": float(isi.mean()) if n > 1 else nan,
        "cv_isi": float(isi.std() / isi.mean()) if n > 2 else nan,
        "noise_sd": float(noise),
        "snr": float(amp.mean() / noise) if n else nan,
        "dff_max": float(y.max()),
        "dff_p95": float(np.percentile(y, 95)),
        "dff_auc": float(np.clip(y, 0, None).sum() / fs),
        "event_threshold_dff": float(prom_thr),
    }
    ev = [{"peak_number": k + 1, "frame_index": int(p), "time_s": p / fs,
           "dff_value": float(y[p]), "prominence": float(props["prominences"][k]),
           "fwhm_s": float(fwhm[k]),
           "left_base": int(props["left_bases"][k]), "right_base": int(props["right_bases"][k])}
          for k, p in enumerate(peaks)]
    return m, ev, peaks


def network_metrics(dff):
    """dff (nROI, T) at each ROI's best plane."""
    n = dff.shape[0]
    out = {"n_rois": n}
    if n < 2:
        return out, None
    cmat = np.corrcoef(dff)
    iu = np.triu_indices(n, 1)
    var_i = dff.var(axis=1).mean()
    out.update({
        "mean_pairwise_corr": float(np.nanmean(cmat[iu])),
        "median_pairwise_corr": float(np.nanmedian(cmat[iu])),
        # Golomb chi^2: 1 = perfectly synchronous, ~1/n = independent
        "synchrony_chi2": float(dff.mean(axis=0).var() / (var_i + 1e-12)),
    })
    return out, cmat


# =============================================================================
# Figures
# =============================================================================

def fig_rois(img, rois_disp, path, title):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray", vmin=0, vmax=np.percentile(img, 99.7))
    for i, (x0, y0, w, h) in enumerate(rois_disp):
        ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec="r", lw=1))
        ax.text(x0, max(0, y0 - 4), str(i + 1), color="y", fontsize=7)
    ax.set_title(title); ax.axis("off")
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)


def fig_traces(time_s, dff, peaks_list, labels, path, title):
    n = dff.shape[0]
    step = max(np.nanpercentile(dff, 99) * 0.8, 1e-3)
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * n + 1.5)))
    for i in range(n):
        ax.plot(time_s, dff[i] + i * step, lw=0.8, color="k")
        p = peaks_list[i]
        if len(p):
            ax.plot(time_s[p], dff[i][p] + i * step, "o", ms=2.5, color="tab:red")
    ax.set_yticks(np.arange(n) * step); ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("ROI (best z)   dF/F offset"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_kymograph(time_s, dff, path, title):
    fig, ax = plt.subplots(figsize=(7, 3.5))
    im = ax.imshow(dff, aspect="auto", cmap="viridis", interpolation="nearest",
                   extent=[time_s[0], time_s[-1], dff.shape[0] + 0.5, 0.5],
                   vmin=0, vmax=np.nanpercentile(dff, 99.5))
    fig.colorbar(im, label="dF/F"); ax.set_xlabel("Time (s)"); ax.set_ylabel("ROI"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_depth(mat, z_values, path, title, label):
    fig, ax = plt.subplots(figsize=(6, 3.5))
    im = ax.imshow(mat, aspect="auto", cmap="magma", interpolation="nearest",
                   extent=[z_values[0], z_values[-1], mat.shape[0] + 0.5, 0.5])
    fig.colorbar(im, label=label); ax.set_xlabel("z (um)"); ax.set_ylabel("ROI"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def fig_corr(cmat, path, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cmat, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    fig.colorbar(im, label="Pearson r"); ax.set_xlabel("ROI"); ax.set_ylabel("ROI"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


# =============================================================================
# One movie
# =============================================================================

def process_movie(path: Path, out_dir: Path, fps, roi_mode, roi_csv, force):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"; fig_dir.mkdir(exist_ok=True)
    name = path.stem
    size = DYN["display_size"]

    movie = load_movie(path)
    T = movie.shape[0]
    print(f"\n=== {path.name}: {T} frames, {movie.shape[1]}x{movie.shape[2]}, {fps} Hz ===")
    cal = calibrate(movie)
    Ny, Nx = cal["Ny"], cal["Nx"]
    z_values = np.linspace(DYN["z_min"], DYN["z_max"], DYN["z_steps"]).astype(np.float32)
    time_s = np.arange(T) / fps

    # mean volume: quick look + ROI picking image
    mean_obs = frame_to_views(cal["mean_frame"], cal)
    mean_vol = R.refocus_shift_sum(views_full(mean_obs, cal), z_values, cal["geom"])
    mean_disp = to_display(mean_vol.max(axis=0), size)
    tiff.imwrite(out_dir / f"{name}_mean_volume_{size}.tif",
                 R.upscale_stack(mean_vol, (size, size)), imagej=True, metadata={"axes": "ZYX"})

    rois_disp = None
    if roi_mode == "csv":
        df = pd.read_csv(roi_csv)
        rois_disp = [tuple(r) for r in df[["x0", "y0", "w", "h"]].to_numpy(float)]
    elif roi_mode == "interactive":
        rois_disp = pick_rois_interactive(mean_disp)
        if not rois_disp:
            raise SystemExit("No ROIs drawn.")

    # ---- reconstruct every frame (cached) ----
    cache = out_dir / f"{name}_recon4d.npz"
    if cache.exists() and not force:
        print(f"[RECON] reusing {cache.name} (use --force to recompute)")
        vols = dict(np.load(cache))
    else:
        vols = reconstruct_movie(movie, cal, z_values, DYN["methods"])
        np.savez(cache, **vols)
    methods = list(vols.keys())
    main_m = "rl" if "rl" in vols else methods[-1]

    if DYN["save_4d_stacks"]:
        for m in methods:
            if vols[m].shape[1] > 1:
                tiff.imwrite(out_dir / f"{name}_{m}_TZYX_native.tif", vols[m],
                             imagej=True, metadata={"axes": "TZYX", "finterval": 1.0 / fps})
    if DYN["save_mip_movie"]:
        mip = vols[main_m].max(axis=1)
        top = np.percentile(mip, 99.99) + 1e-9
        mov = np.stack([to_display(f, size) for f in mip])
        tiff.imwrite(out_dir / f"{name}_{main_m}_mip_movie_{size}.tif",
                     (np.clip(mov / top, 0, 1) * 65535).astype(np.uint16),
                     imagej=True, metadata={"axes": "TYX", "finterval": 1.0 / fps})

    # ---- ROIs ----
    if roi_mode == "auto":
        rois, act = auto_rois(vols[main_m], Ny, Nx)
        if not rois:
            raise SystemExit("Automatic ROI detection found nothing above threshold.")
        rois_disp = rois_native_to_display(rois, size, Ny, Nx)
        fig_rois(to_display(act, size), rois_disp, fig_dir / f"{name}_rois_on_activity.png",
                 "ROIs on temporal-activity image")
    else:
        rois = rois_display_to_native(rois_disp, size, Ny, Nx)
        rois_disp = rois_native_to_display(rois, size, Ny, Nx)      # snapped to lenslets
    um = R.CONFIG["lens_pitch_um"] / R.CONFIG["magnification"]
    roi_df = pd.DataFrame([{
        "roi_id": i + 1, "x0": d[0], "y0": d[1], "w": d[2], "h": d[3],
        "x0_lenslet": n[0], "y0_lenslet": n[1], "w_lenslet": n[2], "h_lenslet": n[3],
        "x_center_um": (n[0] + n[2] / 2) * um, "y_center_um": (n[1] + n[3] / 2) * um,
    } for i, (d, n) in enumerate(zip(rois_disp, rois))])
    roi_df.to_csv(out_dir / f"{name}_rois.csv", index=False)
    fig_rois(mean_disp, rois_disp, fig_dir / f"{name}_rois_on_mean_mip.png", "ROIs on mean volume (z-MIP)")
    print(f"[ROI] {len(rois)} ROIs ({roi_mode})")

    # ---- traces / dF/F / peaks, every method x ROI x plane ----
    metric_rows, event_rows, net_rows, long_rows = [], [], [], []
    for m in methods:
        zs = z_values if vols[m].shape[1] > 1 else np.array([np.nan], np.float32)
        raw = extract_traces(vols[m], rois)                         # nROI, Z, T
        base = np.percentile(raw, DYN["f0_percentile"], axis=-1)
        floor = DYN["f0_floor_frac"] * float(np.median(base.max(axis=1)))
        dff, f0, low = dff_robust(raw, fps, floor)
        best = np.argmax(raw.std(axis=-1), axis=1)                  # most active plane per ROI

        peak_amp = np.full(raw.shape[:2], np.nan)
        best_peaks = []
        for i in range(len(rois)):
            for j in range(raw.shape[1]):
                met, ev, pk = analyse_trace(dff[i, j], fps)
                peak_amp[i, j] = met["mean_amp"]
                row = {"file": name, "method": m, "ROI": i + 1, "z_index": j,
                       "z_um": float(zs[j]), "is_best_z": bool(j == best[i]),
                       "F0_mean": float(f0[i, j].mean()), "F_mean": float(raw[i, j].mean()),
                       "low_baseline": bool(low[i, j]), **met}
                metric_rows.append(row)
                for e in ev:
                    event_rows.append({"file": name, "method": m, "roi_id": i + 1, "z_index": j,
                                       "z_um": float(zs[j]), "is_best_z": bool(j == best[i]), **e})
                if j == best[i]:
                    best_peaks.append(pk)
                long_rows.append(pd.DataFrame({
                    "file": name, "method": m, "roi_id": i + 1, "z_um": float(zs[j]),
                    "frame": np.arange(T), "time_s": time_s,
                    "raw": raw[i, j], "dff": dff[i, j]}))

        idx = np.arange(len(rois))
        dff_b, raw_b = dff[idx, best], raw[idx, best]
        cols = [f"ROI_{i+1}" for i in idx]
        pd.DataFrame(raw_b.T, columns=cols).assign(time_s=time_s)[["time_s"] + cols] \
            .to_csv(out_dir / f"{name}_{m}_traces_raw_bestz.csv", index=False)
        pd.DataFrame(dff_b.T, columns=cols).assign(time_s=time_s)[["time_s"] + cols] \
            .to_csv(out_dir / f"{name}_{m}_traces_dff_bestz.csv", index=False)

        net, cmat = network_metrics(dff_b)
        net_rows.append({"file": name, "method": m, **net,
                         "mean_event_rate_per_min": float(np.mean(
                             [r["event_rate_per_min"] for r in metric_rows
                              if r["method"] == m and r["is_best_z"] and r["file"] == name]))})
        if cmat is not None:
            pd.DataFrame(cmat, index=cols, columns=cols).to_csv(out_dir / f"{name}_{m}_correlation.csv")
            fig_corr(cmat, fig_dir / f"{name}_{m}_correlation.png", f"{m}: dF/F correlation")

        lab = [f"{i+1}" + ("" if np.isnan(zs[best[i]]) else f" ({zs[best[i]]:+.0f} um)") for i in idx]
        fig_traces(time_s, dff_b, best_peaks, lab, fig_dir / f"{name}_{m}_traces_peaks.png",
                   f"{m}: dF/F with detected events")
        fig_kymograph(time_s, dff_b, fig_dir / f"{name}_{m}_kymograph.png", f"{m}: Ca2+ kymograph")
        if raw.shape[1] > 1:
            fig_depth(raw.std(axis=-1) / (raw.std(axis=-1).max(axis=1, keepdims=True) + 1e-12),
                      z_values, fig_dir / f"{name}_{m}_depth_profile.png",
                      f"{m}: activity vs depth", "temporal SD (norm.)")

    metrics = pd.DataFrame(metric_rows)
    events = pd.DataFrame(event_rows)
    network = pd.DataFrame(net_rows)
    metrics.to_csv(out_dir / f"{name}_metrics_all_planes.csv", index=False)
    metrics[metrics.is_best_z].to_csv(out_dir / f"{name}_metrics_best_z.csv", index=False)
    events.to_csv(out_dir / f"{name}_peaks_events_long.csv", index=False)
    network.to_csv(out_dir / f"{name}_network_metrics.csv", index=False)
    pd.concat(long_rows, ignore_index=True).to_csv(out_dir / f"{name}_traces_all_planes_long.csv", index=False)
    try:
        with pd.ExcelWriter(out_dir / f"{name}_metrics.xlsx") as xl:
            metrics[metrics.is_best_z].to_excel(xl, sheet_name="metrics_best_z", index=False)
            metrics.to_excel(xl, sheet_name="metrics_all_planes", index=False)
            events.to_excel(xl, sheet_name="events", index=False)
            network.to_excel(xl, sheet_name="network", index=False)
            roi_df.to_excel(xl, sheet_name="rois", index=False)
    except Exception as e:                                           # openpyxl missing
        print(f"[WARN] Excel export skipped: {e}")

    meta = {
        "movie": str(path), "frames": int(T), "fps": float(fps),
        "frozen_reconstruction_sha256": check_frozen(),
        "reconstruction_config": {k: v for k, v in R.CONFIG.items()},
        "dynamic_config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in DYN.items()},
        "z_values_um": z_values.tolist(),
        "pitch_px": [cal["info"]["pitch_x"], cal["info"]["pitch_y"]],
        "origin_px": [cal["info"]["origin_x"], cal["info"]["origin_y"]],
        "lenslets_yx": [Ny, Nx], "views_used": int(cal["A"]),
        "lenslet_pixel_um": um, "parallax_gain": cal["k"],
        "intensity_offset": cal["lo"], "intensity_scale": cal["scale"],
        "roi_mode": roi_mode, "n_rois": len(rois),
    }
    with open(out_dir / f"{name}_run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[DONE] {name}: outputs in {out_dir}")
    print(metrics[metrics.is_best_z].groupby("method")[
        ["n_peaks", "event_rate_per_min", "mean_amp", "snr"]].mean().round(3).to_string())
    return metrics, events, network


def main():
    ap = argparse.ArgumentParser(description="Meta-LFM dynamic calcium reconstruction + metrics.")
    ap.add_argument("input", help="raw light-field movie (.tif stack / .avi) or a folder of them")
    ap.add_argument("--fps", type=float, default=DYN["fps"])
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--auto", action="store_true", help="detect ROIs automatically")
    g.add_argument("--rois", default=None, help="CSV with x0,y0,w,h in 512-px display coordinates")
    ap.add_argument("--force", action="store_true", help="recompute even if a cached reconstruction exists")
    args = ap.parse_args()

    check_frozen()
    src = Path(args.input)
    files = sorted(p for p in src.iterdir() if p.suffix.lower() in (".tif", ".tiff", ".avi")) \
        if src.is_dir() else [src]
    if not files:
        raise SystemExit("No movies found.")
    mode = "auto" if args.auto else ("csv" if args.rois else "interactive")

    all_m, all_e, all_n = [], [], []
    for p in files:
        out = p.parent / f"{p.stem}_calcium"
        try:
            m, e, n = process_movie(p, out, args.fps, mode, args.rois, args.force)
            all_m.append(m); all_e.append(e); all_n.append(n)
        except (ValueError, SystemExit) as err:
            if len(files) == 1:
                raise
            print(f"[SKIP] {p.name}: {err}")

    if len(all_m) > 1:
        root = src if src.is_dir() else src.parent
        pd.concat(all_m).to_csv(root / "ALL_metrics_all_planes.csv", index=False)
        pd.concat(all_m).query("is_best_z").to_csv(root / "ALL_metrics_best_z.csv", index=False)
        pd.concat(all_e).to_csv(root / "ALL_peaks_events_long.csv", index=False)
        pd.concat(all_n).to_csv(root / "ALL_network_metrics.csv", index=False)
        print(f"\n[ALL] combined tables for {len(all_m)} movies written to {root}")


if __name__ == "__main__":
    main()
