"""
Functional-imaging performance metrics for MetaLFM Ca2+ recordings.

Quantifies how well a recording supports dynamic functional imaging:

  Noise and SNR
    sigma_frame     frame-to-frame noise of dF/F0 (1.4826*MAD of first differences / sqrt 2)
    nu              standardised noise level, sigma-independent of frame rate
                    (nu = median|diff dF/F0| / sqrt(fs) * 100; Rupprecht et al., 2021, Nat Neurosci),
                    so recordings at different rates can be compared
    event SNR       transient prominence / sigma of the smoothed trace
  Detection reliability
    events/min at thresholds k*sigma in neuron ROIs vs background ROIs (no cell);
    the background rate is the empirical false-positive rate
  Temporal fidelity
    10-90 % rise time, half-decay time, samples on the rising edge, event-triggered average
  Spectral SNR
    power in the Ca2+ band (0.05-1 Hz) relative to the white-noise floor (3 Hz - Nyquist)
  Spatial specificity (where the resolution / sectioning advantage shows up)
    SBR             baseline signal-to-background ratio, F0(cell) / F0(surrounding neuropil ring)
    contamination   neuropil-ring dF/F0 at the cell's event peaks / cell dF/F0 at those peaks
    r_cell_np       correlation of cell and neuropil-ring fluctuations
    pairwise r vs distance between neurons
  Common-mode (field-wide) signal
    The mean of all cell-free pixels is used as a reference. For each neuron:
    r with the reference, slope beta, and fraction of variance it explains.
    Pairwise neuron correlations are reported before and after regressing it out.
    A large shared component means part of the apparent synchrony is
    non-specific (illumination fluctuation or diffuse out-of-focus fluorescence).
    COMMON_MODE_CORRECTION = True runs event detection on corrected traces.
  Local correlation image (each pixel's correlation with its 8 neighbours)

Run standalone:
    python functional_metrics.py --movie combine.tif --rois <prefix>_rois_on_movie.csv --fps 20
or from processing_image_cortical.run_pipeline(..., functional=True).

To compare MetaLFM with conventional LFM, run both recordings and pass the two
*_functional_per_roi.csv files to compare_conditions().
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.ndimage import gaussian_filter1d, median_filter, uniform_filter1d
from scipy.signal import find_peaks, welch
from scipy.stats import mannwhitneyu

import processing_image_cortical as core

# Colours (reference categorical palette, slots 1-2, plus muted ink for the null/background)
C_CELL, C_NP, C_BG = "#2a78d6", "#eb6834", "#898781"
INK, INK2 = "#0b0b0b", "#52514e"

DETECT_K = 4.0            # event threshold in units of sigma (height and prominence)
K_SWEEP = np.arange(2.0, 10.01, 0.5)
SMOOTH_SIGMA_FRAMES = 1.0
MIN_EVENT_SEP_S = 0.5
EXCLUDE_AROUND_STEP_S = 0.75   # ignore detections this close to segment boundaries / intensity steps
CA_BAND = (0.05, 1.0)
RISE_SEARCH_S = 1.0       # look for the pre-event baseline this far before the peak
COMMON_MODE_CORRECTION = False
NOISE_BAND_LO = 3.0


# ---------------------------------------------------------------- ROI geometry
def _box_mask(shape, box, pad=0):
    x0, y0, w, h = box
    m = np.zeros(shape, bool)
    m[max(0, y0 - pad):min(shape[0], y0 + h + pad), max(0, x0 - pad):min(shape[1], x0 + w + pad)] = True
    return m


def neuropil_masks(shape, cell_rois, gap=2, width=6):
    """Ring around each cell box (gap px away, width px wide) excluding every cell box (+gap)."""
    all_cells = np.zeros(shape, bool)
    for b in cell_rois:
        all_cells |= _box_mask(shape, b, pad=gap)
    return [_box_mask(shape, b, pad=gap + width) & ~_box_mask(shape, b, pad=gap) & ~all_cells
            for b in cell_rois]


def pick_background_rois(mean_frame, cell_rois, n=10, margin=15, edge=25, min_F=0.5, seed=0):
    """Cell-free boxes (same median size as the cell ROIs) used as a null for noise/false positives.

    Excludes the image border (reconstruction edge effects) and anything within
    `margin` px of a cell ROI; requires a non-zero baseline so dF/F0 is defined.
    """
    H, W = mean_frame.shape
    bw = int(np.median([r[2] for r in cell_rois])); bh = int(np.median([r[3] for r in cell_rois]))
    blocked = np.zeros((H, W), bool)
    for b in cell_rois:
        blocked |= _box_mask((H, W), b, pad=margin)
    # also keep clear of bright structure not covered by an ROI (unlabelled cells/processes)
    blocked |= mean_frame > np.percentile(mean_frame, 97)
    rng = np.random.default_rng(seed)
    chosen, tries = [], 0
    while len(chosen) < n and tries < 20000:
        tries += 1
        x0 = int(rng.integers(edge, W - edge - bw)); y0 = int(rng.integers(edge, H - edge - bh))
        box = (x0, y0, bw, bh)
        if blocked[y0:y0 + bh, x0:x0 + bw].any() or mean_frame[y0:y0 + bh, x0:x0 + bw].mean() < min_F:
            continue
        chosen.append(box)
        blocked |= _box_mask((H, W), box, pad=4)
    return chosen


def background_area_mask(mean_frame, cell_rois, pad=25, edge=25):
    """All pixels far from any cell ROI and from bright unlabelled structure (cell-free reference)."""
    H, W = mean_frame.shape
    m = np.ones((H, W), bool)
    for b in cell_rois:
        m &= ~_box_mask((H, W), b, pad=pad)
    m &= mean_frame < np.percentile(mean_frame, 97)
    m[:edge] = m[-edge:] = False; m[:, :edge] = m[:, -edge:] = False
    return m


# ---------------------------------------------------------------- extraction (one streaming pass)
def stream_extract(movie_path, cell_rois, bg_rois, np_masks, bg_area, bin_t=4, bin_s=2):
    """Return F_cell, F_np, F_bg (n, T), F_area (T,), mean frame and a binned movie."""
    Fc, Fn, Fb, Fa, binned, acc, buf, n = [], [], [], [], [], None, [], 0
    np_idx = [np.nonzero(m) for m in np_masks]
    a_idx = np.nonzero(bg_area)
    for f in core.iter_frames(movie_path):
        Fc.append([f[y:y + h, x:x + w].mean() for x, y, w, h in cell_rois])
        Fb.append([f[y:y + h, x:x + w].mean() for x, y, w, h in bg_rois])
        Fn.append([f[i].mean() if len(i[0]) else np.nan for i in np_idx])
        Fa.append(f[a_idx].mean())
        acc = f.astype(np.float64) if acc is None else acc + f
        n += 1
        buf.append(f)
        if len(buf) == bin_t:
            b = np.mean(buf, axis=0)
            H, W = (b.shape[0] // bin_s) * bin_s, (b.shape[1] // bin_s) * bin_s
            binned.append(b[:H, :W].reshape(H // bin_s, bin_s, W // bin_s, bin_s).mean((1, 3)).astype(np.float16))
            buf = []
    return (np.asarray(Fc).T, np.asarray(Fn).T, np.asarray(Fb).T, np.asarray(Fa),
            (acc / n).astype(np.float32), np.asarray(binned))


def local_correlation_image(movie, fs_binned, hp_s=10.0):
    """Mean correlation of each pixel with its 8 neighbours after removing slow trends."""
    x = movie.astype(np.float32)
    x = x - uniform_filter1d(x, size=max(3, int(hp_s * fs_binned)), axis=0, mode="nearest")
    x /= x.std(axis=0) + 1e-6
    T, H, W = x.shape
    acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy == 0 and dx == 0:
                continue
            ys, yd = slice(max(0, dy), H + min(0, dy)), slice(max(0, -dy), H + min(0, -dy))
            xs, xd = slice(max(0, dx), W + min(0, dx)), slice(max(0, -dx), W + min(0, -dx))
            acc[yd, xd] += (x[:, yd, xd] * x[:, ys, xs]).mean(0)
            cnt[yd, xd] += 1
    return acc / cnt


# ---------------------------------------------------------------- noise, events, kinetics
def frame_noise(d):
    """sigma of white noise in dF/F0 from first differences (robust to transients and drift)."""
    dd = np.diff(d)
    return 1.4826 * np.median(np.abs(dd - np.median(dd))) / np.sqrt(2)


def nu_noise_level(d, fs):
    """Standardised noise level (% / sqrt(Hz)); Rupprecht et al. 2021."""
    return 100 * np.median(np.abs(np.diff(d))) / np.sqrt(fs)


def excluded_frames(T, fs, step_frames):
    bad = np.zeros(T, bool)
    r = int(EXCLUDE_AROUND_STEP_S * fs)
    for s in step_frames:
        bad[max(0, s - r):min(T, s + r + 1)] = True
    return bad


def detect_events(d, fs, k=DETECT_K, bad=None):
    """Events on a lightly smoothed trace; threshold k * sigma_s for both height and prominence."""
    xs = gaussian_filter1d(d, SMOOTH_SIGMA_FRAMES)
    hp = xs - median_filter(xs, size=int(2 * fs) | 1, mode="nearest")
    sig = 1.4826 * np.median(np.abs(hp - np.median(hp))) + 1e-12
    pk, pr = find_peaks(xs, height=k * sig, prominence=k * sig, distance=max(1, int(MIN_EVENT_SEP_S * fs)))
    if bad is not None and len(pk):
        keep = ~bad[pk]
        pk = pk[keep]; pr = {key: v[keep] for key, v in pr.items()}
    return pk, pr, xs, sig


def event_kinetics(xs, pk, pr, fs):
    """10-90 % rise time and half-decay time (s) for each event on the smoothed trace."""
    out = []
    w = int(RISE_SEARCH_S * fs)
    for p in pk:
        lo = max(0, p - w)
        lb = lo + int(np.argmin(xs[lo:p + 1]))        # local pre-event minimum
        base = xs[lb]; amp = xs[p] - base
        if amp <= 0 or lb == p:
            continue
        seg = xs[lb:p + 1]
        t10 = lb + np.nonzero(seg >= base + 0.1 * amp)[0][0]
        t90 = lb + np.nonzero(seg >= base + 0.9 * amp)[0][0]
        tail = xs[p:p + int(5 * fs)]
        below = np.nonzero(tail <= base + 0.5 * amp)[0]
        out.append({"rise_10_90_s": (t90 - t10) / fs,
                    "samples_on_rise": int(t90 - t10) + 1,
                    "half_decay_s": below[0] / fs if len(below) else np.nan})
    return pd.DataFrame(out)


def spectral_snr(d, fs):
    f, p = welch(d - d.mean(), fs=fs, nperseg=min(len(d), int(40 * fs)))
    band = (f >= CA_BAND[0]) & (f <= CA_BAND[1])
    noise = f >= NOISE_BAND_LO
    floor = np.median(p[noise])
    return f, p / floor, float(np.mean(p[band]) / floor)


def highpass(d, fs, s=10.0):
    return d - uniform_filter1d(d, size=int(s * fs), mode="nearest")


# ---------------------------------------------------------------- main analysis
def common_mode(d_cell, d_ref, fs):
    """Per-neuron r, beta and variance fraction explained by the cell-free reference (high-passed)."""
    X = np.vstack([highpass(d, fs) for d in d_cell]); g = highpass(d_ref, fs)
    beta = X @ g / (g @ g)
    R = X - np.outer(beta, g)
    r = np.array([np.corrcoef(x, g)[0, 1] for x in X])
    frac = 1 - R.var(1) / X.var(1)
    corrected = d_cell - np.outer(beta, d_ref - np.median(d_ref))
    return r, beta, frac, X, R, corrected


def analyze(F_cell, F_np, F_bg, fs, cell_rois, step_frames=(), F_area=None):
    T = F_cell.shape[1]
    d_cell_raw, F0_cell = core.dff0(F_cell, fs, warn=False)
    if F_area is not None:
        d_area = core.dff0(F_area[None, :], fs, warn=False)[0][0]
        cm_r, cm_beta, cm_frac, X_hp, R_hp, d_corr = common_mode(d_cell_raw, d_area, fs)
    else:
        d_area = None; cm_r = cm_beta = cm_frac = np.full(F_cell.shape[0], np.nan)
        X_hp = np.vstack([highpass(d, fs) for d in d_cell_raw]); R_hp = None; d_corr = d_cell_raw
    d_cell = d_corr if COMMON_MODE_CORRECTION else d_cell_raw
    d_np, F0_np = core.dff0(F_np, fs, warn=False)
    d_bg, _ = core.dff0(np.maximum(F_bg, 1e-3), fs, warn=False)
    bad = excluded_frames(T, fs, list(step_frames) + [a for a, _ in core.segment_bounds(T)[1:]])
    minutes = (~bad).sum() / fs / 60

    rows, ev_rows, kin_all = [], [], []
    centroids = np.array([[x + w / 2, y + h / 2] for x, y, w, h in cell_rois])
    for i in range(F_cell.shape[0]):
        d = d_cell[i]
        pk, pr, xs, sig = detect_events(d, fs, bad=bad)
        kin = event_kinetics(xs, pk, pr, fs)
        snr = pr["prominences"] / sig if len(pk) else np.array([])
        # fraction of the cell's absolute signal change that also appears in the surrounding ring
        dF_np = gaussian_filter1d(F_np[i] - F0_np[i], SMOOTH_SIGMA_FRAMES)
        dF_c = gaussian_filter1d(F_cell[i] - F0_cell[i], SMOOTH_SIGMA_FRAMES)
        contam = float(np.median(dF_np[pk] / dF_c[pk])) if len(pk) else np.nan
        pk_c, pr_c, _, sig_c = detect_events(d_corr[i], fs, bad=bad)
        snr_c = pr_c["prominences"] / sig_c if len(pk_c) else np.array([])
        _, _, ssnr = spectral_snr(d, fs)
        rows.append({
            "roi_id": i + 1, "type": "neuron",
            "F0_median": float(np.median(F0_cell[i])),
            "F0_neuropil_median": float(np.median(F0_np[i])),
            "SBR": float(np.median(F0_cell[i]) / np.median(F0_np[i])),
            "sigma_frame_dFF0": float(frame_noise(d)),
            "nu_pct_per_sqrtHz": float(nu_noise_level(d, fs)),
            "sigma_smoothed_dFF0": float(sig),
            "n_events": int(len(pk)),
            "event_rate_per_min": float(len(pk) / minutes),
            "median_event_amp_dFF0": float(np.median(pr["prominences"])) if len(pk) else np.nan,
            "median_event_SNR": float(np.median(snr)) if len(pk) else np.nan,
            "max_event_SNR": float(np.max(snr)) if len(pk) else np.nan,
            "median_rise_10_90_s": float(kin.rise_10_90_s.median()) if len(kin) else np.nan,
            "median_samples_on_rise": float(kin.samples_on_rise.median()) if len(kin) else np.nan,
            "median_half_decay_s": float(kin.half_decay_s.median()) if len(kin) else np.nan,
            "spectral_SNR": ssnr,
            "neuropil_contamination": float(contam),
            "r_cell_neuropil": float(np.corrcoef(highpass(d, fs), highpass(d_np[i], fs))[0, 1]),
            "r_common_mode": float(cm_r[i]), "beta_common_mode": float(cm_beta[i]),
            "common_mode_var_frac": float(cm_frac[i]),
            "n_events_common_removed": int(len(pk_c)),
            "median_event_SNR_common_removed": float(np.median(snr_c)) if len(snr_c) else np.nan,
        })
        for p, a, s in zip(pk, pr.get("prominences", []), snr):
            ev_rows.append({"roi_id": i + 1, "frame": int(p), "time_s": p / fs, "amp_dFF0": float(a), "SNR": float(s)})
        kin_all.append(kin.assign(roi_id=i + 1))

    bg_rows, bg_snr_all = [], []
    for j in range(F_bg.shape[0]):
        pk, pr, xs, sig = detect_events(d_bg[j], fs, bad=bad)
        _, _, ssnr = spectral_snr(d_bg[j], fs)
        bg_snr_all.extend(list(pr["prominences"] / sig) if len(pk) else [])
        bg_rows.append({"roi_id": j + 1, "type": "background",
                        "n_events": int(len(pk)), "event_rate_per_min": float(len(pk) / minutes),
                        "max_event_SNR": float(np.max(pr["prominences"] / sig)) if len(pk) else np.nan,
                        "spectral_SNR": ssnr})

    # threshold sweep: cells vs background (false-positive rate)
    sweep = []
    for k in K_SWEEP:
        for kind, D in (("neuron", d_cell), ("background", d_bg)):
            for i, d in enumerate(D):
                sweep.append({"k_sigma": k, "type": kind, "roi_id": i + 1,
                              "events_per_min": len(detect_events(d, fs, k=k, bad=bad)[0]) / minutes})

    # pairwise correlation vs distance
    C = np.corrcoef(X_hp)
    C2 = np.corrcoef(R_hp) if R_hp is not None else np.full_like(C, np.nan)
    pairs = [{"roi_a": a + 1, "roi_b": b + 1,
              "distance_px": float(np.linalg.norm(centroids[a] - centroids[b])),
              "r": float(C[a, b]), "r_common_removed": float(C2[a, b])}
             for a in range(len(C)) for b in range(a + 1, len(C))]

    return dict(per_roi=pd.DataFrame(rows), background=pd.DataFrame(bg_rows),
                events=pd.DataFrame(ev_rows), kinetics=pd.concat(kin_all, ignore_index=True) if kin_all else pd.DataFrame(),
                sweep=pd.DataFrame(sweep), pairs=pd.DataFrame(pairs),
                d_cell=d_cell, d_cell_raw=d_cell_raw, d_corr=d_corr, d_area=d_area,
                d_np=d_np, d_bg=d_bg, bad=bad, minutes=minutes, bg_event_snr=np.array(bg_snr_all))


# ---------------------------------------------------------------- figure
def _panel(ax, letter):
    ax.text(-0.16, 1.06, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", color=INK)


def plot_functional_figure(res, mean_frame, corr_img, cell_rois, bg_rois, np_masks, fs, bin_s, outpath,
                           um_per_px=None, bg_area=None):
    pr, bg, ev, sw, pairs = res["per_roi"], res["background"], res["events"], res["sweep"], res["pairs"]
    d_cell, d_bg = res["d_cell"], res["d_bg"]
    T = d_cell.shape[1]; t = np.arange(T) / fs

    fig = plt.figure(figsize=(7.2, 8.6))
    gs = fig.add_gridspec(3, 3, hspace=0.62, wspace=0.55, left=0.08, right=0.98, top=0.96, bottom=0.06)

    # a: ROIs
    ax = fig.add_subplot(gs[0, 0]); _panel(ax, "a")
    ax.imshow(mean_frame, cmap="gray", vmax=np.percentile(mean_frame, 99.7))
    np_any = np.zeros(mean_frame.shape, bool)
    for m in np_masks:
        np_any |= m
    ax.imshow(np.ma.masked_where(~np_any, np_any), cmap=plt.matplotlib.colors.ListedColormap([C_NP]), alpha=0.45)
    for i, (x, y, w, h) in enumerate(cell_rois):
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=C_CELL, lw=0.8))
        ax.text(x, y - 3, str(i + 1), color="w", fontsize=5)
    if bg_area is not None:
        ax.contour(bg_area, levels=[0.5], colors=[C_BG], linewidths=0.4)
    for x, y, w, h in bg_rois:
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="w", lw=0.6, ls=(0, (2, 1.5))))
    ax.set_title("ROIs: neuron, neuropil ring,\nbackground box, cell-free area", fontsize=6.5, color=INK)
    ax.axis("off")

    # b: local correlation image
    ax = fig.add_subplot(gs[0, 1]); _panel(ax, "b")
    im = ax.imshow(corr_img, cmap="magma", vmin=0, vmax=np.percentile(corr_img, 99.8))
    for x, y, w, h in cell_rois:
        ax.add_patch(Rectangle((x / bin_s, y / bin_s), w / bin_s, h / bin_s, fill=False, ec="w", lw=0.5))
    ax.set_title("Local correlation image", fontsize=7, color=INK); ax.axis("off")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02); cb.ax.tick_params(labelsize=6); cb.set_label("r (8 neighbours)", fontsize=6)

    # c: example traces in sigma units
    ax = fig.add_subplot(gs[0, 2]); _panel(ax, "c")
    order = pr.sort_values("max_event_SNR", ascending=False).roi_id.values
    show = [order[0], order[len(order) // 2], order[-1]]
    off = 0
    for r in show:
        z = gaussian_filter1d(d_cell[r - 1], SMOOTH_SIGMA_FRAMES) / pr.loc[pr.roi_id == r, "sigma_smoothed_dFF0"].item()
        ax.plot(t, z + off, lw=0.4, color=C_CELL); ax.text(t[-1] + 3, off, f"ROI {r}", fontsize=5.5, color=INK2, va="center")
        off += max(12, np.nanpercentile(z, 99.9) + 4)
    for j in range(min(2, d_bg.shape[0])):
        _, _, xs, sig = detect_events(d_bg[j], fs)
        ax.plot(t, xs / sig + off, lw=0.4, color=C_BG); ax.text(t[-1] + 3, off, f"bg {j+1}", fontsize=5.5, color=INK2, va="center")
        off += 12
    ax.set_xlabel("Time (s)", fontsize=7); ax.set_ylabel(r"$\Delta F/F_0$ / $\sigma$ (offset)", fontsize=7)
    ax.set_title(f"Traces in noise units (threshold {DETECT_K:g}$\\sigma$)", fontsize=7, color=INK)
    ax.tick_params(labelsize=6); ax.set_yticks([])

    # d: threshold sweep
    ax = fig.add_subplot(gs[1, 0]); _panel(ax, "d")
    for kind, col in (("neuron", C_CELL), ("background", C_BG)):
        g = sw[sw.type == kind].groupby("k_sigma").events_per_min
        med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
        ax.fill_between(med.index, lo.clip(lower=1e-2), hi.clip(lower=1e-2), color=col, alpha=0.2, lw=0)
        ax.plot(med.index, med.clip(lower=1e-2), color=col, lw=1.5, label="neurons" if kind == "neuron" else "background")
    ax.axvline(DETECT_K, color=INK2, lw=0.6, ls=":")
    ax.set_yscale("log"); ax.set_ylim(bottom=1e-2)
    ax.set_xlabel(r"Detection threshold ($\sigma$)", fontsize=7); ax.set_ylabel("Events / min", fontsize=7)
    ax.set_title("Detections vs threshold\n(background = false positives)", fontsize=7, color=INK)
    ax.legend(fontsize=6); ax.tick_params(labelsize=6)

    # e: event SNR per ROI
    ax = fig.add_subplot(gs[1, 1]); _panel(ax, "e")
    rng = np.random.default_rng(1)
    for r in pr.roi_id:
        s = ev.loc[ev.roi_id == r, "SNR"].values
        if len(s):
            ax.scatter(r + rng.uniform(-0.18, 0.18, len(s)), s, s=5, color=C_CELL, alpha=0.5, lw=0)
            ax.plot([r - 0.3, r + 0.3], [np.median(s)] * 2, color=INK, lw=1)
    bg_snr = res.get("bg_event_snr", np.array([]))
    if len(bg_snr):
        ax.scatter(rng.uniform(-0.2, 0.2, len(bg_snr)), bg_snr, s=5, color=C_BG, alpha=0.5, lw=0)
        ax.plot([-0.3, 0.3], [np.median(bg_snr)] * 2, color=INK, lw=1)
    ax.axhline(DETECT_K, color=INK2, lw=0.6, ls=":")
    ax.set_yscale("log"); ax.set_xticks([0] + list(pr.roi_id)); ax.set_xticklabels(["bg"] + [str(r) for r in pr.roi_id])
    ax.set_xlim(-0.6, len(pr) + 0.6)
    ax.set_xlabel("Neuron ROI", fontsize=7); ax.set_ylabel(r"Event SNR (prominence / $\sigma$)", fontsize=7)
    ax.set_title("Transient SNR", fontsize=7, color=INK); ax.tick_params(labelsize=6)

    # f: event-triggered average (temporal sampling)
    ax = fig.add_subplot(gs[1, 2]); _panel(ax, "f")
    w = int(2 * fs); seg = []
    top = ev.sort_values("SNR", ascending=False)
    top = top[top.SNR >= np.nanpercentile(top.SNR, 50)] if len(top) else top
    for _, e in top.iterrows():
        p = int(e.frame)
        if w <= p < T - w:
            seg.append(d_cell[int(e.roi_id) - 1, p - w:p + w + 1])
    if seg:
        seg = np.vstack(seg); m = seg.mean(0); se = seg.std(0) / np.sqrt(len(seg)); tt = np.arange(-w, w + 1) / fs
        ax.fill_between(tt, m - se, m + se, color=C_CELL, alpha=0.2, lw=0)
        ax.plot(tt, m, color=C_CELL, lw=1); ax.plot(tt, m, "o", ms=2, color=C_CELL)
        lo_, hi_ = (m - se).min(), (m + se).max()
        ax.set_ylim(lo_ - 0.05 * (hi_ - lo_), hi_ + 0.55 * (hi_ - lo_))
        ax.text(0.02, 0.97, f"n = {len(seg)} events\n{1000/fs:.0f} ms / sample",
                transform=ax.transAxes, fontsize=5.5, va="top", color=INK2)
        ax.text(0.98, 0.97, f"rise (10–90%)\n{res['kinetics'].rise_10_90_s.median()*1000:.0f} ms (median)",
                transform=ax.transAxes, fontsize=5.5, va="top", ha="right", color=INK2)
    ax.set_xlabel("Time from peak (s)", fontsize=7); ax.set_ylabel(r"$\Delta F/F_0$", fontsize=7)
    ax.set_title("Event-triggered average", fontsize=7, color=INK); ax.tick_params(labelsize=6)

    # g: PSD relative to noise floor
    ax = fig.add_subplot(gs[2, 0]); _panel(ax, "g")
    for D, col, lab in ((d_cell, C_CELL, "neurons"), (d_bg, C_BG, "background")):
        P = np.vstack([spectral_snr(d, fs)[1] for d in D]); f = spectral_snr(D[0], fs)[0]
        ax.fill_between(f[1:], np.percentile(P, 25, 0)[1:], np.percentile(P, 75, 0)[1:], color=col, alpha=0.2, lw=0)
        ax.plot(f[1:], np.median(P, 0)[1:], color=col, lw=1, label=lab)
    ax.axvspan(*CA_BAND, color=INK2, alpha=0.06, lw=0)
    ax.axhline(1, color=INK2, lw=0.5, ls=":")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Frequency (Hz)", fontsize=7); ax.set_ylabel("Power / noise floor", fontsize=7)
    ax.set_title("Spectral SNR", fontsize=7, color=INK); ax.legend(fontsize=6); ax.tick_params(labelsize=6)

    # h: spatial specificity
    ax = fig.add_subplot(gs[2, 1]); _panel(ax, "h")
    x = pr.roi_id.values
    ax.bar(x, pr.SBR, color=C_CELL, width=0.6)
    ax.axhline(1, color=INK2, lw=0.5, ls=":"); ax.set_yscale("log")
    ax.set_xticks(x); ax.set_xlabel("Neuron ROI", fontsize=7); ax.set_ylabel("SBR  (F$_0$ cell / F$_0$ neuropil)", fontsize=7)
    ax.set_title(f"Signal-to-background (median {pr.SBR.median():.1f})\n"
                 f"ring ΔF / cell ΔF at events: {np.nanmedian(pr.neuropil_contamination):.2f}",
                 fontsize=7, color=INK)
    ax.tick_params(labelsize=6)

    # i: pairwise correlation vs distance
    ax = fig.add_subplot(gs[2, 2]); _panel(ax, "i")
    dist = pairs.distance_px * (um_per_px or 1)
    ax.scatter(dist, pairs.r, s=8, color=C_CELL, lw=0, label=f"as recorded (median {pairs.r.median():.2f})")
    if pairs.r_common_removed.notna().any():
        ax.scatter(dist, pairs.r_common_removed, s=8, color=C_NP, lw=0,
                   label=f"common mode removed ({pairs.r_common_removed.median():.2f})")
    ax.axhline(0, color=INK2, lw=0.5, ls=":")
    ax.set_xlabel("Distance (µm)" if um_per_px else "Distance (px)", fontsize=7); ax.set_ylabel("Pairwise r (high-passed)", fontsize=7)
    ax.set_title(f"Neuron–neuron correlation\n(shared signal: {np.nanmedian(pr.common_mode_var_frac)*100:.0f}% of variance)",
                 fontsize=7, color=INK)
    ylo = np.nanmin([pairs.r.min(), pairs.r_common_removed.min(), 0])
    ax.set_ylim(ylo - 0.35, None)
    ax.legend(fontsize=5.5, loc="lower right")
    ax.tick_params(labelsize=6)

    plt.savefig(outpath, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- comparison (MetaLFM vs LFM)
COMPARE_METRICS = ["median_event_SNR", "nu_pct_per_sqrtHz", "SBR", "neuropil_contamination",
                   "common_mode_var_frac", "spectral_SNR"]


def compare_conditions(per_roi_csvs, outpath, metrics=COMPARE_METRICS):
    """per_roi_csvs: {"MetaLFM": path, "LFM": path}. Strip plots + two-sided Mann-Whitney U."""
    dfs = {k: pd.read_csv(v) for k, v in per_roi_csvs.items()}
    labels = list(dfs)
    cols = [C_CELL, C_NP, "#1baf7a", "#eda100"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(1.25 * len(metrics), 2.4))
    stats = []
    for ax, m in zip(np.atleast_1d(axes), metrics):
        vals = [dfs[k][m].dropna().values for k in labels]
        for i, (v, c) in enumerate(zip(vals, cols)):
            ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=8, color=c, lw=0)
            ax.plot([i - 0.25, i + 0.25], [np.median(v)] * 2, color=INK, lw=1)
        p = mannwhitneyu(vals[0], vals[1]).pvalue if len(labels) == 2 and min(map(len, vals)) > 1 else np.nan
        stats.append({"metric": m, **{f"median_{k}": float(np.median(v)) for k, v in zip(labels, vals)}, "p_mannwhitney": p})
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=6, rotation=30)
        ax.set_title(f"{m}\np = {p:.2g}", fontsize=6); ax.tick_params(labelsize=6)
    plt.tight_layout(); plt.savefig(outpath, dpi=300); plt.close(fig)
    out = pd.DataFrame(stats); out.to_csv(os.path.splitext(outpath)[0] + ".csv", index=False)
    return out


# ---------------------------------------------------------------- entry point
def run_functional_analysis(movie_path, rois, fps, out_dir, out_prefix, n_background=10, um_per_px=None,
                            bin_t=4, bin_s=2):
    """rois: movie-space (x0, y0, w, h). Writes CSVs, a JSON summary and <prefix>_functional_performance.png."""
    os.makedirs(out_dir, exist_ok=True)
    P = lambda name: os.path.join(out_dir, f"{out_prefix}_{name}")
    first = next(core.iter_frames(movie_path)); shape = first.shape[:2]

    # pass 1 is cheap if a mean frame is available from the main pipeline; otherwise compute it here
    mean_path = P("mean_frame.npy")
    if os.path.exists(mean_path):
        mean_frame = np.load(mean_path)
    else:
        acc, n = None, 0
        for f in core.iter_frames(movie_path):
            acc = f.astype(np.float64) if acc is None else acc + f; n += 1
        mean_frame = (acc / n).astype(np.float32); np.save(mean_path, mean_frame)

    bg_rois = pick_background_rois(mean_frame, rois, n=n_background)
    np_masks = neuropil_masks(shape, rois)
    bg_area = background_area_mask(mean_frame, rois)
    F_cell, F_np, F_bg, F_area, mean_frame, binned = stream_extract(movie_path, rois, bg_rois, np_masks, bg_area,
                                                                    bin_t, bin_s)
    steps = core.find_segment_steps(F_cell)
    res = analyze(F_cell, F_np, F_bg, fps, rois, step_frames=steps, F_area=F_area)
    corr_img = local_correlation_image(binned, fps / bin_t)

    res["per_roi"].to_csv(P("functional_per_roi.csv"), index=False)
    res["background"].to_csv(P("functional_background_rois.csv"), index=False)
    res["events"].to_csv(P("functional_events.csv"), index=False)
    res["kinetics"].to_csv(P("functional_event_kinetics.csv"), index=False)
    res["sweep"].to_csv(P("functional_threshold_sweep.csv"), index=False)
    res["pairs"].to_csv(P("functional_pairwise_corr.csv"), index=False)
    core.rois_to_df(bg_rois, "background").to_csv(P("background_rois.csv"), index=False)
    np.save(P("local_corr_image.npy"), corr_img)
    t = np.arange(F_cell.shape[1]) / fps
    core.traces_to_wide_df(t, res["d_corr"], prefix="dFF0_common_removed_ROI").assign(
        dFF0_cellfree_area=res["d_area"]).to_csv(P("roi_traces_dFF0_common_mode_removed.csv"), index=False)

    pr, bg, sw = res["per_roi"], res["background"], res["sweep"]
    fp_at_k = sw[(sw.type == "background") & (np.isclose(sw.k_sigma, DETECT_K))].events_per_min
    summary = {
        "fps_Hz": fps, "sample_interval_ms": 1000 / fps, "analysed_minutes": round(res["minutes"], 2),
        "n_neurons": int(len(pr)), "n_background_rois": int(len(bg)),
        "detection_threshold_sigma": DETECT_K,
        "median_sigma_frame_dFF0": float(pr.sigma_frame_dFF0.median()),
        "median_nu_pct_per_sqrtHz": float(pr.nu_pct_per_sqrtHz.median()),
        "median_event_SNR": float(pr.median_event_SNR.median()),
        "max_event_SNR": float(pr.max_event_SNR.max()),
        "neurons_with_events": int((pr.n_events > 0).sum()),
        "median_event_rate_neuron_per_min": float(pr.event_rate_per_min.median()),
        "median_false_positive_rate_background_per_min": float(fp_at_k.median()),
        "median_rise_10_90_ms": float(res["kinetics"].rise_10_90_s.median() * 1000),
        "median_samples_on_rise": float(res["kinetics"].samples_on_rise.median()),
        "median_half_decay_s": float(res["kinetics"].half_decay_s.median()),
        "median_spectral_SNR_neuron": float(pr.spectral_SNR.median()),
        "median_spectral_SNR_background": float(bg.spectral_SNR.median()),
        "median_SBR": float(pr.SBR.median()),
        "median_neuropil_contamination": float(pr.neuropil_contamination.median()),
        "median_r_cell_neuropil": float(pr.r_cell_neuropil.median()),
        "median_pairwise_r_neurons": float(res["pairs"].r.median()),
        "median_pairwise_r_neurons_common_removed": float(res["pairs"].r_common_removed.median()),
        "median_r_neuron_vs_cellfree_area": float(pr.r_common_mode.median()),
        "median_common_mode_var_frac": float(pr.common_mode_var_frac.median()),
        "median_event_SNR_common_removed": float(pr.median_event_SNR_common_removed.median()),
        "common_mode_correction_applied": COMMON_MODE_CORRECTION,
        "segment_steps_excluded_frames": [int(s) for s in steps],
        "median_background_roi_F": float(np.median(F_bg)),
        "background_null_valid": bool(np.median(F_bg) >= 10),
    }
    if not summary["background_null_valid"]:
        print(f"⚠️  Background ROIs average {np.median(F_bg):.1f} counts: the false-positive null (panels d, e) is "
              "limited by intensity quantisation. Use the 32-bit reconstructed stack for these two panels.")
    with open(P("functional_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    plot_functional_figure(res, mean_frame, corr_img, rois, bg_rois, np_masks, fps, bin_s,
                           P("functional_performance.png"), um_per_px=um_per_px, bg_area=bg_area)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--movie", required=True)
    ap.add_argument("--rois", required=True, help="*_rois_on_movie.csv (x0,y0,w,h in movie pixels)")
    ap.add_argument("--fps", type=float, default=core.FPS)
    ap.add_argument("--prefix", default="meta-lfm-cortical")
    ap.add_argument("--um-per-px", type=float, default=None)
    a = ap.parse_args()
    rois = core.load_rois_csv(a.rois)
    out_dir = core.make_run_dir(a.prefix)           # new folder; earlier outputs untouched
    s = run_functional_analysis(a.movie, rois, a.fps, out_dir, a.prefix, um_per_px=a.um_per_px)
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
