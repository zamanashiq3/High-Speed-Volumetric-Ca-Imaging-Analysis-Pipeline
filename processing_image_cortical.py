"""
ROI-based Ca2+ analysis for MetaLFM recordings (Fig. 6 of the manuscript).

Notation (used consistently in code, figures, CSV headers and manuscript):
    F(t)    raw fluorescence = mean pixel intensity inside the ROI in frame t
    F0(t)   baseline fluorescence = running low-percentile of F(t)
            (default: 10th percentile in a 10-s sliding window centred on t,
             nearest-value padding at the edges)
    dF/F0   = (F(t) - F0(t)) / F0(t), reported as a fraction (0.10 = 10 %)

Why a running baseline: the raw traces drift slowly over the ~200-s recording
(dye loading / de-esterification, illumination), so a single static F0 would
mix that drift into dF/F0. A low percentile keeps brief transients from
raising the baseline; the window is long compared with a transient (< ~2 s)
and short compared with the drift.
"""
import json
import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import RectangleSelector
from PIL import Image, ImageSequence
from scipy.ndimage import percentile_filter
from scipy.signal import butter, filtfilt, find_peaks

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.linewidth": 1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False
})

DFF_LABEL = r"$\Delta F/F_0$"

# ---------------- Analysis parameters (reported in Methods 4.2.2) ----------------
FPS = 20               # volume rate (Hz). MUST equal the acquisition rate stated in the manuscript.
F0_WIN_S = 10.0        # sliding-window length for F0 (s)
F0_PERCENTILE = 10     # percentile of F used as F0
BACKGROUND = 0.0       # constant background subtracted from F before dF/F0 (0 = none)
SEGMENT_FRAMES = 1000  # frames per acquisition/reconstruction batch; F0 is not computed across
                       # batch boundaries. Set None for a single continuous recording.

# Settings tested in the baseline sensitivity check
SENS_WINDOWS_S = (5, 10, 20, 30)
SENS_PERCENTILES = (5, 10, 20)


# ---------------- Image / movie loading ----------------
def safe_imread(path):
    """Load a single image (JPEG/PNG/TIFF page) as an array."""
    ext = path.lower()
    if ext.endswith((".jpg", ".jpeg", ".png")):
        return np.array(Image.open(path)).astype(np.uint8)
    img = Image.open(path)
    return np.array(img).astype(np.float32)


def iter_frames(path):
    """Yield grayscale float32 frames one at a time (multipage TIFF or movie).

    Streaming avoids loading the whole movie into memory; a 4000-frame AVI
    would otherwise need several GB.
    """
    ext = path.lower()
    if ext.endswith((".tif", ".tiff")):
        with Image.open(path) as img:
            for frame in ImageSequence.Iterator(img):
                yield np.array(frame, dtype=np.float32)
    else:
        reader = imageio.get_reader(path)
        try:
            for frame in reader:
                f = np.asarray(frame, dtype=np.float32)
                yield f.mean(axis=-1) if f.ndim == 3 else f
        finally:
            reader.close()


def movie_dtype_note(path):
    """Warn if the movie is an 8-bit (possibly lossy) export rather than the
    reconstructed floating-point stack: with dim ROIs (F ~ 5 counts) one 8-bit
    step is already ~20 % dF/F0."""
    if not path.lower().endswith((".tif", ".tiff")):
        print("⚠️  Movie is not a TIFF stack. For quantitative dF/F0 use the "
              "reconstructed 32-bit TIFF, not an 8-bit/MJPEG AVI export.")


# ---------------- ROI picker ----------------
class BoxROISelector:
    def __init__(self, img):
        self.img = img
        self.rois = []
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(img)
        self.RS = RectangleSelector(self.ax, self.onselect, useblit=True, interactive=True)
        self.fig.canvas.mpl_connect('key_press_event', self.toggle_selector)
        plt.title("Draw ROI boxes, press Enter when done")
        plt.show()

    def onselect(self, eclick, erelease):
        x0, y0 = int(eclick.xdata), int(eclick.ydata)
        x1, y1 = int(erelease.xdata), int(erelease.ydata)
        w, h = abs(x1 - x0), abs(y1 - y0)
        if w < 2 or h < 2:          # ignore accidental clicks (zero-size ROI -> NaN trace)
            return
        self.rois.append((min(x0, x1), min(y0, y1), w, h))
        self.ax.add_patch(plt.Rectangle((min(x0, x1), min(y0, y1)), w, h,
                                        fill=False, edgecolor="r", lw=1.5))
        self.fig.canvas.draw_idle()

    def toggle_selector(self, event):
        if event.key == 'enter':
            plt.close(self.fig)


def load_rois_csv(path):
    """Re-use ROIs saved by a previous run (*_rois_on_image.csv) for exact reproducibility."""
    df = pd.read_csv(path)
    return [tuple(int(v) for v in r) for r in df[["x0", "y0", "w", "h"]].to_numpy()]


# ---------------- ROI rescaling ----------------
def rescale_rois(rois, from_shape, to_shape):
    sy = to_shape[0] / from_shape[0]
    sx = to_shape[1] / from_shape[1]
    return [[int(round(x0 * sx)), int(round(y0 * sy)),
             max(1, int(round(w * sx))), max(1, int(round(h * sy)))]
            for x0, y0, w, h in rois]


# ---------------- Trace extraction (streaming) ----------------
def extract_traces(movie_path, rois):
    """Return raw traces F (nROI, T) and the mean frame, reading one frame at a time."""
    traces, acc, n = [], None, 0
    for f in iter_frames(movie_path):
        traces.append([f[y0:y0 + h, x0:x0 + w].mean() for (x0, y0, w, h) in rois])
        acc = f.astype(np.float64) if acc is None else acc + f
        n += 1
    if n < 2:
        raise ValueError("Movie must be multi-frame for Ca2+ analysis.")
    return np.asarray(traces, dtype=np.float64).T, (acc / n).astype(np.float32)


def movie_shape(movie_path):
    return next(iter_frames(movie_path)).shape


# ---------------- dF/F0 with running-percentile baseline ----------------
def segment_bounds(T, segment_frames=SEGMENT_FRAMES):
    """[(start, stop), ...] covering 0..T. One segment if segment_frames is None."""
    if not segment_frames:
        return [(0, T)]
    edges = list(range(0, T, int(segment_frames))) + [T]
    return list(zip(edges[:-1], edges[1:]))


def compute_f0(F, fs, win_s=F0_WIN_S, p=F0_PERCENTILE, segment_frames=SEGMENT_FRAMES):
    """F0(t): p-th percentile of F within a win_s-second window centred on t.

    The window never crosses a segment boundary: F0 is computed independently
    inside each acquisition/reconstruction segment (nearest-value padding at
    segment edges). Otherwise an intensity offset between segments leaks into
    F0 and appears in dF/F0 as a flat-topped pseudo-event about win_s/2 long.
    """
    win = max(3, int(round(win_s * fs)))
    F0 = np.empty_like(F, dtype=np.float64)
    for a, b in segment_bounds(F.shape[1], segment_frames):
        for i, y in enumerate(F):
            F0[i, a:b] = percentile_filter(y[a:b], size=min(win, b - a), percentile=p, mode='nearest')
    return F0


def find_segment_steps(F, z_thresh=8.0):
    """Frames where several ROIs jump at once (possible segment boundaries)."""
    d = np.diff(F, axis=1)
    mad = np.median(np.abs(d - np.median(d, axis=1, keepdims=True)), axis=1, keepdims=True) * 1.4826 + 1e-9
    z = np.abs(d) / mad
    return np.where((z > z_thresh).sum(axis=0) >= 2)[0] + 1


def dff0(F, fs, win_s=F0_WIN_S, p=F0_PERCENTILE, background=BACKGROUND, warn=True,
         segment_frames=SEGMENT_FRAMES):
    """Return (dF/F0, F0). Both (nROI, T)."""
    Fc = F - background
    F0 = compute_f0(Fc, fs, win_s, p, segment_frames)
    if np.any(F0 <= 0):
        raise ValueError("F0 <= 0 for at least one ROI; check background subtraction.")
    low = np.where(np.median(F0, axis=1) < 10)[0]
    if warn and len(low):
        print(f"⚠️  Low baseline (median F0 < 10 counts) in ROI(s) {[int(i) for i in low + 1]}: "
              "dF/F0 there is sensitive to noise/quantisation.")
    return (Fc - F0) / F0, F0


# ---------------- Ca2+ peak analysis ----------------
def bandpass(x, fs, lo=0.1, hi=5.0, order=2):
    b, a = butter(order, [lo, hi], btype='band', fs=fs)
    return filtfilt(b, a, x)


def find_peaks_ca(dff, fs):
    peak_positions, rows = [], []
    for i, y in enumerate(dff):
        yf = bandpass(y, fs)
        mad = np.median(np.abs(yf - np.median(yf))) + 1e-9
        prom = 3.0 * mad
        peaks, props = find_peaks(yf, prominence=prom, distance=int(0.25 * fs))
        freq_hz = 1.0 / np.mean(np.diff(peaks) / fs) if len(peaks) > 1 else np.nan
        rows.append({
            "ROI": i + 1,
            "n_peaks": int(len(peaks)),
            "mean_peak_dFF0": float(np.mean(y[peaks])) if len(peaks) else np.nan,
            "event_rate_Hz": float(freq_hz) if np.isfinite(freq_hz) else np.nan,
            "prominence_threshold_used": float(prom),
        })
        peak_positions.append((peaks, y[peaks] if len(peaks) else np.array([]), props))
    return pd.DataFrame(rows), peak_positions


# ---------------- Baseline sensitivity check (reviewer request) ----------------
def baseline_sensitivity(F, fs, out_prefix):
    """Compare the default F0 with other windows/percentiles and with a static F0.

    Writes <prefix>_F0_sensitivity.csv (per ROI) and a summary table, and
    returns the summary DataFrame.
    """
    ref, _ = dff0(F, fs, warn=False)
    rows = []

    def add(label, win, pct, d):
        res, _ = find_peaks_ca(d, fs)
        for i in range(F.shape[0]):
            rows.append({"F0_method": label, "win_s": win, "percentile": pct, "ROI": i + 1,
                         "r_vs_default": float(np.corrcoef(d[i], ref[i])[0, 1]),
                         "n_peaks": int(res.n_peaks[i]),
                         "mean_peak_dFF0": float(res.mean_peak_dFF0[i]),
                         "sd_dFF0": float(d[i].std())})

    for w in SENS_WINDOWS_S:
        for p in SENS_PERCENTILES:
            add("running", w, p, dff0(F, fs, w, p, warn=False)[0])
    F0s = np.percentile(F - BACKGROUND, F0_PERCENTILE, axis=1, keepdims=True)
    add("static (whole-trace percentile)", np.nan, F0_PERCENTILE, (F - BACKGROUND - F0s) / F0s)
    if SEGMENT_FRAMES:
        add("running, ignoring segment boundaries", F0_WIN_S, F0_PERCENTILE,
            dff0(F, fs, warn=False, segment_frames=None)[0])

    df = pd.DataFrame(rows)
    df.to_csv(f"{out_prefix}_F0_sensitivity_per_roi.csv", index=False)
    summary = (df.groupby(["F0_method", "win_s", "percentile"], dropna=False)
                 .agg(median_r_vs_default=("r_vs_default", "median"),
                      min_r_vs_default=("r_vs_default", "min"),
                      total_peaks=("n_peaks", "sum"),
                      mean_peak_dFF0=("mean_peak_dFF0", "mean"))
                 .reset_index())
    summary.to_csv(f"{out_prefix}_F0_sensitivity_summary.csv", index=False)
    return summary


def plot_baseline_qc(time, F, F0, out_prefix, rois_to_show=None):
    """Raw F with the estimated F0 overlaid, one panel per ROI (QC of the baseline)."""
    idx = list(range(F.shape[0])) if rois_to_show is None else [r - 1 for r in rois_to_show]
    fig, axes = plt.subplots(len(idx), 1, figsize=(6, 1.0 * len(idx) + 0.6), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, i in zip(axes, idx):
        ax.plot(time, F[i], lw=0.5, color="0.4", label="F")
        ax.plot(time, F0[i], lw=1.2, color="C3", label=r"$F_0$ (running %d$^{th}$ pct, %g s)"
                % (F0_PERCENTILE, F0_WIN_S))
        ax.set_ylabel(f"ROI {i+1}", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].legend(fontsize=6, ncol=2, loc="upper left")
    axes[-1].set_xlabel("Time (s)")
    fig.supylabel("Raw F (a.u.)", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_F0_qc.png")
    plt.close()


def plot_segment_boundary_check(time, F, fs, out_prefix, rois_to_show=None):
    """Raw F with F0 computed across vs within segments, and the resulting dF/F0.

    Shows how an intensity offset at a batch boundary turns into a flat-topped
    pseudo-event when the F0 window crosses the boundary. By default shows the
    two ROIs with the largest step at a boundary.
    """
    T = F.shape[1]
    bounds = [a for a, _ in segment_bounds(T)[1:]]
    d_old, F0_old = dff0(F, fs, warn=False, segment_frames=None)
    d_new, F0_new = dff0(F, fs, warn=False)
    if rois_to_show is None:
        r = int(fs)
        step = np.zeros(F.shape[0])
        for b in bounds:
            lo, hi = max(0, b - r), min(T, b + r + 2)
            step = np.maximum(step, np.abs(F[:, b + 1:hi].mean(1) - F[:, lo:b].mean(1)) / np.median(F, 1))
        rois_to_show = list(np.argsort(-step)[:2] + 1)
    fig, ax = plt.subplots(3, len(rois_to_show), figsize=(3.6 * len(rois_to_show), 5.2), sharex=True,
                           squeeze=False, gridspec_kw=dict(height_ratios=[1.2, 1, 1]))
    for c, roi in enumerate(rois_to_show):
        i = roi - 1
        ax[0, c].plot(time, F[i], lw=.4, c="0.55", label="F")
        ax[0, c].plot(time, F0_old[i], c="C3", lw=1, label="F$_0$, window crosses boundary")
        ax[0, c].plot(time, F0_new[i], c="C0", lw=1, ls="--", label="F$_0$, segment-aware")
        ax[0, c].set_title(f"ROI {roi}", fontsize=8)
        ax[1, c].plot(time, d_old[i], lw=.5, c="C3")
        ax[2, c].plot(time, d_new[i], lw=.5, c="C0")
        lim = max(d_old[i].max(), d_new[i].max()) * 1.05
        lo = min(d_old[i].min(), d_new[i].min()) * 1.05
        for a in ax[:, c]:
            for b in bounds:
                a.axvline(time[b], c="k", lw=.4, ls=":")
        ax[1, c].set_ylim(lo, lim); ax[2, c].set_ylim(lo, lim)
        ax[2, c].set_xlabel("Time (s)")
    ax[0, 0].set_ylabel("Raw F")
    ax[1, 0].set_ylabel(DFF_LABEL + ", F$_0$\nacross segments", fontsize=8)
    ax[2, 0].set_ylabel(DFF_LABEL + ", F$_0$\nper segment", fontsize=8)
    ax[0, 0].legend(fontsize=6, loc="upper left")
    fig.suptitle(f"Dotted lines: {SEGMENT_FRAMES}-frame segment boundaries", fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_F0_segment_boundary_check.png")
    plt.close()


# ---------------- Figures ----------------
def plot_roi_overlay(img, rois, outpath, title):
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(title)
    for i, (x0, y0, w, h) in enumerate(rois):
        ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, ec='r', lw=1))
        ax.text(x0, max(0, y0 - 5), f"{i+1}", color='y', fontsize=7)
    ax.axis('off')
    plt.savefig(outpath, bbox_inches='tight')
    plt.close()


def plot_traces(time, dff, peak_positions, out_prefix, offset_step=0.08):
    plt.figure(figsize=(6, 4))
    for i, roi in enumerate(dff):
        plt.plot(time, roi + i * offset_step, lw=1, label=f"ROI {i+1}")
    plt.xlabel("Time (s)")
    plt.ylabel(f"{DFF_LABEL} (offset)")
    plt.title("Ca²⁺ responses (offset)")
    plt.legend(ncol=2, fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_offset_traces.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    for i, roi in enumerate(dff):
        peaks, peak_vals, _ = peak_positions[i]
        plt.plot(time, roi, lw=1, label=f"ROI {i+1}")
        if len(peaks):
            plt.plot(time[peaks], peak_vals, "o", markersize=3)
    plt.xlabel("Time (s)")
    plt.ylabel(DFF_LABEL)
    plt.title("Ca²⁺ peaks")
    plt.legend(ncol=2, fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_traces_peaks.png")
    plt.close()


def plot_kymograph(time, dff, out_prefix):
    plt.figure(figsize=(6, 4))
    # origin='lower' puts ROI 1 (row 0) at the bottom, matching the y-tick labels.
    # (With the default origin='upper' ROI 1 was drawn in the row labelled n.)
    plt.imshow(dff, aspect="auto", cmap="viridis", origin="lower",
               extent=[time.min(), time.max(), 0.5, dff.shape[0] + 0.5])
    for a, _ in segment_bounds(dff.shape[1])[1:]:
        plt.axvline(time[a], color="w", lw=0.5, ls=":")
    plt.colorbar(label=DFF_LABEL)
    plt.yticks(np.arange(1, dff.shape[0] + 1))
    plt.xlabel("Time (s)")
    plt.ylabel("ROI")
    plt.title("Ca²⁺ kymograph")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_kymograph.png")
    plt.close()


# ---------------- CSV export helpers ----------------
def rois_to_df(rois, label):
    rows = [{"roi_id": i, "x0": int(x0), "y0": int(y0), "w": int(w), "h": int(h),
             "x1": int(x0 + w), "y1": int(y0 + h), "source": label}
            for i, (x0, y0, w, h) in enumerate(rois, start=1)]
    return pd.DataFrame(rows)


def traces_to_wide_df(time_s, arr, prefix="ROI"):
    data = {"time_s": time_s.astype(float)}
    for i in range(arr.shape[0]):
        data[f"{prefix}_{i+1}"] = arr[i].astype(float)
    return pd.DataFrame(data)


def peaks_to_long_df(time_s, peak_positions, roi_meta_df=None):
    rows = []
    for i, (peaks_idx, peak_vals, props) in enumerate(peak_positions, start=1):
        if len(peaks_idx) == 0:
            continue
        prom = props.get("prominences", np.full(len(peaks_idx), np.nan))
        lb = props.get("left_bases", np.full(len(peaks_idx), -1))
        rb = props.get("right_bases", np.full(len(peaks_idx), -1))
        for k, (pi, pv) in enumerate(zip(peaks_idx, peak_vals)):
            rows.append({"roi_id": i, "peak_number": k + 1, "frame_index": int(pi),
                         "time_s": float(time_s[pi]), "dFF0_value": float(pv),
                         "prominence_bandpassed": float(prom[k]),
                         "left_base": int(lb[k]), "right_base": int(rb[k])})
    df = pd.DataFrame(rows)
    if roi_meta_df is not None and not df.empty:
        df = df.merge(roi_meta_df, on="roi_id", how="left")
    return df


# ---------------- Main pipeline ----------------
def make_run_dir(out_prefix, run_tag=None):
    """New folder per run: <prefix>_outputs/run_<timestamp>/. Earlier outputs are never touched."""
    from datetime import datetime
    tag = run_tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(f"{out_prefix}_outputs", f"run_{tag}")
    if os.path.exists(out_dir):
        raise FileExistsError(f"{out_dir} already exists; choose another run_tag so nothing is overwritten.")
    os.makedirs(out_dir)
    return out_dir


def run_pipeline(roi_image_path, movie_path, fps=FPS, out_prefix="calcium", roi_csv=None,
                 functional=True, um_per_px=None, run_tag=None):
    """Full analysis. Outputs go to a NEW folder <prefix>_outputs/run_<timestamp>/;
    files from earlier runs (including <prefix>_outputs/* from the original script) are left as they are."""
    out_dir = make_run_dir(out_prefix, run_tag)
    print(f"📁 Writing to {out_dir} (earlier outputs are not modified)")
    P = lambda name: os.path.join(out_dir, f"{out_prefix}_{name}")

    movie_dtype_note(movie_path)
    comp_img = safe_imread(roi_image_path)

    # ROIs: interactive, or re-loaded from a previous run for exact reproducibility
    rois_img = load_rois_csv(roi_csv) if roi_csv else BoxROISelector(comp_img).rois
    if len(rois_img) == 0:
        raise RuntimeError("No ROIs were selected. Draw at least one box and press Enter.")

    Hi, Wi = comp_img.shape[:2]
    Hm, Wm = movie_shape(movie_path)[:2]
    rois = rescale_rois(rois_img, (Hi, Wi), (Hm, Wm))

    rois_img_df = rois_to_df(rois_img, label="roi_on_image")
    rois_mov_df = rois_to_df(rois, label="roi_on_movie")
    rois_img_df.to_csv(P("rois_on_image.csv"), index=False)
    rois_mov_df.to_csv(P("rois_on_movie.csv"), index=False)

    # Raw traces F (streamed) + mean frame for QC
    F, mean_frame = extract_traces(movie_path, rois)
    T = F.shape[1]
    time = np.arange(T) / fps
    print(f"{T} frames at {fps} Hz -> {T / fps:.1f} s recording "
          "(check this matches the duration and rate stated in the manuscript)")

    np.save(P("mean_frame.npy"), mean_frame)   # re-used by functional_metrics (saves one pass)
    plot_roi_overlay(comp_img, rois_img, P("roi_on_image.png"), "ROIs on projection image")
    plot_roi_overlay(mean_frame, rois, P("roi_on_movie.png"), "ROIs on movie mean")

    steps = find_segment_steps(F)
    if len(steps):
        print(f"⚠️  Simultaneous intensity steps across ROIs at frames {steps.tolist()} "
              f"(SEGMENT_FRAMES = {SEGMENT_FRAMES}). Steps that are not segment boundaries "
              "need checking before interpreting synchronous events.")

    # dF/F0 with running-percentile F0
    dff, F0 = dff0(F, fs=fps)

    traces_to_wide_df(time, F, prefix="F_raw_ROI").to_csv(P("roi_traces_F_raw.csv"), index=False)
    traces_to_wide_df(time, F0, prefix="F0_ROI").to_csv(P("roi_traces_F0.csv"), index=False)
    traces_to_wide_df(time, dff, prefix="dFF0_ROI").to_csv(P("roi_traces_dFF0.csv"), index=False)

    # Record every parameter that defines F0 (cite these in Methods)
    params = {"fps_Hz": fps, "n_frames": int(T), "duration_s": T / fps,
              "F0_definition": "running percentile of F, centred window, nearest-value edge padding",
              "F0_window_s": F0_WIN_S, "F0_window_frames": max(3, int(round(F0_WIN_S * fps))),
              "F0_percentile": F0_PERCENTILE, "background_subtracted": BACKGROUND,
              "segment_frames": SEGMENT_FRAMES,
              "segment_bounds": segment_bounds(T),
              "detected_simultaneous_steps_frames": steps.tolist(),
              "dFF0_formula": "(F - F0) / F0, fraction (0.1 = 10%)",
              "peak_detection": "bandpass 0.1-5 Hz (Butterworth, order 2, zero-phase); "
                                "prominence >= 3*MAD; min distance 0.25 s"}
    with open(P("analysis_parameters.json"), "w") as fh:
        json.dump(params, fh, indent=2)

    results, peak_positions = find_peaks_ca(dff, fs=fps)
    results.to_csv(P("metrics.csv"), index=False)
    try:
        results.to_excel(P("metrics.xlsx"), index=False)
    except ImportError:
        pass

    roi_meta = rois_mov_df.drop(columns=["source"])
    peaks_to_long_df(time, peak_positions, roi_meta).to_csv(P("peaks_events_long.csv"), index=False)

    # Baseline QC + sensitivity (answers "how was F0 determined / does it matter")
    plot_baseline_qc(time, F, F0, os.path.join(out_dir, out_prefix))
    summary = baseline_sensitivity(F, fps, os.path.join(out_dir, out_prefix))
    print("\nF0 sensitivity (r = Pearson correlation of dF/F0 with the default setting):")
    print(summary.round(3).to_string(index=False))

    plot_traces(time, dff, peak_positions, os.path.join(out_dir, out_prefix))
    plot_kymograph(time, dff, os.path.join(out_dir, out_prefix))
    if SEGMENT_FRAMES:
        plot_segment_boundary_check(time, F, fps, os.path.join(out_dir, out_prefix))

    if functional:
        import functional_metrics as fm
        summary = fm.run_functional_analysis(movie_path, rois, fps, out_dir, out_prefix, um_per_px=um_per_px)
        print("\nFunctional-imaging performance:")
        for k, v in summary.items():
            print(f"  {k}: {v}")

    print(f"\n✅ Analysis complete. Outputs saved to: {out_dir}")


# ---------------- Entry ----------------
def main():
    run_pipeline(
        roi_image_path="combine.jpg",   # projection image used for ROI picking
        movie_path="combine.tif",       # reconstructed z = 0 µm stack (32-bit TIFF preferred)
        fps=FPS,
        out_prefix="meta-lfm-cortical",
        roi_csv=None,                   # e.g. "meta-lfm-cortical_outputs/meta-lfm-cortical_rois_on_image.csv"
        functional=True,                # SNR / noise / kinetics / crosstalk metrics + figure
        um_per_px=None,                 # pixel size in the reconstructed plane (µm) for distance axes
    )


if __name__ == "__main__":
    main()
