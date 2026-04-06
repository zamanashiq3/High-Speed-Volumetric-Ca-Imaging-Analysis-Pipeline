import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image, ImageSequence
import imageio.v2 as imageio
import pandas as pd
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import percentile_filter
import os

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.linewidth": 1,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False
})

# ---------------- Safe Image/Movie Loader ----------------
def safe_imread(path):
    """Load JPEG/TIFF/AVI safely under NumPy 2.x without using tifffile."""
    ext = path.lower()
    if ext.endswith((".jpg", ".jpeg", ".png")):
        return np.array(Image.open(path)).astype(np.uint8)

    elif ext.endswith((".tif", ".tiff")):
        img = Image.open(path)
        frames = [np.array(frame) for frame in ImageSequence.Iterator(img)]
        if len(frames) > 1:
            arr = np.stack(frames)   # (T,H,W) multipage TIFF
        else:
            arr = frames[0]          # (H,W) single frame
        return arr.astype(np.float32)

    else:  # AVI or other movie formats
        frames = imageio.mimread(path)
        if len(frames) > 1:
            arr = np.stack(frames)
            if arr.ndim == 4:  # color movie (T,H,W,C)
                arr = arr.mean(axis=-1)  # grayscale
        else:
            arr = np.array(frames[0])
        return arr.astype(np.float32)


# ---------------- ROI Picker ----------------
class BoxROISelector:
    def __init__(self, img):
        self.img = img
        self.rois = []
        self.fig, self.ax = plt.subplots()
        self.ax.imshow(img)
        self.RS = RectangleSelector(
            self.ax, self.onselect,
            useblit=True, interactive=True
        )
        plt.connect('key_press_event', self.toggle_selector)
        plt.title("Draw ROI boxes, press Enter when done")
        plt.show()

    def onselect(self, eclick, erelease):
        x0, y0 = int(eclick.xdata), int(eclick.ydata)
        x1, y1 = int(erelease.xdata), int(erelease.ydata)
        w, h = abs(x1-x0), abs(y1-y0)
        self.rois.append((min(x0, x1), min(y0, y1), w, h))
        self.ax.add_patch(plt.Rectangle((min(x0, x1), min(y0, y1)), w, h,
                                        fill=False, edgecolor="r", lw=1.5))
        self.fig.canvas.draw_idle()

    def toggle_selector(self, event):
        if event.key == 'enter':
            plt.close(self.fig)


# ---------------- ROI Rescaling ----------------
def rescale_rois(rois, from_shape, to_shape):
    sy = to_shape[0] / from_shape[0]
    sx = to_shape[1] / from_shape[1]
    out = []
    for x0, y0, w, h in rois:
        out.append([int(round(x0*sx)), int(round(y0*sy)),
                    int(round(w*sx)),  int(round(h*sy))])
    return out


# ---------------- Trace Extraction ----------------
def extract_traces(movie_stack, rois):
    if movie_stack.ndim == 2:
        raise ValueError("Movie stack must be multi-frame for Ca²⁺ analysis.")
    stack = movie_stack.astype(np.float32)  # (T,H,W)
    traces = []
    for (x0, y0, w, h) in rois:
        roi = stack[:, y0:y0+h, x0:x0+w]
        roi_trace = roi.mean(axis=(1, 2))
        traces.append(roi_trace)
    return np.array(traces)  # (nROI, T)


# ---------------- ΔF/F with Rolling Baseline ----------------
def dff_robust(traces, f0_win_s=10, fs=20, p=10):
    win = max(3, int(round(f0_win_s * fs)))
    out = []
    for y in traces:
        F0 = percentile_filter(y, size=win, percentile=p, mode='nearest')
        F0 = np.clip(F0, 1e-6, None)
        out.append((y - F0) / F0)
    return np.vstack(out)  # (nROI, T)


# ---------------- Ca²⁺ Peak Analysis ----------------
def bandpass(x, fs, lo=0.1, hi=5.0, order=2):
    b, a = butter(order, [lo, hi], btype='band', fs=fs)
    return filtfilt(b, a, x)

def find_peaks_ca(dff, fs):
    peak_positions = []
    rows = []
    for i, y in enumerate(dff):
        yf = bandpass(y, fs)
        mad = np.median(np.abs(yf - np.median(yf))) + 1e-9
        prom = 3.0 * mad
        peaks, props = find_peaks(yf, prominence=prom, distance=int(0.25 * fs))
        freq_hz = 1.0 / np.mean(np.diff(peaks) / fs) if len(peaks) > 1 else np.nan

        rows.append({
            "ROI": i + 1,
            "n_peaks": int(len(peaks)),
            "mean_amp": float(np.mean(y[peaks])) if len(peaks) else np.nan,
            "Ca2+_frequency_Hz": float(freq_hz) if np.isfinite(freq_hz) else np.nan,
            "prominence_threshold_used": float(prom),
        })
        peak_positions.append((peaks, y[peaks] if len(peaks) else np.array([]), props))
    return pd.DataFrame(rows), peak_positions


# ---------------- Visualizations ----------------
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
    plt.ylabel("ΔF/F (offset)")
    plt.title("Ca²⁺ Responses (offset)")
    plt.legend(ncol=2, fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_offset_traces.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    for i, roi in enumerate(dff):
        peaks, peak_vals, _props = peak_positions[i]
        plt.plot(time, roi, lw=1, label=f"ROI {i+1}")
        if len(peaks):
            plt.plot(time[peaks], peak_vals, "o", markersize=3)
    plt.xlabel("Time (s)")
    plt.ylabel("ΔF/F")
    plt.title("Ca²⁺ Peaks")
    plt.legend(ncol=2, fontsize=7)
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_traces_peaks.png")
    plt.close()

def plot_kymograph(time, dff, out_prefix):
    plt.figure(figsize=(6, 4))
    plt.imshow(dff, aspect="auto", cmap="viridis",
               extent=[time.min(), time.max(), 0.5, dff.shape[0] + 0.5])
    plt.colorbar(label="ΔF/F")
    plt.yticks(np.arange(1, dff.shape[0] + 1))
    plt.xlabel("Time (s)")
    plt.ylabel("ROI")
    plt.title("Ca²⁺ Kymograph")
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_kymograph.png")
    plt.close()


# ---------------- CSV Export Helpers ----------------
def rois_to_df(rois, label):
    """rois: list of (x0,y0,w,h)"""
    rows = []
    for i, (x0, y0, w, h) in enumerate(rois, start=1):
        rows.append({
            "roi_id": i,
            "x0": int(x0), "y0": int(y0),
            "w": int(w), "h": int(h),
            "x1": int(x0 + w), "y1": int(y0 + h),
            "source": label
        })
    return pd.DataFrame(rows)

def traces_to_wide_df(time_s, arr, prefix="ROI"):
    """
    arr: (nROI, T)
    returns wide df: time_s + ROI_1..ROI_n
    """
    data = {"time_s": time_s.astype(float)}
    for i in range(arr.shape[0]):
        data[f"{prefix}_{i+1}"] = arr[i].astype(float)
    return pd.DataFrame(data)

def peaks_to_long_df(time_s, peak_positions, roi_meta_df=None):
    """
    peak_positions: list of (peaks_idx, peak_vals, props)
    returns long df: roi_id, peak_index, peak_time_s, peak_value, prominence, left_bases, right_bases
    """
    rows = []
    for i, (peaks_idx, peak_vals, props) in enumerate(peak_positions, start=1):
        if len(peaks_idx) == 0:
            continue
        prom = props.get("prominences", np.full(len(peaks_idx), np.nan))
        lb = props.get("left_bases", np.full(len(peaks_idx), -1))
        rb = props.get("right_bases", np.full(len(peaks_idx), -1))
        for k, (pi, pv) in enumerate(zip(peaks_idx, peak_vals)):
            rows.append({
                "roi_id": i,
                "peak_number": k + 1,
                "frame_index": int(pi),
                "time_s": float(time_s[pi]),
                "dff_value": float(pv),
                "prominence": float(prom[k]) if k < len(prom) else np.nan,
                "left_base": int(lb[k]) if k < len(lb) else -1,
                "right_base": int(rb[k]) if k < len(rb) else -1,
            })
    df = pd.DataFrame(rows)
    if roi_meta_df is not None and not df.empty:
        df = df.merge(roi_meta_df.rename(columns={"roi_id": "roi_id"}), on="roi_id", how="left")
    return df


# ---------------- Main Pipeline ----------------
def run_pipeline(roi_image_path, movie_path, fps=20, out_prefix="calcium"):
    # keep outputs organized
    out_dir = f"{out_prefix}_outputs"
    os.makedirs(out_dir, exist_ok=True)

    comp_img = safe_imread(roi_image_path)
    movie_stack = safe_imread(movie_path)

    # ROI selection on JPEG
    selector = BoxROISelector(comp_img)
    rois_img = selector.rois

    if len(rois_img) == 0:
        raise RuntimeError("No ROIs were selected. Draw at least one box and press Enter.")

    # Rescale ROIs to movie frame size
    Hi, Wi = comp_img.shape[:2]
    Hm, Wm = movie_stack.shape[1], movie_stack.shape[2]
    rois = rescale_rois(rois_img, (Hi, Wi), (Hm, Wm))

    # --- Save ROI definitions as CSV ---
    rois_img_df = rois_to_df(rois_img, label="roi_on_image")
    rois_mov_df = rois_to_df(rois, label="roi_on_movie")
    rois_img_df.to_csv(os.path.join(out_dir, f"{out_prefix}_rois_on_image.csv"), index=False)
    rois_mov_df.to_csv(os.path.join(out_dir, f"{out_prefix}_rois_on_movie.csv"), index=False)

    # QC overlays
    mean_frame = movie_stack.mean(axis=0)
    plot_roi_overlay(comp_img, rois_img, os.path.join(out_dir, f"{out_prefix}_roi_on_image.png"), "ROIs on JPEG")
    plot_roi_overlay(mean_frame, rois, os.path.join(out_dir, f"{out_prefix}_roi_on_movie.png"), "ROIs on movie mean")

    # Extract raw traces (mean intensity per ROI)
    traces = extract_traces(movie_stack, rois)  # (nROI,T)

    # ΔF/F
    dff = dff_robust(traces, f0_win_s=10, fs=fps, p=10)

    # Time vector
    T = dff.shape[1]
    time = np.arange(T) / fps

    # --- Save ALL data as CSV (wide format) ---
    traces_df = traces_to_wide_df(time, traces, prefix="ROI_raw")
    dff_df = traces_to_wide_df(time, dff, prefix="ROI_dff")
    traces_df.to_csv(os.path.join(out_dir, f"{out_prefix}_roi_traces_raw.csv"), index=False)
    dff_df.to_csv(os.path.join(out_dir, f"{out_prefix}_roi_traces_dff.csv"), index=False)

    # Ca²⁺ responses (metrics + peaks)
    results, peak_positions = find_peaks_ca(dff, fs=fps)

    # --- Save metrics as CSV (and Excel if you still want it) ---
    results.to_csv(os.path.join(out_dir, f"{out_prefix}_metrics.csv"), index=False)
    results.to_excel(os.path.join(out_dir, f"{out_prefix}_metrics.xlsx"), index=False)

    # --- Save peak events as CSV (long/tidy format) ---
    # include movie ROI geometry columns so you can map back later
    roi_meta = rois_mov_df.drop(columns=["source"]).copy()
    peaks_long = peaks_to_long_df(time, peak_positions, roi_meta_df=roi_meta)
    peaks_long.to_csv(os.path.join(out_dir, f"{out_prefix}_peaks_events_long.csv"), index=False)

    # Figures
    plot_traces(time, dff, peak_positions, os.path.join(out_dir, out_prefix))
    plot_kymograph(time, dff, os.path.join(out_dir, out_prefix))

    print("✅ Analysis complete.")
    print(f"📁 Outputs saved to: {out_dir}")
    print("CSV files written:")
    print(f"  - {out_prefix}_rois_on_image.csv")
    print(f"  - {out_prefix}_rois_on_movie.csv")
    print(f"  - {out_prefix}_roi_traces_raw.csv")
    print(f"  - {out_prefix}_roi_traces_dff.csv")
    print(f"  - {out_prefix}_metrics.csv")
    print(f"  - {out_prefix}_peaks_events_long.csv")


# ---------------- Entry ----------------
def main():
    roi_image_path = "combine.jpg"   # JPEG for ROI picking
    movie_path = "combine.tif"       # TIFF stack or AVI
    fps = 20
    out_prefix = "meta-lfm-cortical"
    run_pipeline(roi_image_path, movie_path, fps=fps, out_prefix=out_prefix)

if __name__ == "__main__":
    main()