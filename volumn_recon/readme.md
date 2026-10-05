# Meta-LFM volumetric reconstruction and calcium analysis

Reconstruction of metalens-array light-field microscope (Meta-LFM) images into
depth stacks, and analysis of calcium dynamics from raw light-field movies.

| File | Purpose |
|------|---------|
| `metalfm_reconstruct.py` | One raw light-field image → depth stack (shift-sum, dense refocus, Richardson-Lucy). All parameters are fixed in `CONFIG`. |
| `metalfm_calcium.py` | Raw light-field movie → 4-D volume → ROI traces, ΔF/F, events, metrics. Imports the reconstruction and does not modify it. |
| `legacy/processing_image_cortical.py` | The original 2-D ROI script the calcium analysis is derived from. |
| `examples/1_stack/` | Figures and run metadata from the first real recording (4054 frames, 20 Hz). |

## Install

```bash
pip install -r requirements.txt
```

PyTorch is optional. Without it the Richardson-Lucy steps are skipped; with a
CUDA build they run on the GPU.

## Usage

Single image:

```bash
python metalfm_reconstruct.py raw_image.tif
```

Movie (multi-page TIFF, one raw light-field frame per page):

```bash
python metalfm_calcium.py movie.tif                  # draw ROI boxes, press Enter
python metalfm_calcium.py movie.tif --auto           # automatic ROIs
python metalfm_calcium.py movie.tif --rois rois.csv  # reuse saved ROIs
python metalfm_calcium.py folder_of_movies --auto    # every movie in a folder
```

Options: `--fps` (default 20) and `--force` (ignore the cached reconstruction).
Outputs go to `<name>_recon/` or `<name>_calcium/` next to the input.

## Method

1. **Grid.** Lenslet pitch and origin are measured from the image by FFT of the
   row and column profiles (sub-pixel).
2. **Views.** 9 × 9 positions are sampled inside every lenslet; views outside
   the pupil image (dimmer than 25 % of the brightest) are discarded.
3. **Parallax.** One formula is used by every stage. For a view offset of
   `d` sensor pixels and a defocus `z` in µm, the shift in lenslets is

   `shift = z · d · pixel · M² / (n · f_lens · pitch_lens)`

   so the depth axis is in micrometres.
4. **Shift-sum refocus** on the lenslet grid, and a dense shift-and-add stack.
5. **Multi-view Richardson-Lucy** with a Gaussian PSF whose blur grows with
   defocus, plus the per-view parallax shift.
6. **Movies.** Grid, view selection, view gains and intensity scale are measured
   once on the time-averaged frame and held fixed for every frame. Each frame
   gets the same fixed number of RL iterations.
7. **Calcium analysis.** Mean intensity per ROI at every depth plane; ΔF/F with
   a rolling 10th-percentile baseline (10 s window); band-pass 0.1–5 Hz; an
   event must exceed 3.5 noise SDs in both height and prominence, where the
   noise SD is estimated from frame-to-frame differences.

Three signal sources are reported side by side in the `method` column:
`lenslet2d` (sum over the aperture, no depth), `shift_sum` and `rl`.

## Configuration

Optics and reconstruction parameters are in `CONFIG` at the top of
`metalfm_reconstruct.py`; analysis parameters are in `DYN` at the top of
`metalfm_calcium.py`. Values used for `examples/1_stack`:

| Parameter | Value |
|-----------|-------|
| Emission wavelength | 525 nm |
| Objective | 20×, NA 0.5 |
| Immersion index `n_medium` | 1.33 |
| Camera pixel | 6.5 µm |
| Metalens pitch / focal length | 75 µm / 1430 µm |
| `shift_scale` | −1 (z axis flipped) |
| Depth planes (movie) | −30 … +30 µm, 13 planes |
| RL iterations per frame | 10 |

`metalfm_calcium.py` stores the SHA-256 of the reconstruction file it was
validated with, warns if the file differs, and writes the hash to
`*_run_metadata.json`. Update `FROZEN_SHA256` whenever `CONFIG` is changed on
purpose.

## Outputs of `metalfm_calcium.py`

| File | Content |
|------|---------|
| `*_metrics_best_z.csv` | One row per method × ROI at its most active plane |
| `*_metrics_all_planes.csv` | One row per method × ROI × depth plane |
| `*_peaks_events_long.csv` | One row per detected event |
| `*_network_metrics.csv` | Pairwise correlation, synchrony, global-event fraction |
| `*_<method>_traces_{raw,dff}_bestz.csv` | Traces, wide format |
| `*_traces_all_planes_long.csv` | Every trace at every plane, long format |
| `*_metrics.xlsx` | The tables above in one workbook |
| `*_rois.csv` | ROI boxes in display pixels, lenslets and µm |
| `*_<method>_TZYX_native.tif` | 4-D stacks on the lenslet grid |
| `*_rl_mip_movie_512.tif` | z-MIP movie upscaled to 512 × 512 |
| `*_run_metadata.json` | Every parameter of the run |
| `figures/` | ROI overlays, traces with events, kymographs, depth profiles, correlation matrices |

Per-ROI metrics: event count and rate, `Ca2+_frequency_Hz` (1 / mean
inter-event interval), mean / max / SD amplitude, prominence, FWHM,
inter-event interval and its CV, noise SD, SNR, ΔF/F maximum, 95th percentile
and area, baseline F0.

Quality flags:

- `low_baseline` – F0 was below the floor, so ΔF/F is unreliable.
- `best_z_at_edge` – the most active plane is the first or last one, so the
  depth of that ROI is not localised within the reconstructed range.
- `coactive_fraction` (events) – share of the other ROIs with an event within
  ± 2 frames. `global_event_fraction` (network) is the share of events where
  this is ≥ 0.8. Such events are flagged, not removed.

## Limitations

- **Resolution.** In focus, one lenslet is one sample: 3.75 µm in the object
  for 75 µm lenslets at 20×. The 512 × 512 outputs are interpolated for
  viewing and do not add resolution.
- **Depth calibration.** The parallax formula assumes the metalens array is at
  the native image plane with the sensor at its focal plane. The depth scale
  and its sign have not been checked against beads at known z positions.
- **PSF.** The RL forward model uses a Gaussian approximation, not a measured
  or wave-optics PSF.
- **Validation.** Event detection was tuned on a synthetic movie (6 cells, 33
  planted events: 29 detected, at most one false event across 36 noise ROIs).
  It has not been validated against ground truth on real recordings.

### Notes on `examples/1_stack`

The figures were produced by the first commit on this branch, before the
quality flags above were added. They show the pipeline running end to end on
real data, and three things to resolve before the numbers are used:

- ROIs 1–3 cover identical lenslets and ROIs 4–5 nearly so, which is why their
  correlations are 1.0. Duplicate boxes are now dropped automatically.
- Most detected events are single-frame spikes occurring in all ROIs at the
  same time (near 8, 15, 20, 44, 80, 143 and 168 s). That pattern points to a
  whole-field artefact rather than calcium transients. ΔF/F stays below ~5 %.
- The most active plane is +30 µm, the edge of the range, for six of seven
  ROIs, so depth is not localised in this run.