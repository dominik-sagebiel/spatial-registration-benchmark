# SIFT vs SuperPoint vs ASHLAR vs VALIS – registration benchmarking

Four scripts that register two images with four pipelines and, if ground-truth landmarks are given, measure the registration error (TRE) in pixels:

- **SIFT** and **SuperPoint** – FLANN matching → RANSAC homography → affine transform
- **ASHLAR-style phase correlation** – ASHLAR's registration step applied to the two whole images; translation only
- **VALIS** – the full VALIS rigid registration

All four produce a 2×3 affine (img1 → img2) on the same preprocessed images, so TRE and flicker GIFs are directly comparable.

## Scripts

| Script | Compares | Input formats | Results folder |
|---|---|---|---|
| `Visium_Visium_adjacent.py` | adjacent Visium H&E sections (same modality, lower resolution) | PNG / JPEG | `VisiumH&E_VisiumH&E_adjacent` |
| `Xenium_HE_LungTMA_same.py` | Xenium DAPI vs H&E, same section (lung TMA) | + TIFF | `Xenium_H&E_SameSection` |
| `Xenium_mIF_LymphNode_same.py` | mIF DAPI vs Xenium DAPI, same section (lymph node) | + TIFF / OME-TIFF | `Xenium_mIF_SameSection` |
| `Xenium_mIF_LymphNode_adjacent.py` | mIF DAPI vs Xenium DAPI, adjacent sections (lymph node) | + TIFF / OME-TIFF | `Xenium_mIF_adjacent` |

The scripts are self-contained and share the same pipeline; they differ only in the config block at the top (images, preprocessing, results folder) and in the loader (the Visium script has no TIFF support). TIFF vs. other formats is detected from the file's first bytes, not its extension. Multi-layer stacks: pick the layer with `IMAGE1_LAYER` / `IMAGE2_LAYER`.

## Pipeline functions (same in all scripts)

1. **Load** – `load_grayscale_image` → 8-bit grayscale; `load_ground_truth_points` (optional CSV with columns `KeyPoint,x,y` in the raw image's pixel space)
2. **Preprocess**, in this fixed order: `enhance_image` → `negate_image` → `rescale_image` → `rotate_image_cw` → `flip_image` (each toggled in the config; ground-truth points are carried along)
3. **SIFT** – `extract_sift_features`
4. **SuperPoint** – `load_superpoint_model`, `extract_superpoint_features` (downscales to ≤ 30 MP first if needed)
5. **Match** – `get_flann_matcher`, `match_features_flann` (Lowe ratio 0.8, one direction; RNG seeded with `cv2.setRNGSeed(42)` for reproducibility)
6. **Geometry** – `filter_with_ransac` (homography, threshold 5.0, confidence 0.95), `estimate_affine_transform`
7. **ASHLAR-style phase correlation** (if `RUN_ASHLAR`) – `ashlar_phase_correlation`: a port of `ashlar.utils.register` (whitening with a Laplacian / Laplacian-of-Gaussian → Hann window → unnormalized phase correlation → quadrant check → error value). Adapted to whole images: both are downscaled to fit `ASHLAR_MAX_PIXELS` and zero-padded to a common shape. ASHLAR's tile-based stitching is not used, since it needs unstitched tiles.
8. **VALIS** (if `RUN_VALIS`) – `register_valis`: writes both preprocessed images to disk, runs VALIS rigid registration with img1 as reference (non-rigid off) and reads the result back as an affine. Runs last because it shuts down its JVM afterwards.
9. **Accuracy** – `compute_tre` (mean / median / min / max error of the warped ground-truth points), for every pipeline
10. **Output** – `draw_keypoints`, `draw_matches`, `draw_gt_accuracy`, `create_flicker_gif`, `write_pipeline_summary` (SIFT / SuperPoint), `write_transform_summary` (ASHLAR / VALIS)

## Usage

Edit the config block at the top of a script (`IMG1_SRC` / `IMG2_SRC`, optional `GT1_CSV_PATH` / `GT2_CSV_PATH`, preprocessing toggles, `USE_SCRIPTED_SUPERPOINT`), then run `python <script>.py`.

The two additional pipelines have their own config block:

- `RUN_ASHLAR`, `RUN_VALIS` – switch each pipeline on or off
- `ASHLAR_FILTER_SIGMA` – whitening filter: 0 = plain Laplacian, > 0 = Laplacian-of-Gaussian (in pixels after the internal downscale); a small value above 0 is more robust on noisy or cross-modality pairs
- `ASHLAR_MAX_PIXELS`, `ASHLAR_UPSAMPLE_FACTOR` – memory budget and sub-pixel precision
- `VALIS_FEATURES` (`"default"` or `"superpoint_superglue"`), `VALIS_MAX_PROCESSED_DIM`, `VALIS_CHECK_REFLECTIONS`

Packages: `opencv-python`, `numpy`, `torch`, `pillow`, `tifffile` (not needed for the Visium script); for ASHLAR `scikit-image` (≥ 0.19) and `scipy`; for VALIS `valis-wsi` (has non-Python prerequisites such as Java – see the VALIS installation docs).

## Where results are saved

The results folder `benchmarking results` is created automatically, **next to the folder the scripts are in** (one level above the scripts):

```
Repo/
└──Python/
    ├──benchmarking/                      <- the scripts
    └──benchmarking results/
        └── <results folder from the table above>/
            ├── SIFT/<YYYYMMDD_HHMMSS>/
            ├── SuperPoint/<YYYYMMDD_HHMMSS>/
            ├── ASHLAR/<YYYYMMDD_HHMMSS>/        (if RUN_ASHLAR)
            └── VALIS/<YYYYMMDD_HHMMSS>/         (if RUN_VALIS)
```

Every run creates one timestamped folder per pipeline. `SIFT/` and `SuperPoint/` contain:

- `comparison_summary.txt` – keypoints, matches, RANSAC inliers, affine matrix, TRE
- `{sift,superpoint}_keypoints_img1.png` / `_img2.png` – detected keypoints (ground truth as yellow crosses)
- `{sift,superpoint}_matches.png` – matches (green = inliers, red = outliers)
- `{sift,superpoint}_gt_accuracy.png` – actual ground truth (green) vs. ground truth warped by the estimated affine (magenta)
- `{sift,superpoint}_flicker.gif` – overlay of both images after registration
- `results.npz` – homography, affine matrix, match counts
- ground-truth CSVs in the coordinate frame the detector used (only if ground truth is given)

`ASHLAR/` and `VALIS/` have no keypoints or matches, so they contain:

- `comparison_summary.txt` – method settings, registration time, affine matrix, TRE (ASHLAR also its error value: 0 = identical, larger = worse)
- `{ashlar,valis}_gt_accuracy.png` and `{ashlar,valis}_flicker.gif` – as above
- `results.npz` – affine matrix (ASHLAR also the error value)
- ground-truth CSVs in the preprocessed frame (only if ground truth is given)
- VALIS only: `valis/` with the input PNGs, VALIS's own outputs and `valis_error_estimates.csv`

If VALIS fails, the failure is written to its summary and the other pipelines' results are kept.

The results folders are per comparison type, not per tissue; runs are told apart by timestamp and the image paths in the summary.

## SuperPoint repo (rpautrat)

Only needed when `USE_SCRIPTED_SUPERPOINT = False`. Clone <https://github.com/rpautrat/SuperPoint.git> **directly into the project root** – the folder three levels above the scripts' folder (`project_root` in `setup_paths()`):

```
<project_root>/
├── SuperPoint/
│   ├── superpoint_pytorch.py
│   └── weights/superpoint_v6_from_tf.pth
├── Repo/
    └──Python/
        ├──benchmarking/                      <- the scripts
        └──benchmarking results/
```

- The script takes the first folder in the project root whose name contains "superpoint" (case-insensitive) – keep it the only one.
- Weights are looked for in `weights/superpoint_v6_from_tf.pth`, `weights/superpoint_v1.pth`, or the same file names in the repo root.
- Alternative without the repo: set `USE_SCRIPTED_SUPERPOINT = True` and `SCRIPTED_MODEL_PATH` to a TorchScript `.pt` file.