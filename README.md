# spatial-registration-benchmark

Benchmarking SIFT vs. SuperPoint for image registration of spatial omics /
pathology images (DAPI, H&E), as part of a Bachelor's thesis on scalable
spatial omics data integration.

## What this benchmarks

Given two grayscale images, each script:
1. Detects keypoints with **SIFT** and **SuperPoint** independently
2. Matches descriptors with **FLANN** (Lowe's ratio test)
3. Filters matches with **RANSAC** (homography-based)
4. Estimates an **affine transform** from the RANSAC inliers
5. Produces keypoint/match visualizations and a **flicker GIF** for visually
   judging registration quality
6. Writes a `comparison_summary.txt` + `results.npz` per run

## Scripts

| Script | Use case |
|---|---|
| `SIFTvsSuperPoint_2img.py` | Baseline two-image comparison. Two separate input images, no DAPI/H&E-specific preprocessing — just SIFT vs. SuperPoint on whatever you give it. |
| `SIFTvsSuperPoint_2imgHE_DAPI.py` | DAPI vs. H&E comparison. Same core pipeline, plus the full flexible preprocessing toggle set (negation, brightness enhancement, CLAHE, 90° pre-rotation), independently configurable per image, and TIFF support. |
| `SIFTvsSuperPoint_2img_ome.py` | DAPI vs. DAPI comparison, with **OME-TIFF** support — select a specific layer/channel per image (e.g. adjacent vs. same-slide). Preprocessing intentionally kept minimal: negation + optional brightness enhancement only, since both images are the same modality. |

All three share the same core pipeline (SIFT/SuperPoint → FLANN → RANSAC →
affine → visualize); they differ in **input handling** and **preprocessing
options** for their specific use case.

## Setup

Each script auto-detects its paths from its own location on disk
(`setup_paths()`), expecting a project layout like:

```
<project_root>/
├── data/
│   └── ...                          # your image folders
├── Results/Python/<script-name>/    # auto-created, timestamped run folders
└── Code/Python/Benchmarking/        # scripts live here
```

and expects a local clone of [rpautrat/SuperPoint](https://github.com/rpautrat/SuperPoint)
with weights !The Path has to be adjusted!

**Dependencies**: `opencv-python`, `numpy`, `torch`, `matplotlib`, `Pillow`,
`tifffile` (for OME-TIFF support).

## Configuration

Each script exposes its options as plain module-level constants near the
top (no CLI args) — edit and rerun:

- `IMAGE1_NAME` / `IMAGE2_NAME` — filenames to use (auto-picks first found
  in the folder if left `None`)
- `NEGATE_IMG1` / `NEGATE_IMG2` — invert intensities (DAPI: white
  background, black nuclei) (`2imgHE_DAPI` and `2img_ome` only —
  `SIFTvsSuperPoint_2img.py` has no preprocessing step at all)
- `ENHANCE_IMG1` / `ENHANCE_IMG2` / `ENHANCE_FACTOR` — brightness boost
  (PIL `ImageEnhance`), applied before negation (`2imgHE_DAPI` and
  `2img_ome`)
- `CLAHE_IMG1` / `CLAHE_IMG2` / `CLAHE_CLIP_LIMIT_*` / `CLAHE_TILE_GRID_SIZE_*`
  — local adaptive contrast enhancement (`2imgHE_DAPI` only)
- `ROTATE_IMG1_90CW` / `ROTATE_IMG2_90CW` — 90° clockwise pre-rotation
  (`2imgHE_DAPI` only)
- `IMAGE1_LAYER` / `IMAGE2_LAYER` — OME-TIFF layer/channel index
  (`2img_ome` only)
- `SUPERPOINT_MAX_PIXELS` — memory-safe cap on SuperPoint's input
  resolution (independent of source image resolution; SIFT always runs at
  full resolution)
- `VIS_MAX_DIM` — cap on saved keypoint/match/GIF image size (display
  only, doesn't affect detection/matching)

