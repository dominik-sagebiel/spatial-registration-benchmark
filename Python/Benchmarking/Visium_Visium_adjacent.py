"""
SIFT vs SuperPoint registration: same-modality, adjacent-section pairs
(e.g. two neighboring Visium H&E sections)
"""

import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ============================================================================
# CONFIG - Image sources (PNG/JPEG only)
# ============================================================================
IMG1_SRC = r"C:\Users\domin\Nextcloud\Uni\FU Ba. BioInf\11. Sem\Bachelorarbeit\data\Visium\DLPFC_Visium\DLPFC_Visium_Cropped\151675.png"
IMG2_SRC = r"C:\Users\domin\Nextcloud\Uni\FU Ba. BioInf\11. Sem\Bachelorarbeit\data\Visium\DLPFC_Visium\DLPFC_Visium_Cropped\151676.png"

# ============================================================================
# CONFIG - Ground truth (optional)
# ============================================================================

GT1_CSV_PATH = None
GT2_CSV_PATH = None

SHOW_GT_ON_VISUALIZATIONS = True

# ============================================================================
# CONFIG - SuperPoint model source
# ============================================================================

USE_SCRIPTED_SUPERPOINT = False
SCRIPTED_MODEL_PATH = r"C:\path\to\superpoint_scripted.pt"  

# ============================================================================
# CONFIG - Preprocessing toggles
# ============================================================================
NEGATE_IMG1 = False
NEGATE_IMG2 = False

ROTATE_IMG1 = False
ROTATE_ANGLE_IMG1 = 0.0
ROTATE_IMG2 = False
ROTATE_ANGLE_IMG2 = 0.0

ENHANCE_IMG1 = False
ENHANCE_IMG2 = False
ENHANCE_FACTOR = 3   # >1 brightens, <1 darkens, 1 = no change

RESCALE_IMG1 = False
RESCALE_FACTOR_IMG1 = 1.0
RESCALE_IMG2 = False
RESCALE_FACTOR_IMG2 = 1.0

FLIP_IMG1 = False
FLIP_IMG2 = False
FLIP_CODE = 1
'''
Flip/mirror. flip_code: 1 = horizontal (left-right), 0 = vertical
    (upside-down), -1 = both.
'''
# ===========================================================================
# CONFIG - Detector / matching parameters
# ===========================================================================
SIFT_MAX_KEYPOINTS = 20000

SUPERPOINT_MAX_KEYPOINTS = 20000
SUPERPOINT_NMS_RADIUS = 4
SUPERPOINT_DETECTION_THRESHOLD = 0.005
SUPERPOINT_REMOVE_BORDERS = 4
SUPERPOINT_MAX_PIXELS = 30_000_000
SUPERPOINT_STRIDE = 8

LOWES_RATIO = 0.8
RANSAC_THRESH = 5.0
RANSAC_CONFIDENCE = 0.95  

# ============================================================================
# CONFIG - Additional pipelines: ASHLAR-style phase correlation + VALIS
# ============================================================================
RUN_ASHLAR = True

ASHLAR_MAX_PIXELS = 16_000_000   # e.g. 4000x4000
# Whitening filter, like ashlar --filter-sigma: 0 = plain Laplacian, > 0 =
# Laplacian-of-Gaussian with this sigma (in working-resolution px, i.e.
# after the ASHLAR_MAX_PIXELS downscale). Raise it for noisy images.
ASHLAR_FILTER_SIGMA = 4
ASHLAR_UPSAMPLE_FACTOR = 10      # sub-pixel precision: shift resolved to 1/N px

RUN_VALIS = True
# "superpoint_superglue": VALIS's SuperPointFD + SuperGlueMatcher
# "default": DISK + LightGlue 
VALIS_FEATURES = "default"
# Longest side (px) VALIS downsamples to for rigid feature matching.
# None = VALIS's own default (small, ~512 px).
VALIS_MAX_PROCESSED_DIM = None
# Let VALIS test mirrored versions of the images (needed for FLIP tests).
VALIS_CHECK_REFLECTIONS = False

# ========================================================================
# CONFIG - Visualization-only
# ======================================================================
VIS_MAX_DIM = 1920
FLICKER_ALPHA = 0.5
FLICKER_N_CYCLES = 4
FLICKER_FRAME_MS = 1000


RESULTS_ROOT_NAME = "benchmarking results"

RESULTS_SUBFOLDER = "VisiumH&E_VisiumH&E_adjacent"

# ===================================================================
# PATH SETUP
# ===================================================================
def setup_paths():
    script_dir = Path(__file__).resolve().parent
    comparison_dir = script_dir.parent / RESULTS_ROOT_NAME / RESULTS_SUBFOLDER
    project_root = script_dir.parent.parent.parent
    return {'project_root': project_root, 'comparison_dir': comparison_dir}


paths = setup_paths()


def create_output_dir(pipeline_name):
    """Create a timestamped output directory under results/<folder>/<pipeline_name>/."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = paths['comparison_dir'] / pipeline_name / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[{pipeline_name}] Output directory: {output_dir}")
    return output_dir


# ============================================================================
# IMAGE LOADING (PNG / JPEG only)
# ============================================================================
def load_grayscale_image(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not load image from {path}")
    return image


# ============================================================================
# GROUND-TRUTH POINT LOADING / SAVING
# ============================================================================
def load_ground_truth_points(csv_path):
    """
    Load ground-truth keypoints from a CSV with columns "KeyPoint", "x", "y".

    Returns (ids, points) where points is an (N, 2) float64 array of (x, y)
    in the raw/as-loaded image's pixel space, sorted by KeyPoint id.
    """
    if csv_path is None:
        return None, None

    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Ground-truth CSV not found: {csv_path}")

    ids, xs, ys = [], [], []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row['KeyPoint'])
            xs.append(float(row['x']))
            ys.append(float(row['y']))

    order = sorted(range(len(ids)), key=lambda i: ids[i])
    ids = [ids[i] for i in order]
    points = np.array([[xs[i], ys[i]] for i in order], dtype=np.float64)

    print(f"   Loaded {len(points)} ground-truth points from {csv_path.name}")
    return ids, points


def save_points_csv(save_path, ids, points):
    """Save (ids, points) back out in the same "KeyPoint,x,y" format."""
    if points is None:
        return
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['KeyPoint', 'x', 'y'])
        for kp_id, (x, y) in zip(ids, points):
            writer.writerow([kp_id, x, y])


# ============================================================================
# PREPROCESSING
# ============================================================================
def negate_image(image, points=None):
    return cv2.bitwise_not(image), points


def enhance_image(image, factor=3, points=None):
    """
    Enhances image brightness (e.g. for low-contrast DAPI signal).

    Uses cv2.convertScaleAbs (image*factor, saturated to uint8) instead of
    PIL's ImageEnhance.Brightness. matches C++'s cv::Mat::convertTo(out,
    -1, factor, 0.0).
    """
    return cv2.convertScaleAbs(image, alpha=factor, beta=0), points


def rescale_image(image, factor, points=None):
    """
    Downscale by `factor` (0 < factor <= 1.0).

    gaussian anti-aliasing pre-filter runs before cv2.INTER_AREA
    subsampling to avoid new structure patterns to appear.
    """
    if factor == 1.0:
        return image, points
    if factor > 1.0:
        raise ValueError(f"Upscaling is not supported (factor={factor} > 1.0).")
    if factor <= 0.0:
        raise ValueError(f"factor must be > 0 (got {factor})")

    sigma = (1.0 / factor - 1.0) / 2.0
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma)
    resized = cv2.resize(blurred, None, fx=factor, fy=factor, interpolation=cv2.INTER_AREA)

    if points is not None:
        points = points * factor
    return resized, points


def rotate_image_cw(image, angle_deg, points=None):
    """
    Rotate clockwise by an angle, expanding the canvas
    so nothing is cropped. Points are mapped with the exact same 2x3
    affine matrix used for the image (via cv2.transform).
    """
    if angle_deg == 0:
        return image, points

    h, w = image.shape[:2]
    center = (w / 2, h / 2)

    # cv2.getRotationMatrix2D treats positive angles as counter-clockwise;
    # negate so this function's angle_deg means "clockwise" as named.
    M = cv2.getRotationMatrix2D(center, -angle_deg, 1.0)

    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(image, M, (new_w, new_h))

    if points is not None:
        pts = points.reshape(-1, 1, 2).astype(np.float64)
        points = cv2.transform(pts, M).reshape(-1, 2)
    return rotated, points


def flip_image(image, flip_code=1, points=None):
    """
    Point mapping uses continuous-coordinate convention (consistent
    with how SIFT/SuperPoint report subpixel keypoints): horizontal flip
    maps x -> w - x, vertical flip maps y -> h - y.
    """
    h, w = image.shape[:2]
    flipped = cv2.flip(image, flip_code)

    if points is not None:
        points = points.copy()
        if flip_code == 1:
            points[:, 0] = w - points[:, 0]
        elif flip_code == 0:
            points[:, 1] = h - points[:, 1]
        elif flip_code == -1:
            points[:, 0] = w - points[:, 0]
            points[:, 1] = h - points[:, 1]
        else:
            raise ValueError(f"Invalid flip_code {flip_code} (expected 1, 0, or -1)")
    return flipped, points


def describe_preprocessing(enhance_flag, negate_flag, rescale_flag, rotate_flag,
                            flip_flag, enhance_factor, rescale_sizes, rotate_angle):
    steps = []
    if enhance_flag:
        steps.append(f"brightness enhanced (factor={enhance_factor})")
    if negate_flag:
        steps.append("negated (inverted intensities)")
    if rescale_flag:
        (ow, oh), (nw, nh) = rescale_sizes
        steps.append(f"rescaled (original resolution {ow}x{oh}, downscaled to {nw}x{nh}), gaussian filter applied before")
    if rotate_flag:
        steps.append(f"rotated {rotate_angle}\u00b0 clockwise")
    if flip_flag:
        steps.append("flipped (mirrored)")
    return " -> ".join(steps) if steps else "none"


def describe_superpoint_downscale(sp_result):
    ow, oh = sp_result['sp_original_size']
    if sp_result['was_downscaled']:
        nw, nh = sp_result['sp_final_size']
        return f"downscaled for SuperPoint: {ow}x{oh} -> {nw}x{nh} (gaussian filter)"
    return f"not downscaled for SuperPoint (processed at {ow}x{oh})"


# ==========================================================================
# IMAGE RESIZING FOR SUPERPOINT (memory-safe, stride-aligned, GT-aware)
# ==========================================================================
def resize_for_superpoint(image, max_pixels=SUPERPOINT_MAX_PIXELS, stride=SUPERPOINT_STRIDE):
    """
    Downscales to <= max_pixels and crops the bottom/right
    to a multiple of stride (no coordinate shift).
    Returns (resized_image, scale, was_downscaled, original_size, final_size);
    scale = (new_w / w, new_h / h), (1.0, 1.0) if only cropped.
    """
    h, w = image.shape[:2]
    total_pixels = h * w
    original_size = (w, h)

    if total_pixels <= max_pixels:
        new_h = (h // stride) * stride
        new_w = (w // stride) * stride

        if new_h != h or new_w != w:
            image = image[:new_h, :new_w]
            print(f"   Cropped from {w}x{h} to {new_w}x{new_h} (bottom/right crop for stride {stride})")

        return image, (1.0, 1.0), False, original_size, (new_w, new_h)

    scale = np.sqrt(max_pixels / total_pixels)
    new_h = (int(h * scale) // stride) * stride
    new_w = (int(w * scale) // stride) * stride

    print(f"   Original: {w}x{h} ({total_pixels/1e6:.1f}MP)")
    print(f"   Resizing to: {new_w}x{new_h} ({new_w*new_h/1e6:.1f}MP)")

    sigma = (1.0 / scale - 1.0) / 2.0
    ksize = max(3, int(2 * round(3 * sigma) + 1))
    blurred = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma)
    resized = cv2.resize(blurred, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # actual per-axis factors (stride rounding makes them differ slightly)
    return resized, (new_w / w, new_h / h), True, original_size, (new_w, new_h)


def sp_to_original(points, scale):
    """
    Maps (N, 2) points from SuperPoint's input frame back to the frame
    resize_for_superpoint received. Exact inverse of cv2.resize, which
    aligns pixel centres: x_orig = (x_sp + 0.5) / scale_x - 0.5.
    """
    out = np.asarray(points, dtype=np.float64).copy()
    out[:, 0] = (out[:, 0] + 0.5) / scale[0] - 0.5
    out[:, 1] = (out[:, 1] + 0.5) / scale[1] - 0.5
    return out


# ========================================================
# SIFT Feature Extraction
# ========================================================
def extract_sift_features(image, max_keypoints=SIFT_MAX_KEYPOINTS):
    """Extract SIFT keypoints and descriptors."""
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    keypoints, descriptors = sift.detectAndCompute(image, None)

    if keypoints is None:
        return {'keypoints': np.empty((0, 2)), 'descriptors': None,
                'scores': np.empty((0,)), 'num_keypoints': 0}

    kp_array = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
    scores = np.array([kp.response for kp in keypoints])
    print(f"   Found {len(kp_array)} keypoints")

    return {'keypoints': kp_array, 'descriptors': descriptors,
            'scores': scores, 'num_keypoints': len(kp_array)}


# ============================================================================
# SuperPoint Setup
# ============================================================================
def find_eager_superpoint_repo_and_weights():
    """
    finds SuperPoint git clone + .pth weights. Only when
    USE_SCRIPTED_SUPERPOINT is False. SuperPoint repo is expected
    inside the project root (any folder there with "superpoint"
    in its name).
    """
    project_root = paths['project_root']
    repos_dir = project_root

    superpoint_path = None
    if repos_dir.exists():
        sp_candidates = [d for d in repos_dir.iterdir()
                          if d.is_dir() and 'superpoint' in d.name.lower()]
        if sp_candidates:
            superpoint_path = sp_candidates[0]
            print(f"   Found SuperPoint at: {superpoint_path}")

    if superpoint_path is None:
        raise FileNotFoundError(
            f"\n\u274c SuperPoint not found!\n"
            f"Expected location: {repos_dir}/SuperPoint or similar\n"
            f"Please clone it:\n"
            f"  cd {repos_dir}\n"
            f"  git clone https://github.com/rpautrat/SuperPoint.git\n"
        )

    weights_path = None
    weight_candidates = [
        superpoint_path / 'weights' / 'superpoint_v6_from_tf.pth',
        superpoint_path / 'weights' / 'superpoint_v1.pth',
        superpoint_path / 'superpoint_v6_from_tf.pth',
        superpoint_path / 'superpoint_v1.pth'
    ]
    for candidate in weight_candidates:
        if candidate.exists():
            weights_path = candidate
            print(f"   Found weights at: {weights_path}")
            break

    if weights_path is None:
        raise FileNotFoundError(
            f"\n\u274c SuperPoint weights not found!\n"
            f"Searched in:\n" +
            "\n".join(f"  - {p}" for p in weight_candidates) +
            f"\n\nPlease ensure weights file is in the SuperPoint/weights/ directory"
        )

    return superpoint_path, weights_path


def load_superpoint_model():
    """Loads SuperPoint either as the scripted TorchScript .pt file or the eager PyTorch model + .pth weights."""
    if USE_SCRIPTED_SUPERPOINT:
        print(f"   Loading scripted TorchScript model: {SCRIPTED_MODEL_PATH}")
        model = torch.jit.load(SCRIPTED_MODEL_PATH, map_location='cpu')
        model.eval()
        return model

    superpoint_path, weights_path = find_eager_superpoint_repo_and_weights()
    sys.path.insert(0, str(superpoint_path))
    from superpoint_pytorch import SuperPoint  

    model = SuperPoint(
        nms_radius=SUPERPOINT_NMS_RADIUS,
        max_num_keypoints=SUPERPOINT_MAX_KEYPOINTS,
        detection_threshold=SUPERPOINT_DETECTION_THRESHOLD,
        remove_borders=SUPERPOINT_REMOVE_BORDERS,
        descriptor_dim=256,
        channels=[64, 64, 128, 128, 256]
    )
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print(f"   Loaded eager PyTorch model, weights: {weights_path}")
    return model


def extract_superpoint_features(model, image, gt_points=None, max_pixels=SUPERPOINT_MAX_PIXELS):
    """
    Runs SuperPoint on `image` (via resize_for_superpoint) and maps the
    keypoints back onto `image`'s ("preprocessed") frame. `gt_points` are
    already in that frame and are passed through unchanged.
    """
    image_sp, sp_scale, was_downscaled, sp_original_size, sp_final_size = \
        resize_for_superpoint(image, max_pixels=max_pixels, stride=SUPERPOINT_STRIDE)

    img_normalized = image_sp.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(img_normalized).float().unsqueeze(0).unsqueeze(0)

    start_time = time.time()
    with torch.no_grad():
        output = model({"image": input_tensor})
    inference_time = time.time() - start_time

    keypoints = output['keypoints'][0].cpu().numpy()
    descriptors = output['descriptors'][0].cpu().numpy()
    scores = output['keypoint_scores'][0].cpu().numpy()

    # only a downscale changes coordinates; the bottom/right crop does not
    if was_downscaled:
        keypoints = sp_to_original(keypoints, sp_scale).astype(np.float32)
        print(f"   Mapped keypoints back to preprocessed frame: {sp_original_size[0]}x{sp_original_size[1]}")

    print(f"   SuperPoint inference: {inference_time*1000:.1f} ms")
    print(f"   Found {len(keypoints)} keypoints")

    return {
        'keypoints': keypoints, 'descriptors': descriptors, 'scores': scores,
        'num_keypoints': len(keypoints), 'time': inference_time,
        'was_downscaled': was_downscaled, 'sp_original_size': sp_original_size,
        'sp_final_size': sp_final_size, 'sp_scale': sp_scale,
        'gt_points_preprocessed_frame': gt_points,
    }


# =======================================================================
# Matching / RANSAC / Affine
# ======================================================================
def get_flann_matcher():
    """Get FLANN matcher."""
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


def match_features_flann(desc1, desc2, flann_matcher, lowes_ratio=LOWES_RATIO):
    """Match features using FLANN with Lowe's ratio test (one direction only - desc1 -> desc2)."""
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    if len(desc1) < 2 or len(desc2) < 2:
        return []

    desc1 = desc1.astype(np.float32)
    desc2 = desc2.astype(np.float32)

    start_time = time.time()
    matches = flann_matcher.knnMatch(desc1, desc2, k=2)
    matching_time = time.time() - start_time

    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < lowes_ratio * n.distance:
                good_matches.append(m)

    print(f"   FLANN matching: {matching_time*1000:.1f} ms")
    print(f"   Found {len(good_matches)} good matches")
    return good_matches


def filter_with_ransac(matches, kp1, kp2, ransac_thresh=RANSAC_THRESH, confidence=RANSAC_CONFIDENCE):
    """Filter matches using RANSAC to find geometrically consistent ones."""
    if len(matches) < 4:
        print(f"   RANSAC: Need at least 4 matches, have {len(matches)}")
        return [], None, None

    src_pts = np.float32([kp1[m.queryIdx] for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx] for m in matches]).reshape(-1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh, confidence=confidence)

    if H is not None:
        inliers = [matches[i] for i in range(len(matches)) if mask[i]]
        print(f"   RANSAC: {len(inliers)}/{len(matches)} inliers ({len(inliers)/len(matches)*100:.1f}%)")
        return inliers, H, mask
    print("   RANSAC: Failed to find homography")
    return [], None, None


def estimate_affine_transform(matches, kp1, kp2, ransac_thresh=RANSAC_THRESH):
    """Estimate a 2D affine transformation (img1 -> img2) from inlier matches."""
    if len(matches) < 3:
        print(f"   Affine estimation: need at least 3 matches, have {len(matches)}")
        return None

    src_pts = np.float32([kp1[m.queryIdx] for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx] for m in matches]).reshape(-1, 2)

    A_fwd, inlier_mask = cv2.estimateAffine2D(
        src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh
    )
    if A_fwd is None:
        print("   Affine estimation: failed to find a transform")
        return None

    n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else len(matches)
    print(f"   Affine matrix estimated from {n_inliers}/{len(matches)} matches")
    return A_fwd


# ============================================================================
# ASHLAR-style phase correlation (translation only)
# ============================================================================
_ASHLAR_LAPLACE_KERNEL = np.array([[0, -1, 0],
                                   [-1, 4, -1],
                                   [0, -1, 0]], dtype=np.float32)


def _ashlar_whiten(img, sigma):
    """
    Port of ashlar.utils.whiten: converts to float32 (0..1), then keeps only
    local intensity changes, plain Laplacian at sigma = 0,
    Laplacian-of-Gaussian at sigma > 0. Flat areas and gradual background
    come out as ~0.
    """
    import scipy.ndimage
    from skimage import img_as_float32

    img = img_as_float32(img)
    if sigma == 0:
        return scipy.ndimage.convolve(img, _ASHLAR_LAPLACE_KERNEL)
    return scipy.ndimage.gaussian_laplace(img, sigma)


def _ashlar_window(img):
    """Port of ashlar.utils.window: 2D Hann window, fades the image to 0 towards its borders."""
    wy = np.hanning(img.shape[0]).astype(np.float32)
    wx = np.hanning(img.shape[1]).astype(np.float32)
    return img * np.outer(wy, wx)


def ashlar_phase_correlation(img1, img2, max_pixels=ASHLAR_MAX_PIXELS,
                             filter_sigma=ASHLAR_FILTER_SIGMA,
                             upsample_factor=ASHLAR_UPSAMPLE_FACTOR):
    """
    Steps taken from ASHLAR, in order:
      1. whiten both images (Laplacian / Laplacian-of-Gaussian)
      2. multiply by a 2D Hann window
      3. skimage phase_cross_correlation with normalization=None
      4. quadrant check: the FFT treats the images as periodic, so each
         axis' shift is ambiguous by one image length; all four candidates
         are tested by direct correlation and the best one is kept
      5. error = -log(correlation / total amplitude) at that shift
         (0 = identical, larger = worse, inf = no correlation)
    """
    try:
        import scipy.ndimage
        from skimage.registration import phase_cross_correlation
    except ImportError as e:
        raise ImportError("RUN_ASHLAR needs scikit-image (>= 0.19) and scipy: "
                          "pip install scikit-image scipy") from e

    canvas_h = max(img1.shape[0], img2.shape[0])
    canvas_w = max(img1.shape[1], img2.shape[1])
    scale = min(1.0, float(np.sqrt(max_pixels / (canvas_h * canvas_w))))

    a, _ = rescale_image(img1, scale)
    b, _ = rescale_image(img2, scale)

    work_h = max(a.shape[0], b.shape[0])
    work_w = max(a.shape[1], b.shape[1])

    def pad_to_canvas(img):
        out = np.zeros((work_h, work_w), dtype=np.float32)
        out[:img.shape[0], :img.shape[1]] = img
        return out

    start_time = time.time()

    # 1 + 2: whiten -> (zero-pad to common shape) -> Hann window
    a = _ashlar_window(pad_to_canvas(_ashlar_whiten(a, filter_sigma)))
    b = _ashlar_window(pad_to_canvas(_ashlar_whiten(b, filter_sigma)))

    # 3: returns the (dy, dx) shift that moves img2 onto img1: a point at q
    # in img2 lies at q + shift in img1  ->  img1 -> img2 is x2 = x1 - shift.
    shift = phase_cross_correlation(a, b, upsample_factor=upsample_factor,
                                    normalization=None)[0]

    # 4: quadrant check
    shape = np.array(a.shape)
    shift_pos = (shift + shape) % shape
    shift_neg = shift_pos - shape
    candidates = [(sy, sx) for sy in (shift_pos[0], shift_neg[0])
                  for sx in (shift_pos[1], shift_neg[1])]
    correlations = [float(np.abs(np.sum(a * scipy.ndimage.shift(b, s, order=0))))
                    for s in candidates]
    best = int(np.argmax(correlations))
    shift = candidates[best]
    correlation = correlations[best]

    # 5: error
    total_amplitude = float(np.linalg.norm(a) * np.linalg.norm(b))
    if correlation > 0 and total_amplitude > 0:
        error = float(-np.log(correlation / total_amplitude))
    else:
        error = float('inf')

    elapsed = time.time() - start_time

    dy, dx = shift[0] / scale, shift[1] / scale
    A_fwd = np.array([[1.0, 0.0, -dx],
                      [0.0, 1.0, -dy]], dtype=np.float64)

    print(f"   Working size: {work_w}x{work_h} (scale {scale:.3f})")
    print(f"   Whitening + window + phase correlation + quadrant check: {elapsed*1000:.1f} ms")
    print(f"   Shift img1 -> img2 (preprocessed-frame px): dx={-dx:.2f}, dy={-dy:.2f}")
    print(f"   ASHLAR error: {error:.4f}")

    return {
        'A': A_fwd,
        'error': error,
        'time': elapsed,
        'scale': scale,
        'working_size': (work_w, work_h),
    }


# =================================================================
# VALIS (rigid registration)
# ================================================================
def register_valis(img1, img2, output_dir, features=VALIS_FEATURES,
                   max_processed_dim=VALIS_MAX_PROCESSED_DIM,
                   check_reflections=VALIS_CHECK_REFLECTIONS):
    """
    Run VALIS's rigid registration on the two (already preprocessed) images
    and express the result as a 2x3 img1 -> img2 affine.
    """
    try:
        from valis import registration, feature_detectors, feature_matcher
    except ImportError as e:
        raise ImportError("RUN_VALIS needs VALIS: pip install valis-wsi") from e

    valis_dir = output_dir / "valis"
    src_dir = valis_dir / "input"
    dst_dir = valis_dir / "results"
    src_dir.mkdir(parents=True, exist_ok=True)
    dst_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(src_dir / "img1.png"), img1)
    cv2.imwrite(str(src_dir / "img2.png"), img2)

    kwargs = dict(
        reference_img_f="img1.png",
        align_to_reference=True,
        non_rigid_registrar_cls=None,
        check_for_reflections=check_reflections,
    )
    if max_processed_dim is not None:
        kwargs['max_processed_image_dim_px'] = max_processed_dim
    if features == "superpoint_superglue":
        kwargs['feature_detector_cls'] = feature_detectors.SuperPointFD
        kwargs['matcher'] = feature_matcher.SuperGlueMatcher()
    elif features != "default":
        raise ValueError(f"Unknown VALIS_FEATURES: {features!r}")

    try:
        start_time = time.time()
        registrar = registration.Valis(str(src_dir), str(dst_dir), **kwargs)
        if not hasattr(registrar, "non_rigid_reg_kwargs"):
            registrar.non_rigid_reg_kwargs = {}

        rigid_registrar, _, error_df = registrar.register()
        elapsed = time.time() - start_time

        if rigid_registrar is None:
            print("   VALIS: registration failed (see VALIS output above)")
            return {'A': None, 'time': elapsed, 'fit_residual': None, 'error': 'VALIS returned no rigid registrar'}

        if error_df is not None:
            error_df.to_csv(valis_dir / "valis_error_estimates.csv", index=False)

        slide1 = registrar.get_slide("img1")
        slide2 = registrar.get_slide("img2")

        h, w = img1.shape[:2]
        gx, gy = np.meshgrid(np.linspace(0, w - 1, 5), np.linspace(0, h - 1, 5))
        pts1 = np.column_stack([gx.ravel(), gy.ravel()])
        pts2 = np.asarray(slide1.warp_xy_from_to(pts1, slide2, non_rigid=False), dtype=np.float64)

        # Least-squares affine fit
        X = np.column_stack([pts1, np.ones(len(pts1))])
        A_T, _, _, _ = np.linalg.lstsq(X, pts2, rcond=None)
        A_fwd = A_T.T
        fit_residual = float(np.abs(X @ A_T - pts2).max())
    except Exception as e:
        print(f"   VALIS: registration failed with {type(e).__name__}: {e}")
        return {'A': None, 'time': None, 'fit_residual': None, 'error': f"{type(e).__name__}: {e}"}
    finally:
        try:
            registration.kill_jvm()
        except Exception:
            pass

    print(f"   VALIS registration: {elapsed:.1f} s")
    print(f"   Affine matrix (img1 -> img2):\n{A_fwd}")
    print(f"   Affine fit max residual: {fit_residual:.3f} px (should be ~0 for rigid)")

    return {'A': A_fwd, 'time': elapsed, 'fit_residual': fit_residual, 'error': None}


# ============================================================================
# GROUND-TRUTH ACCURACY (Target Registration Error)
# ============================================================================
def compute_tre(gt1_ids, gt1_points, gt2_ids, gt2_points, A_fwd):
   
    if gt1_points is None or gt2_points is None or A_fwd is None:
        return None, None, None, None, None

    gt2_by_id = {kp_id: pt for kp_id, pt in zip(gt2_ids, gt2_points)}
    common_ids = [kp_id for kp_id in gt1_ids if kp_id in gt2_by_id]
    if not common_ids:
        return None, None, None, None, None

    gt1_by_id = {kp_id: pt for kp_id, pt in zip(gt1_ids, gt1_points)}
    src = np.array([gt1_by_id[kp_id] for kp_id in common_ids], dtype=np.float64)
    dst = np.array([gt2_by_id[kp_id] for kp_id in common_ids], dtype=np.float64)

    src_h = np.hstack([src, np.ones((len(src), 1))])
    warped = (A_fwd @ src_h.T).T

    errors = np.linalg.norm(warped - dst, axis=1)
    return (len(common_ids), float(np.mean(errors)), float(np.median(errors)),
            float(np.min(errors)), float(np.max(errors)))


# ============================================================================
# Visualization
# ============================================================================
def resize_for_display(image, max_dim=VIS_MAX_DIM):
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image, scale


def _scaled_marker_params(w, h):
    ref = min(w, h)
    radius = max(3, int(round(ref * 0.0025)))
    circle_thickness = 1
    line_thickness = max(2, radius // 2)
    font_scale = max(1.0, ref / 1200)
    font_thickness = max(2, int(round(font_scale * 2)))
    return radius, circle_thickness, line_thickness, font_scale, font_thickness


def _wrap_text_to_width(text, max_width, font_scale, font_thickness):
    words = text.split(' ')
    lines, current = [], ""
    for word in words:
        candidate = f"{current} {word}".strip() if current else word
        (w, _), _ = cv2.getTextSize(candidate, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        if w <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_label_bar(img, text, font_scale, font_thickness):
    margin = 10
    max_text_width = img.shape[1] - 2 * margin
    lines = _wrap_text_to_width(text, max_text_width, font_scale, font_thickness)

    (_, text_h), baseline = cv2.getTextSize(lines[0], cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    line_spacing = text_h + baseline + int(8 * font_scale)
    top_pad = int(12 * font_scale)
    bar_h = top_pad + line_spacing * len(lines)

    cv2.rectangle(img, (0, 0), (img.shape[1], bar_h), (0, 0, 0), -1)
    y = top_pad + text_h
    for line in lines:
        cv2.putText(img, line, (margin, y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        y += line_spacing
    return img


def draw_keypoints(image, keypoints, title, save_path, gt_points=None, color=(0, 255, 0)):
    """Draws detected keypoints (circles) and, if given, GT points (yellow crosses)."""
    display_img, scale = resize_for_display(image)
    img_color = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    h, w = img_color.shape[:2]
    radius, circle_thickness, _, font_scale, font_thickness = _scaled_marker_params(w, h)

    for pt in keypoints:
        x, y = int(round(pt[0] * scale)), int(round(pt[1] * scale))
        cv2.circle(img_color, (x, y), radius, color, circle_thickness)

    if gt_points is not None and SHOW_GT_ON_VISUALIZATIONS:
        cross = radius + 2
        for pt in gt_points:
            x, y = int(round(pt[0] * scale)), int(round(pt[1] * scale))
            cv2.line(img_color, (x - cross, y), (x + cross, y), (0, 255, 255), circle_thickness)
            cv2.line(img_color, (x, y - cross), (x, y + cross), (0, 255, 255), circle_thickness)

    _draw_label_bar(img_color, title, font_scale, font_thickness)
    cv2.imwrite(str(save_path), img_color)


def draw_matches(img1, img2, kp1, kp2, matches, inliers, title, save_path):
    disp1, scale1 = resize_for_display(img1)
    disp2, scale2 = resize_for_display(img2)

    h1, w1 = disp1.shape[:2]
    h2, w2 = disp2.shape[:2]
    match_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    match_img[:h1, :w1] = cv2.cvtColor(disp1, cv2.COLOR_GRAY2BGR)
    match_img[:h2, w1:w1 + w2] = cv2.cvtColor(disp2, cv2.COLOR_GRAY2BGR)

    radius, circle_thickness, line_thickness, font_scale, font_thickness = _scaled_marker_params(
        w1 + w2, max(h1, h2))

    inlier_set = {m.queryIdx for m in inliers}
    for m in matches:
        pt1 = (int(round(kp1[m.queryIdx][0] * scale1)), int(round(kp1[m.queryIdx][1] * scale1)))
        pt2 = (int(round(kp2[m.trainIdx][0] * scale2 + w1)), int(round(kp2[m.trainIdx][1] * scale2)))
        if m.queryIdx in inlier_set:
            color, thickness = (0, 255, 0), line_thickness
        else:
            color, thickness = (0, 0, 255), max(1, line_thickness // 2)
        cv2.line(match_img, pt1, pt2, color, thickness)
        cv2.circle(match_img, pt1, radius, color, -1)
        cv2.circle(match_img, pt2, radius, color, -1)

    _draw_label_bar(match_img, f'{title} - {len(inliers)} inliers / {len(matches)} total',
                     font_scale, font_thickness)
    cv2.imwrite(str(save_path), match_img)


def draw_gt_accuracy(img2, gt1_ids, gt1_points, gt2_ids, gt2_points, A_fwd, title, save_path):

    display_img, scale = resize_for_display(img2)
    img_color = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    h, w = img_color.shape[:2]
    radius, circle_thickness, line_thickness, font_scale, font_thickness = _scaled_marker_params(w, h)

    if gt1_points is None or gt2_points is None or A_fwd is None:
        _draw_label_bar(img_color, f"{title} - ground truth unavailable", font_scale, font_thickness)
        cv2.imwrite(str(save_path), img_color)
        return

    gt2_by_id = {kp_id: pt for kp_id, pt in zip(gt2_ids, gt2_points)}
    common_ids = [kp_id for kp_id in gt1_ids if kp_id in gt2_by_id]
    if not common_ids:
        _draw_label_bar(img_color, f"{title} - no matching GT ids", font_scale, font_thickness)
        cv2.imwrite(str(save_path), img_color)
        return

    gt1_by_id = {kp_id: pt for kp_id, pt in zip(gt1_ids, gt1_points)}
    src = np.array([gt1_by_id[kp_id] for kp_id in common_ids], dtype=np.float64)
    dst = np.array([gt2_by_id[kp_id] for kp_id in common_ids], dtype=np.float64)
    src_h = np.hstack([src, np.ones((len(src), 1))])
    warped = (A_fwd @ src_h.T).T

    cross = radius + 2
    for (wx, wy), (ax, ay) in zip(warped, dst):
        wx_d, wy_d = int(round(wx * scale)), int(round(wy * scale))
        ax_d, ay_d = int(round(ax * scale)), int(round(ay * scale))
        cv2.line(img_color, (wx_d, wy_d), (ax_d, ay_d), (0, 255, 255), max(1, line_thickness // 2))
        cv2.circle(img_color, (ax_d, ay_d), radius, (0, 255, 0), circle_thickness)       # actual GT: green circle
        cv2.line(img_color, (wx_d - cross, wy_d), (wx_d + cross, wy_d), (255, 0, 255), circle_thickness)
        cv2.line(img_color, (wx_d, wy_d - cross), (wx_d, wy_d + cross), (255, 0, 255), circle_thickness)  # predicted: magenta cross

    errors = np.linalg.norm(warped - dst, axis=1)
    mean_err = float(np.mean(errors))
    _draw_label_bar(img_color,
                     f"{title} - mean TRE {mean_err:.2f}px, n={len(common_ids)} "
                     f"(green=actual GT, magenta=predicted from img1)",
                     font_scale, font_thickness)
    cv2.imwrite(str(save_path), img_color)


def create_flicker_gif(img1, img2, A_fwd, save_path, alpha=FLICKER_ALPHA,
                        n_cycles=FLICKER_N_CYCLES, frame_duration_ms=FLICKER_FRAME_MS):
    if A_fwd is None:
        print("   Skipping flicker GIF: no affine transform available")
        return

    h1, w1 = img1.shape[:2]
    A_inv = cv2.invertAffineTransform(A_fwd)
    img2_warped = cv2.warpAffine(img2, A_inv, (w1, h1))

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2_rgb = cv2.cvtColor(img2_warped, cv2.COLOR_GRAY2RGB)
    blended = cv2.addWeighted(img1_rgb, 1 - alpha, img2_rgb, alpha, 0)

    img1_disp, _ = resize_for_display(img1_rgb)
    blended_disp, _ = resize_for_display(blended)

    frame_off = Image.fromarray(img1_disp)
    frame_on = Image.fromarray(blended_disp)
    frames = []
    for _ in range(n_cycles):
        frames.append(frame_off)
        frames.append(frame_on)

    frames[0].save(save_path, save_all=True, append_images=frames[1:],
                    duration=frame_duration_ms, loop=0)
    print(f"   Saved flicker GIF: {save_path}")


def write_pipeline_summary(output_dir, pipeline_name, image1_path, image2_path,
                            desc1, desc2, result1, result2, num_matches, inliers,
                            A_fwd, tre_stats, extra_lines=None):
    tre_n, tre_mean, tre_median, tre_min, tre_max = tre_stats
    with open(output_dir / "comparison_summary.txt", 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"{pipeline_name} Registration Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Image 1: {image1_path}\n  Preprocessing applied: {desc1}\n")
        f.write(f"Image 2: {image2_path}\n  Preprocessing applied: {desc2}\n\n")
        if extra_lines:
            for line in extra_lines:
                f.write(line + "\n")
            f.write("\n")
        f.write(f"Keypoints (image 1): {result1['num_keypoints']}\n")
        f.write(f"Keypoints (image 2): {result2['num_keypoints']}\n")
        f.write(f"FLANN matches: {num_matches}\n")
        f.write(f"RANSAC inliers: {len(inliers)} ({len(inliers)/max(num_matches,1)*100:.1f}%)\n")
        f.write(f"Affine matrix (img1 -> img2):\n{A_fwd}\n\n")
        f.write("GROUND-TRUTH TARGET REGISTRATION ERROR (TRE, pixels)\n" + "-" * 40 + "\n")
        if tre_mean is not None:
            f.write(f"mean={tre_mean:.3f}  median={tre_median:.3f}  min={tre_min:.3f}  "
                    f"max={tre_max:.3f}  (n={tre_n})\n")
        else:
            f.write("n/a (no matching GT points or no transform found)\n")


def write_transform_summary(output_dir, pipeline_name, image1_path, image2_path,
                             desc1, desc2, A_fwd, tre_stats, extra_lines=None):
    
    tre_n, tre_mean, tre_median, tre_min, tre_max = tre_stats
    with open(output_dir / "comparison_summary.txt", 'w') as f:
        f.write("=" * 70 + "\n")
        f.write(f"{pipeline_name} Registration Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Image 1: {image1_path}\n  Preprocessing applied: {desc1}\n")
        f.write(f"Image 2: {image2_path}\n  Preprocessing applied: {desc2}\n\n")
        if extra_lines:
            for line in extra_lines:
                f.write(line + "\n")
            f.write("\n")
        f.write(f"Affine matrix (img1 -> img2):\n{A_fwd}\n\n")
        f.write("GROUND-TRUTH TARGET REGISTRATION ERROR (TRE, pixels)\n" + "-" * 40 + "\n")
        if tre_mean is not None:
            f.write(f"mean={tre_mean:.3f}  median={tre_median:.3f}  min={tre_min:.3f}  "
                    f"max={tre_max:.3f}  (n={tre_n})\n")
        else:
            f.write("n/a (no matching GT points or no transform found)\n")


# ============================================================================
# Main
# ============================================================================
def main():
    cv2.setRNGSeed(42)

    image1_path = Path(IMG1_SRC)
    image2_path = Path(IMG2_SRC)
    if not image1_path.exists():
        raise FileNotFoundError(f"IMG1_SRC not found: {image1_path}")
    if not image2_path.exists():
        raise FileNotFoundError(f"IMG2_SRC not found: {image2_path}")

    print("=" * 70)
    print("SIFT vs SuperPoint vs VALIS vs ASHLAR Registration Comparison")
    print("=" * 70)
    print(f"Image 1: {image1_path.name}")
    print(f"Image 2: {image2_path.name}")

    print("\n[1] Loading images...")
    img1 = load_grayscale_image(image1_path)
    img2 = load_grayscale_image(image2_path)
    print(f"[1] img1: {img1.shape}, img2: {img2.shape}")

    print("[1] Loading ground-truth points...")
    gt1_ids, gt1_raw = load_ground_truth_points(GT1_CSV_PATH)
    gt2_ids, gt2_raw = load_ground_truth_points(GT2_CSV_PATH)

    # ---- Preprocessing: enhance -> negate -> rescale -> rotate -> flip ----
    gt1_points, gt2_points = gt1_raw, gt2_raw
    rescale_sizes_1 = rescale_sizes_2 = None

    if ENHANCE_IMG1:
        img1, gt1_points = enhance_image(img1, factor=ENHANCE_FACTOR, points=gt1_points)
    if ENHANCE_IMG2:
        img2, gt2_points = enhance_image(img2, factor=ENHANCE_FACTOR, points=gt2_points)

    if NEGATE_IMG1:
        img1, gt1_points = negate_image(img1, points=gt1_points)
    if NEGATE_IMG2:
        img2, gt2_points = negate_image(img2, points=gt2_points)

    if RESCALE_IMG1:
        orig_wh_1 = (img1.shape[1], img1.shape[0])
        img1, gt1_points = rescale_image(img1, RESCALE_FACTOR_IMG1, points=gt1_points)
        rescale_sizes_1 = (orig_wh_1, (img1.shape[1], img1.shape[0]))
    if RESCALE_IMG2:
        orig_wh_2 = (img2.shape[1], img2.shape[0])
        img2, gt2_points = rescale_image(img2, RESCALE_FACTOR_IMG2, points=gt2_points)
        rescale_sizes_2 = (orig_wh_2, (img2.shape[1], img2.shape[0]))

    if ROTATE_IMG1:
        img1, gt1_points = rotate_image_cw(img1, ROTATE_ANGLE_IMG1, points=gt1_points)
    if ROTATE_IMG2:
        img2, gt2_points = rotate_image_cw(img2, ROTATE_ANGLE_IMG2, points=gt2_points)

    if FLIP_IMG1:
        img1, gt1_points = flip_image(img1, flip_code=FLIP_CODE, points=gt1_points)
    if FLIP_IMG2:
        img2, gt2_points = flip_image(img2, flip_code=FLIP_CODE, points=gt2_points)

    desc1 = describe_preprocessing(ENHANCE_IMG1, NEGATE_IMG1, RESCALE_IMG1, ROTATE_IMG1, FLIP_IMG1,
                                    ENHANCE_FACTOR, rescale_sizes_1, ROTATE_ANGLE_IMG1)
    desc2 = describe_preprocessing(ENHANCE_IMG2, NEGATE_IMG2, RESCALE_IMG2, ROTATE_IMG2, FLIP_IMG2,
                                    ENHANCE_FACTOR, rescale_sizes_2, ROTATE_ANGLE_IMG2)

    flann = get_flann_matcher()

    # ========================================================================
    # SIFT
    # ========================================================================
    print("\n" + "=" * 70)
    print("SIFT REGISTRATION")
    print("=" * 70)
    sift_output_dir = create_output_dir("SIFT")

    print("\n[2a] Extracting SIFT features...")
    sift_1 = extract_sift_features(img1)
    sift_2 = extract_sift_features(img2)

    print("\n[2b] FLANN matching...")
    sift_matches = match_features_flann(sift_1['descriptors'], sift_2['descriptors'], flann)

    print("\n[2c] RANSAC filtering...")
    sift_inliers, sift_H, _ = filter_with_ransac(sift_matches, sift_1['keypoints'], sift_2['keypoints'])

    print("\n[2d] Estimating affine transform...")
    sift_A = estimate_affine_transform(sift_inliers, sift_1['keypoints'], sift_2['keypoints'])

    sift_tre = compute_tre(gt1_ids, gt1_points, gt2_ids, gt2_points, sift_A)

    print("\n[2e] Generating visualizations...")
    draw_keypoints(img1, sift_1['keypoints'],
                    f"SIFT - img1 - {sift_1['num_keypoints']} kpts - {img1.shape[1]}x{img1.shape[0]}",
                    sift_output_dir / "sift_keypoints_img1.png", gt_points=gt1_points)
    draw_keypoints(img2, sift_2['keypoints'],
                    f"SIFT - img2 - {sift_2['num_keypoints']} kpts - {img2.shape[1]}x{img2.shape[0]}",
                    sift_output_dir / "sift_keypoints_img2.png", gt_points=gt2_points)
    draw_matches(img1, img2, sift_1['keypoints'], sift_2['keypoints'],
                 sift_matches, sift_inliers, 'SIFT', sift_output_dir / "sift_matches.png")
    draw_gt_accuracy(img2, gt1_ids, gt1_points, gt2_ids, gt2_points, sift_A,
                      'SIFT', sift_output_dir / "sift_gt_accuracy.png")
    create_flicker_gif(img1, img2, sift_A, sift_output_dir / "sift_flicker.gif")

    np.savez(sift_output_dir / "results.npz", H=sift_H, A=sift_A,
             num_matches=len(sift_matches), num_inliers=len(sift_inliers))
    save_points_csv(sift_output_dir / "gt1_preprocessed.csv", gt1_ids, gt1_points)
    save_points_csv(sift_output_dir / "gt2_preprocessed.csv", gt2_ids, gt2_points)
    write_pipeline_summary(sift_output_dir, "SIFT", image1_path, image2_path, desc1, desc2,
                            sift_1, sift_2, len(sift_matches), sift_inliers, sift_A, sift_tre)
    print(f"[SIFT] Results saved to: {sift_output_dir}")

    # ========================================================================
    # SuperPoint
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUPERPOINT REGISTRATION")
    print("=" * 70)
    sp_output_dir = create_output_dir("SuperPoint")

    print("\n[3a] Loading SuperPoint model...")
    sp_model = load_superpoint_model()

    print("\n[3b] Extracting SuperPoint features...")
    sp_1 = extract_superpoint_features(sp_model, img1, gt_points=gt1_points)
    sp_2 = extract_superpoint_features(sp_model, img2, gt_points=gt2_points)

    print("\n[3c] FLANN matching...")
    sp_matches = match_features_flann(sp_1['descriptors'], sp_2['descriptors'], flann)

    print("\n[3d] RANSAC filtering...")
    sp_inliers, sp_H, _ = filter_with_ransac(sp_matches, sp_1['keypoints'], sp_2['keypoints'])

    print("\n[3e] Estimating affine transform...")
    sp_A = estimate_affine_transform(sp_inliers, sp_1['keypoints'], sp_2['keypoints'])

    # Use GT already scaled back to the common "preprocessed" frame, so
    # SIFT and SuperPoint TRE are directly comparable.
    sp_tre = compute_tre(gt1_ids, sp_1['gt_points_preprocessed_frame'],
                          gt2_ids, sp_2['gt_points_preprocessed_frame'], sp_A)

    print("\n[3f] Generating visualizations...")
    draw_keypoints(img1, sp_1['keypoints'],
                    f"SuperPoint - img1 - {sp_1['num_keypoints']} kpts - {img1.shape[1]}x{img1.shape[0]}",
                    sp_output_dir / "superpoint_keypoints_img1.png", gt_points=gt1_points)
    draw_keypoints(img2, sp_2['keypoints'],
                    f"SuperPoint - img2 - {sp_2['num_keypoints']} kpts - {img2.shape[1]}x{img2.shape[0]}",
                    sp_output_dir / "superpoint_keypoints_img2.png", gt_points=gt2_points)
    draw_matches(img1, img2, sp_1['keypoints'], sp_2['keypoints'],
                 sp_matches, sp_inliers, 'SuperPoint', sp_output_dir / "superpoint_matches.png")
    draw_gt_accuracy(img2, gt1_ids, sp_1['gt_points_preprocessed_frame'],
                      gt2_ids, sp_2['gt_points_preprocessed_frame'], sp_A,
                      'SuperPoint', sp_output_dir / "superpoint_gt_accuracy.png")
    create_flicker_gif(img1, img2, sp_A, sp_output_dir / "superpoint_flicker.gif")

    np.savez(sp_output_dir / "results.npz", H=sp_H, A=sp_A,
             num_matches=len(sp_matches), num_inliers=len(sp_inliers))
    save_points_csv(sp_output_dir / "gt1_preprocessed.csv", gt1_ids, gt1_points)
    save_points_csv(sp_output_dir / "gt2_preprocessed.csv", gt2_ids, gt2_points)
    write_pipeline_summary(
        sp_output_dir, "SuperPoint", image1_path, image2_path, desc1, desc2, sp_1, sp_2,
        len(sp_matches), sp_inliers, sp_A, sp_tre,
        extra_lines=[f"Input to SuperPoint (image 1): {describe_superpoint_downscale(sp_1)}",
                     f"Img1 inference time: {sp_1['time']:.3f} s",
                     f"Input to SuperPoint (image 2): {describe_superpoint_downscale(sp_2)}",
                     f"Img2 inference time: {sp_2['time']:.3f} s"])
    print(f"[SuperPoint] Results saved to: {sp_output_dir}")

    # ========================================================================
    # ASHLAR-style phase correlation
    # ========================================================================
    ashlar = None
    ashlar_tre = (None,) * 5
    if RUN_ASHLAR:
        print("\n" + "=" * 70)
        print("ASHLAR-STYLE PHASE CORRELATION")
        print("=" * 70)
        ashlar_output_dir = create_output_dir("ASHLAR")

        print("\n[4a] Whitening + phase correlation (translation only)...")
        ashlar = ashlar_phase_correlation(img1, img2)

        ashlar_tre = compute_tre(gt1_ids, gt1_points, gt2_ids, gt2_points, ashlar['A'])

        print("\n[4b] Generating visualizations...")
        draw_gt_accuracy(img2, gt1_ids, gt1_points, gt2_ids, gt2_points, ashlar['A'],
                          'ASHLAR', ashlar_output_dir / "ashlar_gt_accuracy.png")
        create_flicker_gif(img1, img2, ashlar['A'], ashlar_output_dir / "ashlar_flicker.gif")

        np.savez(ashlar_output_dir / "results.npz", A=ashlar['A'], error=ashlar['error'])
        save_points_csv(ashlar_output_dir / "gt1_preprocessed.csv", gt1_ids, gt1_points)
        save_points_csv(ashlar_output_dir / "gt2_preprocessed.csv", gt2_ids, gt2_points)
        write_transform_summary(
            ashlar_output_dir, "ASHLAR-style phase correlation", image1_path, image2_path,
            desc1, desc2, ashlar['A'], ashlar_tre,
            extra_lines=["Method: port of ashlar.utils.register on the two whole images "
                         "(whitening -> Hann window -> phase_cross_correlation with "
                         "normalization=None -> quadrant check), translation only",
                         "Adaptations: common downscale to fit the pixel budget, "
                         "zero-padding to a common shape after whitening",
                         f"Working size: {ashlar['working_size'][0]}x{ashlar['working_size'][1]} "
                         f"(scale {ashlar['scale']:.3f}, max {ASHLAR_MAX_PIXELS} px)",
                         f"Whitening sigma: {ASHLAR_FILTER_SIGMA} "
                         f"({'plain Laplacian' if ASHLAR_FILTER_SIGMA == 0 else 'Laplacian-of-Gaussian'}), "
                         f"upsample factor: {ASHLAR_UPSAMPLE_FACTOR}",
                         f"ASHLAR error (-log normalized correlation; 0 = identical, "
                         f"larger = worse): {ashlar['error']:.4f}",
                         f"Registration time: {ashlar['time']:.3f} s"])
        print(f"[ASHLAR] Results saved to: {ashlar_output_dir}")

    # ========================================================================
    # VALIS (kept last: it shuts its JVM down when done)
    # ========================================================================
    valis = None
    valis_tre = (None,) * 5
    if RUN_VALIS:
        print("\n" + "=" * 70)
        print(f"VALIS REGISTRATION (features: {VALIS_FEATURES})")
        print("=" * 70)
        valis_output_dir = create_output_dir("VALIS")

        print("\n[5a] Running VALIS rigid registration...")
        valis = register_valis(img1, img2, valis_output_dir)

        valis_tre = compute_tre(gt1_ids, gt1_points, gt2_ids, gt2_points, valis['A'])

        print("\n[5b] Generating visualizations...")
        draw_gt_accuracy(img2, gt1_ids, gt1_points, gt2_ids, gt2_points, valis['A'],
                          'VALIS', valis_output_dir / "valis_gt_accuracy.png")
        create_flicker_gif(img1, img2, valis['A'], valis_output_dir / "valis_flicker.gif")

        np.savez(valis_output_dir / "results.npz", A=valis['A'])
        save_points_csv(valis_output_dir / "gt1_preprocessed.csv", gt1_ids, gt1_points)
        save_points_csv(valis_output_dir / "gt2_preprocessed.csv", gt2_ids, gt2_points)
        valis_lines = [f"Features: {VALIS_FEATURES}, max processed dim: "
                       f"{VALIS_MAX_PROCESSED_DIM or 'VALIS default'}, "
                       f"check reflections: {VALIS_CHECK_REFLECTIONS}, non-rigid: off"]
        if valis['error']:
            valis_lines.append(f"Registration FAILED: {valis['error']}")
        if valis['time'] is not None:
            valis_lines.append(f"Registration time: {valis['time']:.1f} s")
        if valis['fit_residual'] is not None:
            valis_lines.append(f"Affine fit max residual: {valis['fit_residual']:.3f} px")
        valis_lines.append("VALIS error estimates: valis/valis_error_estimates.csv")
        write_transform_summary(valis_output_dir, f"VALIS ({VALIS_FEATURES})", image1_path, image2_path,
                                 desc1, desc2, valis['A'], valis_tre, extra_lines=valis_lines)
        print(f"[VALIS] Results saved to: {valis_output_dir}")

    # ========================================================================
    # Console summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Metric':<30} {'SIFT':<20} {'SuperPoint':<20}")
    print("-" * 70)
    print(f"{'Keypoints (image 1)':<30} {sift_1['num_keypoints']:<20} {sp_1['num_keypoints']:<20}")
    print(f"{'Keypoints (image 2)':<30} {sift_2['num_keypoints']:<20} {sp_2['num_keypoints']:<20}")
    print(f"{'FLANN matches':<30} {len(sift_matches):<20} {len(sp_matches):<20}")
    print(f"{'RANSAC inliers':<30} {len(sift_inliers):<20} {len(sp_inliers):<20}")
    if sift_tre[1] is not None or sp_tre[1] is not None:
        print(f"{'Mean TRE (px)':<30} "
              f"{sift_tre[1] if sift_tre[1] is not None else 'n/a':<20} "
              f"{sp_tre[1] if sp_tre[1] is not None else 'n/a':<20}")
    if ashlar is not None:
        print(f"\nASHLAR-style phase correlation: {ashlar['time']:.3f} s, "
              f"error: {ashlar['error']:.4f}, "
              f"mean TRE (px): {ashlar_tre[1] if ashlar_tre[1] is not None else 'n/a'}")
    if valis is not None:
        if valis['error']:
            print(f"VALIS ({VALIS_FEATURES}): FAILED - {valis['error']}")
        else:
            print(f"VALIS ({VALIS_FEATURES}): {valis['time']:.1f} s, "
                  f"mean TRE (px): {valis_tre[1] if valis_tre[1] is not None else 'n/a'}")
    print("\n\u2728 Done!")


if __name__ == "__main__":
    main()