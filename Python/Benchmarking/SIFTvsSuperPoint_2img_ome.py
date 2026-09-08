"""
Compare SIFT vs SuperPoint registration using FLANN + RANSAC
DAPI vs DAPI, both as OME-TIFF stacks (select one layer from each).

Preprocessing: negation (toggle per image) is the default; optional
brightness enhancement (toggle per image) can be applied before negation.
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
from datetime import datetime
from PIL import Image, ImageEnhance
import tifffile

# ============================================================================
# AUTO PATH CONFIGURATION
# ============================================================================
def setup_paths():
    """Automatically detect all paths based on script location"""
    # Get current script directory (myproject/Python/)
    script_dir = Path(__file__).resolve().parent
    # Get project root (myproject/)
    project_root = script_dir.parent.parent.parent

    # Define all paths relative to project root
    # These are the DEFAULT folders each image is looked for in.
    images_dir1 = project_root / 'data' / 'mIF' / 'LymphNode' / 'same'
    images_dir2 = project_root / 'data' / 'Xenium' / 'LymphNode'
    repos_dir = project_root.parent.parent / '10. Sem' / 'Praktikum MDC' / 'git'
    results_dir = project_root / 'Results' / 'Python' / 'SIFTvsSuperPoint_DAPI_DAPI_ome'

    # Create results directory if it doesn't exist
    results_dir.mkdir(parents=True, exist_ok=True)

    # Find SuperPoint in Repos directory
    superpoint_path = None
    if repos_dir.exists():
        # Look for any directory containing 'SuperPoint' (case insensitive)
        sp_candidates = [d for d in repos_dir.iterdir()
                        if d.is_dir() and 'superpoint' in d.name.lower()]
        if sp_candidates:
            superpoint_path = sp_candidates[0]
            print(f"✓ Found SuperPoint at: {superpoint_path}")

    if superpoint_path is None:
        raise FileNotFoundError(
            f"\n❌ SuperPoint not found!\n"
            f"Expected location: {repos_dir}/SuperPoint or similar\n"
            f"Please clone it:\n"
            f"  cd {repos_dir}\n"
            f"  git clone https://github.com/rpautrat/SuperPoint.git\n"
        )

    # Add SuperPoint to Python path
    sys.path.insert(0, str(superpoint_path))

    # Find the weights file (try common locations)
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
            print(f"✓ Found weights at: {weights_path}")
            break

    if weights_path is None:
        raise FileNotFoundError(
            f"\n❌ SuperPoint weights not found!\n"
            f"Searched in:\n" +
            "\n".join(f"  - {p}" for p in weight_candidates) +
            f"\n\nPlease ensure weights file is in the SuperPoint/weights/ directory"
        )

    return {
        'project_root': project_root,
        'images_dir1': images_dir1,
        'images_dir2': images_dir2,
        'results_dir': results_dir,
        'superpoint_path': superpoint_path,
        'weights_path': weights_path
    }

# ============================================================================
# Run path setup (make paths available globally)
# ============================================================================
paths = setup_paths()

# Import SuperPoint after path is set
from superpoint_pytorch import SuperPoint

# ============================================================================
# IMAGE SELECTION
# ============================================================================
# Set these to pick specific images from images_dir1 / images_dir2 above.
# Leave as None to auto-pick the first image found in the respective folder.
IMAGE1_NAME = 'Core_13.ome.tif'
IMAGE2_NAME = 'Core_13.tiff'

# Which layer/page to use from each OME-TIFF stack (0 = first layer).
# Ignored for non-stack formats (png/jpg/plain single-page tif).
IMAGE1_LAYER = 0
IMAGE2_LAYER = 0

# Negate an image's intensities (background -> white, nuclei -> black)
# before feature detection. The only preprocessing step in this script.
NEGATE_IMG1 = False
NEGATE_IMG2 = False

# Optional: boost brightness before negation (e.g. when one DAPI scan is
# dimmer/nuclei-less-visible than the other, such as an adjacent-slide
# sample vs. a same-slide sample). Applied to the RAW image, before negation.
ENHANCE_IMG1 = False
ENHANCE_IMG2 = False
ENHANCE_FACTOR = 1.5   # >1 brightens, <1 darkens, 1 = no change

# --- Visualization-only settings (do NOT affect feature detection/matching) ---
VIS_MAX_DIM = 1920      # cap the longer side of saved keypoint/match images (px)

def resolve_input_images():
    """
    Resolve the two input images to compare.
    Each image is looked for in its own directory (paths['images_dir1'] /
    paths['images_dir2']). Uses IMAGE1_NAME / IMAGE2_NAME if set, otherwise
    auto-picks the first image found (sorted) in the respective directory.
    """
    dir1 = paths['images_dir1']
    dir2 = paths['images_dir2']

    def resolve_one(directory, image_name, label):
        if image_name:
            img_path = directory / image_name
            if not img_path.exists():
                raise FileNotFoundError(f"{label} not found: {img_path}")
            return img_path

        image_files = sorted(
            list(directory.glob('*.png')) +
            list(directory.glob('*.jpg')) +
            list(directory.glob('*.tif')) +
            list(directory.glob('*.tiff'))
        )
        if not image_files:
            raise FileNotFoundError(
                f"\n❌ No images found in {directory} for {label}.\n"
                f"Either add an image there, or set the corresponding *_NAME explicitly."
            )
        return image_files[0]

    img1_path = resolve_one(dir1, IMAGE1_NAME, "IMAGE1_NAME")
    img2_path = resolve_one(dir2, IMAGE2_NAME, "IMAGE2_NAME")

    return img1_path, img2_path

# ============================================================================
# IMAGE LOADING (supports plain images and OME-TIFF stacks)
# ============================================================================
def load_grayscale_image(path, layer_index=0):
    """
    Load an image as an 8-bit grayscale numpy array, regardless of whether
    it's a plain PNG/JPG (via cv2), a plain single-plane or RGB/RGBA TIFF,
    or a multi-channel/multi-layer OME-TIFF stack (via tifffile, selecting
    layer_index).

    Handles three TIFF shapes:
      - (H, W)              -> already a single grayscale plane, used as-is
      - (H, W, 3) or (H, W, 4) -> RGB(A) color image, converted to grayscale
      - (N, H, W)            -> channel/layer-first stack, layer_index selects
                                 which plane (true for e.g. a CYX OME-TIFF).

    If your OME-TIFF has a more complex axis order (e.g. separate C/Z/T
    dimensions, so ndim > 3), this raises rather than guessing - check
    tifffile.TiffFile(path).series[0].axes and adjust the indexing below.
    """
    suffix = path.suffix.lower()

    if suffix in ('.tif', '.tiff'):
        arr = tifffile.imread(str(path))
        print(f"   Loaded TIFF array: shape={arr.shape}, dtype={arr.dtype}")

        if arr.ndim == 2:
            image = arr
        elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
            # RGB(A) color image (H, W, C) - not a channel/layer stack
            print(f"   Detected RGB(A) color TIFF, converting to grayscale")
            rgb = arr[..., :3]
            image = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        elif arr.ndim == 3:
            # Channel/layer-first stack (N, H, W), e.g. a CYX OME-TIFF
            if layer_index >= arr.shape[0]:
                raise IndexError(
                    f"layer_index={layer_index} out of range for stack with "
                    f"{arr.shape[0]} layers ({path})"
                )
            image = arr[layer_index]
            print(f"   Selected layer {layer_index}/{arr.shape[0]-1}")
        else:
            raise ValueError(
                f"Unexpected array shape {arr.shape} for {path} (ndim={arr.ndim}). "
                f"Inspect the axis order manually and adjust load_grayscale_image()."
            )

        # Normalize to 8-bit if needed (microscopy TIFFs are often 16-bit)
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return image

    else:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not load image from {path}")
        return image

# ============================================================================
# DAPI PREPROCESSING
# ============================================================================
def negate_image(image):
    """
    Negate an image's intensities: bright <-> dark inverted
    (e.g. for a DAPI scan: background becomes white, nuclei become black).
    """
    return cv2.bitwise_not(image)

def enhance_image(image, factor=3):
    """
    Enhances image brightness (e.g. for a dimmer DAPI scan where nuclei are
    less visible - such as an adjacent-slide sample vs. a same-slide one).

    ImageEnhance.Brightness works on a PIL Image, not a numpy array, so this
    wraps the conversion both ways to stay a drop-in numpy-in/numpy-out step
    alongside negate_image().
    """
    pil_img = Image.fromarray(image)
    enhanced = ImageEnhance.Brightness(pil_img).enhance(factor)
    return np.array(enhanced)

def describe_preprocessing(enhance_flag, negate_flag, enhance_factor):
    """
    Human-readable record of preprocessing applied (in the order it
    actually runs: enhance -> negate), for the summary file.
    """
    steps = []
    if enhance_flag:
        steps.append(f"brightness enhanced (factor={enhance_factor})")
    if negate_flag:
        steps.append("negated (inverted intensities)")
    return " -> ".join(steps) if steps else "none"

# ============================================================================
# IMAGE RESIZING FOR SUPERPOINT (Memory Safe)
# ============================================================================
def resize_for_superpoint(image, max_pixels=4000000, stride=8):
    """
    Resize image to target megapixels while preserving aspect ratio.
    Ensures dimensions are multiples of stride (8).

    Args:
        image: Input image (grayscale or color)
        max_pixels: Maximum number of pixels (e.g., 4_000_000 for 4MP)
        stride: Required stride for SuperPoint (default 8)

    Returns:
        Resized image with dimensions multiple of stride
    """
    h, w = image.shape[:2]
    total_pixels = h * w

    if total_pixels <= max_pixels:
        # Still need to ensure dimensions are multiples of stride
        new_h = (h // stride) * stride
        new_w = (w // stride) * stride

        if new_h != h or new_w != w:
            # Crop from center to make dimensions divisible by stride
            offset_y = (h - new_h) // 2
            offset_x = (w - new_w) // 2
            image = image[offset_y:offset_y + new_h, offset_x:offset_x + new_w]
            print(f"   Cropped from {w}x{h} to {new_w}x{new_h} (center crop for stride {stride})")

        return image

    # Calculate scale factor to reach target pixels
    scale = np.sqrt(max_pixels / total_pixels)
    new_h = int(h * scale)
    new_w = int(w * scale)

    # Round to nearest multiple of stride
    new_h = (new_h // stride) * stride
    new_w = (new_w // stride) * stride

    print(f"   Original: {w}x{h} ({total_pixels/1e6:.1f}MP)")
    print(f"   Resizing to: {new_w}x{new_h} ({new_w*new_h/1e6:.1f}MP)")

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image

# ============================================================================
# SIFT Feature Extraction
# ============================================================================
def extract_sift_features(image, max_keypoints=5000):
    """
    Extract SIFT keypoints and descriptors
    """
    sift = cv2.SIFT_create(nfeatures=max_keypoints)
    keypoints, descriptors = sift.detectAndCompute(image, None)

    if keypoints is None:
        return None

    kp_array = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])
    scores = np.array([kp.response for kp in keypoints])

    print(f"   Found {len(kp_array)} keypoints")

    return {
        'keypoints': kp_array,
        'descriptors': descriptors,
        'scores': scores,
        'num_keypoints': len(kp_array)
    }

# ============================================================================
# SuperPoint Setup
# ============================================================================
def load_superpoint_model():
    """Load SuperPoint model using auto-detected paths"""
    model = SuperPoint(
        nms_radius=4,
        max_num_keypoints=20000,
        detection_threshold=0.005,
        remove_borders=4,
        descriptor_dim=256,
        channels=[64, 64, 128, 128, 256]
    )

    # Use the auto-detected weights path
    state_dict = torch.load(paths['weights_path'], map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    return model

def extract_superpoint_features(model, image, max_pixels=4000000):
    """
    Extract SuperPoint keypoints and descriptors
    Includes automatic downsampling for large images
    """
    # Apply memory-safe resizing (4MP default)
    original_h, original_w = image.shape
    image_resized = resize_for_superpoint(image, max_pixels=max_pixels, stride=8)
    h, w = image_resized.shape

    # Normalize and convert to tensor
    img_normalized = image_resized.astype(np.float32) / 255.0
    input_tensor = torch.from_numpy(img_normalized).float().unsqueeze(0).unsqueeze(0)

    # Run inference
    start_time = time.time()
    with torch.no_grad():
        output = model({"image": input_tensor})
    inference_time = time.time() - start_time

    # Extract results
    keypoints = output['keypoints'][0].cpu().numpy()
    descriptors = output['descriptors'][0].cpu().numpy()
    scores = output['keypoint_scores'][0].cpu().numpy()

    # Scale keypoints back to original image coordinates if resized
    if h != original_h or w != original_w:
        scale_x = original_w / w
        scale_y = original_h / h
        keypoints[:, 0] = keypoints[:, 0] * scale_x
        keypoints[:, 1] = keypoints[:, 1] * scale_y
        print(f"   Scaled keypoints back to original resolution: {original_w}x{original_h}")

    print(f"   SuperPoint inference: {inference_time*1000:.1f} ms")
    print(f"   Found {len(keypoints)} keypoints")

    return {
        'keypoints': keypoints,
        'descriptors': descriptors,
        'scores': scores,
        'num_keypoints': len(keypoints),
        'time': inference_time
    }

# ============================================================================
# FLANN Matcher
# ============================================================================
def get_flann_matcher():
    """Get FLANN matcher"""
    index_params = dict(algorithm=1, trees=5)
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)

def match_features_flann(desc1, desc2, flann_matcher, lowes_ratio=0.8, k=2):
    """
    Match features using FLANN with Lowe's ratio test
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        print(f"   FLANN matching: skipped (empty descriptor set - desc1={0 if desc1 is None else len(desc1)}, desc2={0 if desc2 is None else len(desc2)})")
        return []

    # knnMatch requires each descriptor set to contain at least k entries,
    # otherwise cv2 raises a hard assertion error rather than returning
    # fewer neighbors. Guard against that instead of crashing.
    if len(desc1) < k or len(desc2) < k:
        print(f"   FLANN matching: skipped (need >= {k} keypoints per image for ratio test, "
              f"got {len(desc1)} and {len(desc2)})")
        return []

    desc1 = desc1.astype(np.float32)
    desc2 = desc2.astype(np.float32)

    start_time = time.time()
    matches = flann_matcher.knnMatch(desc1, desc2, k=k)
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

# ============================================================================
# RANSAC Filtering
# ============================================================================
def filter_with_ransac(matches, kp1, kp2, ransac_thresh=5.0):
    """
    Filter matches using RANSAC to find geometrically consistent ones
    """
    if len(matches) < 4:
        print(f"   RANSAC: Need at least 4 matches, have {len(matches)}")
        return [], None, None

    src_pts = np.float32([kp1[m.queryIdx] for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([kp2[m.trainIdx] for m in matches]).reshape(-1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransac_thresh)

    if H is not None:
        inliers = [matches[i] for i in range(len(matches)) if mask[i]]
        print(f"   RANSAC: {len(inliers)}/{len(matches)} inliers ({len(inliers)/len(matches)*100:.1f}%)")
        return inliers, H, mask
    else:
        print(f"   RANSAC: Failed to find homography")
        return [], None, None

# ============================================================================
# Affine Transformation Estimation
# ============================================================================
def estimate_affine_transform(matches, kp1, kp2, ransac_thresh=5.0):
    """
    Estimate a 2D affine transformation (2x3 matrix) mapping points from
    image 1 -> image 2, using only the matches that already passed
    homography RANSAC filtering.

    Returns:
        A_fwd: 2x3 affine matrix mapping img1 -> img2 coordinates (or None)
    """
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
    print(f"   Affine matrix (img1 -> img2):\n{A_fwd}")

    return A_fwd

# ============================================================================
# Visualization
# ============================================================================
def create_output_dir():
    """Create timestamped output directory inside Results/Python/"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = paths['results_dir'] / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}")
    return output_dir

def resize_for_display(image, max_dim=1920):
    """
    Downscale (never upscale) so the longer side is at most max_dim,
    purely to keep saved visualization files a reasonable size.
    Returns (resized_image, scale_factor) - scale_factor lets callers
    map original-image keypoint coordinates onto the resized canvas.
    """
    h, w = image.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image, scale

def _scaled_marker_params(w, h):
    """
    Scale circle radius/thickness and font size relative to image
    dimensions, so keypoints/text stay legible on both small and very
    large (multi-thousand-pixel) pathology images.
    """
    ref = min(w, h)
    radius = max(4, int(round(ref * 0.004)))
    circle_thickness = max(2, radius // 2)
    line_thickness = max(2, radius // 2)
    font_scale = max(1.0, ref / 1200)
    font_thickness = max(2, int(round(font_scale * 2)))
    return radius, circle_thickness, line_thickness, font_scale, font_thickness

def _draw_label_bar(img, text, font_scale, font_thickness):
    """Draw a filled black bar with white text across the top of img (in-place-ish, returns img)."""
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    bar_h = text_h + baseline + int(20 * font_scale)
    cv2.rectangle(img, (0, 0), (img.shape[1], bar_h), (0, 0, 0), -1)
    cv2.putText(img, text, (10, bar_h - baseline - 5), cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    return img

def draw_keypoints(image, keypoints, title, save_path, color=(0, 255, 0)):
    """
    Draw detected keypoints on a single image and save at a display-capped
    resolution (VIS_MAX_DIM).
    """
    display_img, scale = resize_for_display(image, max_dim=VIS_MAX_DIM)

    img_color = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
    h, w = img_color.shape[:2]
    radius, circle_thickness, _, font_scale, font_thickness = _scaled_marker_params(w, h)

    for pt in keypoints:
        x, y = int(round(pt[0] * scale)), int(round(pt[1] * scale))
        cv2.circle(img_color, (x, y), radius, color, circle_thickness)

    _draw_label_bar(img_color, f'{title} - {len(keypoints)} keypoints', font_scale, font_thickness)

    cv2.imwrite(str(save_path), img_color)

def draw_matches(img1, img2, kp1, kp2, matches, inliers, title, save_path):
    """
    Draw matches with inliers in green, outliers in red, saved at a
    display-capped resolution (VIS_MAX_DIM).
    """
    disp1, scale1 = resize_for_display(img1, max_dim=VIS_MAX_DIM)
    disp2, scale2 = resize_for_display(img2, max_dim=VIS_MAX_DIM)

    h1, w1 = disp1.shape[:2]
    h2, w2 = disp2.shape[:2]
    match_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    match_img[:h1, :w1] = cv2.cvtColor(disp1, cv2.COLOR_GRAY2BGR)
    match_img[:h2, w1:w1+w2] = cv2.cvtColor(disp2, cv2.COLOR_GRAY2BGR)

    radius, circle_thickness, line_thickness, font_scale, font_thickness = _scaled_marker_params(
        w1 + w2, max(h1, h2)
    )

    # Create set of inlier query indices
    inlier_set = set()
    for m in inliers:
        inlier_set.add(m.queryIdx)

    for m in matches:
        pt1 = (int(round(kp1[m.queryIdx][0] * scale1)), int(round(kp1[m.queryIdx][1] * scale1)))
        pt2 = (int(round(kp2[m.trainIdx][0] * scale2 + w1)), int(round(kp2[m.trainIdx][1] * scale2)))

        if m.queryIdx in inlier_set:
            color = (0, 255, 0)  # Green
            thickness = line_thickness
        else:
            color = (0, 0, 255)  # Red
            thickness = max(1, line_thickness // 2)

        cv2.line(match_img, pt1, pt2, color, thickness)
        cv2.circle(match_img, pt1, radius, color, -1)
        cv2.circle(match_img, pt2, radius, color, -1)

    _draw_label_bar(match_img, f'{title} - {len(inliers)} inliers / {len(matches)} total',
                     font_scale, font_thickness)

    cv2.imwrite(str(save_path), match_img)

def create_flicker_gif(img1, img2, A_fwd, save_path, alpha=0.5,
                        n_cycles=4, frame_duration_ms=1000):
    """
    Create a "flashing"/flicker comparison GIF for a registration result.

    img1 is used as the static background (full opacity). img2 is warped
    into img1's coordinate frame using the estimated affine transform,
    then alternated on/off (at `alpha` opacity when "on") to produce a
    flicker effect useful for visually judging registration quality.

    The warp/blend itself runs at full resolution (for accuracy), and only
    the final frames are downscaled to VIS_MAX_DIM before saving - keeps
    the GIF file size reasonable regardless of source image resolution.

    Args:
        img1, img2: grayscale images (as loaded, img2 NOT pre-warped)
        A_fwd: 2x3 affine matrix mapping img1 -> img2 (as returned by
               estimate_affine_transform)
        alpha: opacity of img2 when it is "on" (0-1)
        n_cycles: number of on/off flashes
        frame_duration_ms: duration of each frame in milliseconds
    """
    if A_fwd is None:
        print("   Skipping flicker GIF: no affine transform available")
        return

    h1, w1 = img1.shape[:2]

    # We need img2 -> img1 to warp img2 INTO img1's frame for overlay,
    # so invert the img1 -> img2 transform.
    A_inv = cv2.invertAffineTransform(A_fwd)
    img2_warped = cv2.warpAffine(img2, A_inv, (w1, h1))

    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    img2_rgb = cv2.cvtColor(img2_warped, cv2.COLOR_GRAY2RGB)
    blended = cv2.addWeighted(img1_rgb, 1 - alpha, img2_rgb, alpha, 0)

    # Downscale the final frames for display/file-size (does NOT affect the
    # warp/blend accuracy above, which already happened at full resolution)
    img1_rgb_disp, _ = resize_for_display(img1_rgb, max_dim=VIS_MAX_DIM)
    blended_disp, _ = resize_for_display(blended, max_dim=VIS_MAX_DIM)

    frame_off = Image.fromarray(img1_rgb_disp)
    frame_on = Image.fromarray(blended_disp)

    frames = []
    for _ in range(n_cycles):
        frames.append(frame_off)
        frames.append(frame_on)

    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0
    )
    print(f"   Saved flicker GIF: {save_path}")

# ============================================================================
# Main Function
# ============================================================================
def main():
    # Resolve the two input images (no hardcoded paths!)
    image1_path, image2_path = resolve_input_images()
    print(f"\n📷 Image 1: {image1_path.name} (layer {IMAGE1_LAYER}, folder: {image1_path.parent.name})")
    print(f"📷 Image 2: {image2_path.name} (layer {IMAGE2_LAYER}, folder: {image2_path.parent.name})")

    print("=" * 70)
    print("SIFT vs SuperPoint Registration Comparison (DAPI vs DAPI, OME-TIFF)")
    print("=" * 70)

    # Create output directory
    output_dir = create_output_dir()

    # Load both images (handles plain images and OME-TIFF stacks alike)
    print("\n[1] Loading image 1...")
    img1 = load_grayscale_image(image1_path, layer_index=IMAGE1_LAYER)
    print("[1] Loading image 2...")
    img2 = load_grayscale_image(image2_path, layer_index=IMAGE2_LAYER)

    print(f"\n[1] Loaded image 1: {img1.shape}")
    print(f"[1] Loaded image 2: {img2.shape}")

    # Optional brightness enhancement, applied before negation
    if ENHANCE_IMG1:
        img1 = enhance_image(img1, factor=ENHANCE_FACTOR)
        print(f"[1] Enhanced brightness of image 1 (factor={ENHANCE_FACTOR})")
    if ENHANCE_IMG2:
        img2 = enhance_image(img2, factor=ENHANCE_FACTOR)
        print(f"[1] Enhanced brightness of image 2 (factor={ENHANCE_FACTOR})")

    # Negate intensities where configured
    if NEGATE_IMG1:
        img1 = negate_image(img1)
        print("[1] Negated image 1 (inverted intensities)")
    if NEGATE_IMG2:
        img2 = negate_image(img2)
        print("[1] Negated image 2 (inverted intensities)")

    img1_preprocessing = describe_preprocessing(ENHANCE_IMG1, NEGATE_IMG1, ENHANCE_FACTOR)
    img2_preprocessing = describe_preprocessing(ENHANCE_IMG2, NEGATE_IMG2, ENHANCE_FACTOR)

    # ========================================================================
    # SIFT Processing
    # ========================================================================
    print("\n" + "=" * 70)
    print("SIFT REGISTRATION")
    print("=" * 70)

    print("\n[2a] Extracting SIFT features...")
    sift_1 = extract_sift_features(img1, max_keypoints=5000)
    sift_2 = extract_sift_features(img2, max_keypoints=5000)

    print("\n[2b] FLANN matching...")
    flann = get_flann_matcher()
    sift_matches = match_features_flann(sift_1['descriptors'], sift_2['descriptors'], flann)

    print("\n[2c] RANSAC filtering...")
    sift_inliers, sift_H, sift_mask = filter_with_ransac(
        sift_matches, sift_1['keypoints'], sift_2['keypoints'], ransac_thresh=5.0
    )

    print("\n[2d] Estimating affine transform from inlier matches...")
    sift_A = estimate_affine_transform(
        sift_inliers, sift_1['keypoints'], sift_2['keypoints'], ransac_thresh=5.0
    )

    # ========================================================================
    # SuperPoint Processing
    # ========================================================================
    print("\n" + "=" * 70)
    print("SUPERPOINT REGISTRATION")
    print("=" * 70)

    print("\n[3a] Loading SuperPoint model...")
    sp_model = load_superpoint_model()

    print("\n[3b] Extracting SuperPoint features...")
    # Set max_pixels to 4_000_000 (4MP) or 8_000_000 (8MP)
    sp_1 = extract_superpoint_features(sp_model, img1, max_pixels=8000000)
    sp_2 = extract_superpoint_features(sp_model, img2, max_pixels=8000000)

    print("\n[3c] FLANN matching...")
    sp_matches = match_features_flann(sp_1['descriptors'], sp_2['descriptors'], flann)

    print("\n[3d] RANSAC filtering...")
    sp_inliers, sp_H, sp_mask = filter_with_ransac(
        sp_matches, sp_1['keypoints'], sp_2['keypoints'], ransac_thresh=5.0
    )

    print("\n[3e] Estimating affine transform from inlier matches...")
    sp_A = estimate_affine_transform(
        sp_inliers, sp_1['keypoints'], sp_2['keypoints'], ransac_thresh=5.0
    )

    # ========================================================================
    # Results Summary
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
    print(f"{'Inlier % (of matches)':<30} {len(sift_inliers)/max(len(sift_matches),1)*100:<20.1f} {len(sp_inliers)/max(len(sp_matches),1)*100:<20.1f}")

    # ========================================================================
    # Visualizations
    # ========================================================================
    print("\n[4] Generating visualizations...")

    # Keypoint visualizations (one per image, per method)
    draw_keypoints(img1, sift_1['keypoints'], 'SIFT - Image 1', output_dir / "sift_keypoints_img1.png")
    draw_keypoints(img2, sift_2['keypoints'], 'SIFT - Image 2', output_dir / "sift_keypoints_img2.png")
    draw_keypoints(img1, sp_1['keypoints'], 'SuperPoint - Image 1', output_dir / "superpoint_keypoints_img1.png")
    draw_keypoints(img2, sp_2['keypoints'], 'SuperPoint - Image 2', output_dir / "superpoint_keypoints_img2.png")

    # Match visualizations (green = RANSAC inlier, red = outlier)
    draw_matches(
        img1, img2,
        sift_1['keypoints'], sift_2['keypoints'],
        sift_matches, sift_inliers,
        'SIFT',
        output_dir / "sift_matches.png"
    )

    draw_matches(
        img1, img2,
        sp_1['keypoints'], sp_2['keypoints'],
        sp_matches, sp_inliers,
        'SuperPoint',
        output_dir / "superpoint_matches.png"
    )

    # Flicker/flashing GIFs (img1 static, warped img2 flashes on top)
    create_flicker_gif(
        img1, img2, sift_A,
        output_dir / "sift_flicker.gif",
        alpha=0.5, n_cycles=4, frame_duration_ms=1000
    )

    create_flicker_gif(
        img1, img2, sp_A,
        output_dir / "superpoint_flicker.gif",
        alpha=0.5, n_cycles=4, frame_duration_ms=1000
    )

    # ========================================================================
    # Save Results
    # ========================================================================
    print("\n[5] Saving results...")

    np.savez(
        output_dir / "results.npz",
        sift_H=sift_H,
        sift_A=sift_A,
        sp_H=sp_H,
        sp_A=sp_A,
        sift_num_matches=len(sift_matches),
        sift_num_inliers=len(sift_inliers),
        sp_num_matches=len(sp_matches),
        sp_num_inliers=len(sp_inliers),
    )

    # Save summary
    with open(output_dir / "comparison_summary.txt", 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("SIFT vs SuperPoint Registration Comparison (DAPI vs DAPI, OME-TIFF)\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Image 1: {image1_path.name} (layer {IMAGE1_LAYER})\n")
        f.write(f"  Source folder: {image1_path.parent.name}\n")
        f.write(f"  Preprocessing applied: {img1_preprocessing}\n")
        f.write(f"Image 2: {image2_path.name} (layer {IMAGE2_LAYER})\n")
        f.write(f"  Source folder: {image2_path.parent.name}\n")
        f.write(f"  Preprocessing applied: {img2_preprocessing}\n\n")

        f.write("SIFT RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Keypoints (image 1): {sift_1['num_keypoints']}\n")
        f.write(f"Keypoints (image 2): {sift_2['num_keypoints']}\n")
        f.write(f"FLANN matches: {len(sift_matches)}\n")
        f.write(f"RANSAC inliers: {len(sift_inliers)} ({len(sift_inliers)/max(len(sift_matches),1)*100:.1f}%)\n")
        f.write(f"Affine matrix (img1 -> img2):\n{sift_A}\n\n")

        f.write("SUPERPOINT RESULTS\n")
        f.write("-" * 40 + "\n")
        f.write(f"Keypoints (image 1): {sp_1['num_keypoints']}\n")
        f.write(f"Keypoints (image 2): {sp_2['num_keypoints']}\n")
        f.write(f"FLANN matches: {len(sp_matches)}\n")
        f.write(f"RANSAC inliers: {len(sp_inliers)} ({len(sp_inliers)/max(len(sp_matches),1)*100:.1f}%)\n")
        f.write(f"Affine matrix (img1 -> img2):\n{sp_A}\n")

    print(f"\n✅ All results saved to: {output_dir}")
    print(f"   - Keypoints: sift_keypoints_img1.png, sift_keypoints_img2.png")
    print(f"   - Keypoints: superpoint_keypoints_img1.png, superpoint_keypoints_img2.png")
    print(f"   - Matches:   sift_matches.png, superpoint_matches.png")
    print(f"   - Flicker GIFs: sift_flicker.gif, superpoint_flicker.gif")
    print(f"   - Data: results.npz")
    print(f"   - Summary: comparison_summary.txt")
    print("\n✨ Done!")

if __name__ == "__main__":
    main()