"""
Compare SIFT vs SuperPoint registration using FLANN + RANSAC
Two input images 
"""

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import time
from datetime import datetime
from PIL import Image

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
    images_dir = project_root / 'data' / 'Visium' / 'DLPFC_Visium' / 'DLPFC_Visium_Cropped'   
    repos_dir = project_root.parent.parent / '10. Sem' / 'Praktikum MDC' / 'git'
    results_dir = project_root / 'Results' / 'Python' / 'SIFTvsSuperPoint_adjacent'  

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
        'images_dir': images_dir,
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
# Set these to pick specific images from paths['images_dir'].
# Leave as None to auto-pick the first two images found in that folder.
IMAGE1_NAME = '151675.png'   # e.g. "sample_A.tif"
IMAGE2_NAME = '151676.png'   # e.g. "sample_B.tif"

def resolve_input_images():
    """
    Resolve the two input images to compare.
    Uses IMAGE1_NAME / IMAGE2_NAME if set, otherwise the first two images
    found (sorted) in paths['images_dir'].
    """
    images_dir = paths['images_dir']

    if IMAGE1_NAME and IMAGE2_NAME:
        img1_path = images_dir / IMAGE1_NAME
        img2_path = images_dir / IMAGE2_NAME
        if not img1_path.exists():
            raise FileNotFoundError(f"IMAGE1_NAME not found: {img1_path}")
        if not img2_path.exists():
            raise FileNotFoundError(f"IMAGE2_NAME not found: {img2_path}")
        return img1_path, img2_path

    image_files = sorted(
        list(images_dir.glob('*.png')) +
        list(images_dir.glob('*.jpg')) +
        list(images_dir.glob('*.tif'))
    )

    if len(image_files) < 2:
        raise FileNotFoundError(
            f"\n❌ Need at least 2 images in {images_dir}, found {len(image_files)}.\n"
            f"Either add another image, or set IMAGE1_NAME / IMAGE2_NAME explicitly."
        )

    return image_files[0], image_files[1]

# ============================================================================
# IMAGE RESIZING FOR SUPERPOINT (Memory Safe)
# ============================================================================
def resize_for_superpoint(image, max_pixels=400000000000, stride=8):
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

def match_features_flann(desc1, desc2, flann_matcher, lowes_ratio=0.8):
    """
    Match features using FLANN with Lowe's ratio test
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
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

def draw_keypoints(image, keypoints, title, save_path, color=(0, 255, 0)):
    """
    Draw detected keypoints on a single image.
    """
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    for pt in keypoints:
        cv2.circle(img_color, (int(pt[0]), int(pt[1])), 3, color, 1)

    plt.figure(figsize=(10, 9))
    plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
    plt.title(f'{title} - {len(keypoints)} keypoints')
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def draw_matches(img1, img2, kp1, kp2, matches, inliers, title, save_path):
    """
    Draw matches with inliers in green, outliers in red
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    match_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    match_img[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    match_img[:h2, w1:w1+w2] = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)

    # Create set of inlier query indices
    inlier_set = set()
    for m in inliers:
        inlier_set.add(m.queryIdx)

    for m in matches:
        pt1 = (int(kp1[m.queryIdx][0]), int(kp1[m.queryIdx][1]))
        pt2 = (int(kp2[m.trainIdx][0] + w1), int(kp2[m.trainIdx][1]))

        if m.queryIdx in inlier_set:
            color = (0, 255, 0)  # Green
            thickness = 2
        else:
            color = (0, 0, 255)  # Red
            thickness = 1

        cv2.line(match_img, pt1, pt2, color, thickness)
        cv2.circle(match_img, pt1, 3, color, -1)
        cv2.circle(match_img, pt2, 3, color, -1)

    plt.figure(figsize=(16, 9))
    plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
    plt.title(f'{title} - {len(inliers)} inliers / {len(matches)} total')
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # Close figure to avoid memory issues

def create_flicker_gif(img1, img2, A_fwd, save_path, alpha=0.5,
                        n_cycles=4, frame_duration_ms=1000):
    """
    Create a "flashing"/flicker comparison GIF for a registration result.
 
    img1 is used as the static background (full opacity). img2 is warped
    into img1's coordinate frame using the estimated affine transform,
    then alternated on/off (at `alpha` opacity when "on") to produce a
    flicker effect useful for visually judging registration quality.
 
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
 
    frame_off = Image.fromarray(img1_rgb)
    blended = cv2.addWeighted(img1_rgb, 1 - alpha, img2_rgb, alpha, 0)
    frame_on = Image.fromarray(blended)
 
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
    print(f"\n📷 Image 1: {image1_path.name}")
    print(f"📷 Image 2: {image2_path.name}")
 
    print("=" * 70)
    print("SIFT vs SuperPoint Registration Comparison (two images)")
    print("=" * 70)
 
    # Create output directory
    output_dir = create_output_dir()
 
    # Load both images
    img1 = cv2.imread(str(image1_path), cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(str(image2_path), cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        raise ValueError(f"Could not load image from {image1_path}")
    if img2 is None:
        raise ValueError(f"Could not load image from {image2_path}")
 
    print(f"\n[1] Loaded image 1: {img1.shape}")
    print(f"[1] Loaded image 2: {img2.shape}")
 
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
    sp_1 = extract_superpoint_features(sp_model, img1, max_pixels=4000000)
    sp_2 = extract_superpoint_features(sp_model, img2, max_pixels=4000000)
 
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
        f.write("SIFT vs SuperPoint Registration Comparison (two images)\n")
        f.write("=" * 70 + "\n\n")
 
        f.write(f"Image 1: {image1_path.name}\n")
        f.write(f"Image 2: {image2_path.name}\n\n")
 
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
 