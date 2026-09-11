// SIFTvsSuperPoint.cpp
//
// Compares SIFT and SuperPoint keypoint detection + FLANN matching + RANSAC
// homography filtering on two input images, mirroring the Python benchmarking
// scripts but running natively via LibTorch + OpenCV.
//
// Both detectors run through the SAME tiling engine (large images are split
// into overlapping tiles so SuperPoint never has to hold whole-image feature
// maps in memory, and SIFT gets the same treatment for consistency). Overlap
// duplicates are removed by keeping whichever detection sat further from its
// tile's edge.
//
// Usage:
//   SIFTvsSuperPoint.exe --img1 <path> --img2 <path>
//                         [--rotate1 <deg>] [--rotate2 <deg>]
//                         [--flip1 none|h|v|hv] [--flip2 none|h|v|hv]
//                         [--brightness1 <factor>] [--brightness2 <factor>]
//                         [--target-mp <value>]

#include <opencv2/opencv.hpp>
#include <opencv2/flann.hpp>
#include <torch/script.h>
#include "gif.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ============================================================================
// Structural constants (recompile-on-change; not meant to be tweaked per run)
// ============================================================================

struct SIFTParameters
{
    const int sift_nfeatures = 20000;
    const int max_nfeatures  = 10000;
    const int tile_size      = 5000;
    const int tile_overlap   = 50;
};

struct SuperPointParameters
{
    const int   max_num_keypoints   = 10000;
    const int   nms_radius          = 4;
    const float detection_threshold = 0.005f;
    const int   remove_borders      = 4;
    // tile_size 5000x5000 = 25MP, comfortably under the ~28-30MP practical
    // ceiling established during Python benchmarking on 16GB RAM. Tiling
    // means this is a per-tile budget, not a whole-image cap anymore.
    const int   tile_size    = 5000;
    const int   tile_overlap = 50;
};

struct RansacParameters
{
    const double pixel_threshold = 5.0;
    const double confidence      = 0.95;
    const int    max_iters       = 2000;
};

// Distance (px) within which two keypoints from neighboring tile-overlap
// regions are considered "the same" detection and deduplicated.
constexpr float KEYPOINT_DEDUP_RADIUS_PX = 3.0f;

// Lowe's ratio test threshold, shared by both pipelines.
constexpr float LOWE_RATIO = 0.8f;

// Path to the scripted SuperPoint model, relative to the executable.
constexpr const char* SUPERPOINT_MODEL_FILENAME = "superpoint_scripted.pt";

// ============================================================================
// CLI argument parsing
// ============================================================================

struct CLIArgs
{
    std::string img1_path;
    std::string img2_path;

    double rotate1_deg = 0.0;
    double rotate2_deg = 0.0;

    std::string flip1 = "none"; // none | h | v | hv
    std::string flip2 = "none";

    std::optional<double> brightness1;
    std::optional<double> brightness2;

    std::optional<double> target_mp; // rescale both images to this many megapixels
};

static const char* USAGE =
    "Usage: SIFTvsSuperPoint.exe --img1 <path> --img2 <path>\n"
    "                             [--rotate1 <deg>] [--rotate2 <deg>]\n"
    "                             [--flip1 none|h|v|hv] [--flip2 none|h|v|hv]\n"
    "                             [--brightness1 <factor>] [--brightness2 <factor>]\n"
    "                             [--target-mp <value>]\n";

CLIArgs parseArgs(int argc, char** argv)
{
    CLIArgs args;

    auto next = [&](int& i) -> std::string {
        if (i + 1 >= argc) {
            throw std::runtime_error(std::string("Missing value for argument: ") + argv[i]);
        }
        return std::string(argv[++i]);
    };

    for (int i = 1; i < argc; ++i) {
        std::string flag = argv[i];
        if      (flag == "--img1")        args.img1_path = next(i);
        else if (flag == "--img2")        args.img2_path = next(i);
        else if (flag == "--rotate1")     args.rotate1_deg = std::stod(next(i));
        else if (flag == "--rotate2")     args.rotate2_deg = std::stod(next(i));
        else if (flag == "--flip1")       args.flip1 = next(i);
        else if (flag == "--flip2")       args.flip2 = next(i);
        else if (flag == "--brightness1") args.brightness1 = std::stod(next(i));
        else if (flag == "--brightness2") args.brightness2 = std::stod(next(i));
        else if (flag == "--target-mp")   args.target_mp = std::stod(next(i));
        else throw std::runtime_error("Unknown argument: " + flag);
    }

    if (args.img1_path.empty() || args.img2_path.empty()) {
        throw std::runtime_error("Both --img1 and --img2 are required.");
    }
    return args;
}

// ============================================================================
// Preprocessing: rotation, flip, brightness, rescale
// ============================================================================

cv::Mat applyFlip(const cv::Mat& img, const std::string& mode)
{
    if (mode == "none") return img;
    cv::Mat out;
    if      (mode == "h")  cv::flip(img, out, 1);
    else if (mode == "v")  cv::flip(img, out, 0);
    else if (mode == "hv") cv::flip(img, out, -1);
    else throw std::runtime_error("Invalid flip mode '" + mode + "' (expected none|h|v|hv)");
    return out;
}

cv::Mat applyRotation(const cv::Mat& img, double degrees)
{
    if (degrees == 0.0) return img;

    cv::Point2f center(img.cols / 2.0f, img.rows / 2.0f);
    cv::Mat rot_mat = cv::getRotationMatrix2D(center, degrees, 1.0);

    // Expand the canvas so rotated corners aren't cropped.
    double abs_cos = std::abs(rot_mat.at<double>(0, 0));
    double abs_sin = std::abs(rot_mat.at<double>(0, 1));
    int new_w = int(img.rows * abs_sin + img.cols * abs_cos);
    int new_h = int(img.rows * abs_cos + img.cols * abs_sin);
    rot_mat.at<double>(0, 2) += (new_w / 2.0) - center.x;
    rot_mat.at<double>(1, 2) += (new_h / 2.0) - center.y;

    cv::Mat out;
    cv::warpAffine(img, out, rot_mat, cv::Size(new_w, new_h));
    return out;
}

cv::Mat applyBrightness(const cv::Mat& img, double factor)
{
    cv::Mat out;
    img.convertTo(out, -1, factor, 0.0);
    return out;
}

cv::Mat preprocessImage(const cv::Mat& img, double rotate_deg, const std::string& flip_mode,
                         std::optional<double> brightness_factor)
{
    cv::Mat out = img;
    out = applyRotation(out, rotate_deg);
    out = applyFlip(out, flip_mode);
    if (brightness_factor) out = applyBrightness(out, *brightness_factor);
    return out;
}

cv::Mat rescaleToTargetMP(const cv::Mat& img, double target_mp)
{
    double current_mp = (double)img.rows * img.cols / 1'000'000.0;
    double scale = std::sqrt(target_mp / current_mp);
    cv::Mat out;
    cv::resize(img, out, cv::Size(), scale, scale, cv::INTER_AREA);
    return out;
}

// ============================================================================
// Shared tiling engine
// ============================================================================

struct TiledKeypoint
{
    cv::KeyPoint kp;
    float edge_dist; // distance (px) to the nearest border of the tile it came from
};

// Runs detect_fn over overlapping tiles (or the whole image if it's already
// small enough), shifts keypoint coordinates into global image space, and
// records each keypoint's distance to its own tile's border for later dedup.
void computeTiledKeypoints(
    const cv::Mat& image,
    int tile_size, int tile_overlap,
    const std::function<void(const cv::Mat& tile, std::vector<cv::KeyPoint>&, cv::Mat&)>& detect_fn,
    std::vector<TiledKeypoint>& tiled_keypoints,
    cv::Mat& descriptors)
{
    int width = image.cols;
    int height = image.rows;

    if (width <= tile_size && height <= tile_size) {
        std::vector<cv::KeyPoint> kps;
        cv::Mat desc;
        detect_fn(image, kps, desc);
        for (auto& kp : kps) {
            float edge_dist = (float)std::min({ (double)kp.pt.x, (double)kp.pt.y,
                                                 (double)(width - kp.pt.x), (double)(height - kp.pt.y) });
            tiled_keypoints.push_back({ kp, edge_dist });
        }
        descriptors = desc;
        return;
    }

    int step = tile_size - tile_overlap;
    for (int y = 0; y < height - tile_overlap; y += step) {
        for (int x = 0; x < width - tile_overlap; x += step) {
            int tile_x_end = std::min(x + tile_size, width);
            int tile_y_end = std::min(y + tile_size, height);
            cv::Rect tile_rect(x, y, tile_x_end - x, tile_y_end - y);
            cv::Mat tile = image(tile_rect);

            std::vector<cv::KeyPoint> tile_kps;
            cv::Mat tile_desc;
            detect_fn(tile, tile_kps, tile_desc);

            for (auto& kp : tile_kps) {
                float edge_dist = (float)std::min({ (double)kp.pt.x, (double)kp.pt.y,
                                                     (double)(tile_rect.width - kp.pt.x),
                                                     (double)(tile_rect.height - kp.pt.y) });
                cv::KeyPoint global_kp = kp;
                global_kp.pt.x += x;
                global_kp.pt.y += y;
                tiled_keypoints.push_back({ global_kp, edge_dist });
            }

            if (!tile_desc.empty()) {
                if (descriptors.empty()) descriptors = tile_desc.clone();
                else cv::vconcat(descriptors, tile_desc, descriptors);
            }
        }
    }
}

// Removes near-duplicate keypoints from tile-overlap regions. Among any
// cluster of keypoints within KEYPOINT_DEDUP_RADIUS_PX of each other, keeps
// the one furthest from its tile's edge (best local context).
void deduplicateKeypoints(std::vector<TiledKeypoint>& tiled_keypoints, cv::Mat& descriptors, float radius_px)
{
    int n = (int)tiled_keypoints.size();
    if (n == 0) return;

    cv::Mat points(n, 2, CV_32F);
    for (int i = 0; i < n; ++i) {
        points.at<float>(i, 0) = tiled_keypoints[i].kp.pt.x;
        points.at<float>(i, 1) = tiled_keypoints[i].kp.pt.y;
    }
    cv::flann::Index index(points, cv::flann::KDTreeIndexParams(4));

    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return tiled_keypoints[a].edge_dist > tiled_keypoints[b].edge_dist;
    });

    std::vector<bool> discarded(n, false);
    float radius_sq = radius_px * radius_px;

    for (int idx : order) {
        if (discarded[idx]) continue;
        cv::Mat query = points.row(idx);
        std::vector<int> neighbor_idx(16, -1);
        std::vector<float> neighbor_dist(16, -1.0f);
        // NOTE: radius here is squared L2 distance per OpenCV's flann::Index
        // convention with the default L2 distance type. Verify against your
        // OpenCV version if dedup behaves unexpectedly.
        index.radiusSearch(query, neighbor_idx, neighbor_dist, radius_sq, 16, cv::flann::SearchParams(32));

        for (int j : neighbor_idx) {
            if (j < 0 || j == idx) continue;
            discarded[j] = true;
        }
    }

    std::vector<TiledKeypoint> kept_kps;
    cv::Mat kept_desc;
    for (int i = 0; i < n; ++i) {
        if (discarded[i]) continue;
        kept_kps.push_back(tiled_keypoints[i]);
        if (kept_desc.empty()) kept_desc = descriptors.row(i).clone();
        else cv::vconcat(kept_desc, descriptors.row(i), kept_desc);
    }
    tiled_keypoints = kept_kps;
    descriptors = kept_desc;
}

// Caps the keypoint count to max_count, keeping the strongest responses.
// Shared by both SIFT (max_nfeatures) and SuperPoint (max_num_keypoints).
void capKeypointsByScore(std::vector<TiledKeypoint>& tiled_kps, cv::Mat& descriptors, int max_count)
{
    if ((int)tiled_kps.size() <= max_count) return;

    std::vector<int> order(tiled_kps.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = (int)i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return tiled_kps[a].kp.response > tiled_kps[b].kp.response;
    });
    order.resize(max_count);

    std::vector<TiledKeypoint> capped_kps;
    cv::Mat capped_desc;
    for (int idx : order) {
        capped_kps.push_back(tiled_kps[idx]);
        if (capped_desc.empty()) capped_desc = descriptors.row(idx).clone();
        else cv::vconcat(capped_desc, descriptors.row(idx), capped_desc);
    }
    tiled_kps = capped_kps;
    descriptors = capped_desc;
}

// ============================================================================
// SIFT pipeline
// ============================================================================

void computeSIFT(const cv::Mat& image, const SIFTParameters& params,
                  std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    cv::Ptr<cv::SIFT> sift = cv::SIFT::create(params.sift_nfeatures);

    auto detect_fn = [&](const cv::Mat& tile, std::vector<cv::KeyPoint>& kps, cv::Mat& desc) {
        sift->detectAndCompute(tile, cv::Mat(), kps, desc);
    };

    std::vector<TiledKeypoint> tiled_kps;
    computeTiledKeypoints(image, params.tile_size, params.tile_overlap, detect_fn, tiled_kps, descriptors);
    deduplicateKeypoints(tiled_kps, descriptors, KEYPOINT_DEDUP_RADIUS_PX);
    capKeypointsByScore(tiled_kps, descriptors, params.max_nfeatures);

    keypoints.clear();
    keypoints.reserve(tiled_kps.size());
    for (auto& tk : tiled_kps) keypoints.push_back(tk.kp);
}

// ============================================================================
// SuperPoint pipeline
// ============================================================================

torch::Tensor grayMatToTensor(const cv::Mat& gray_float01)
{
    // gray_float01: single channel, CV_32F, range [0,1]
    torch::Tensor t = torch::from_blob(
        (void*)gray_float01.data, { 1, 1, gray_float01.rows, gray_float01.cols }, torch::kFloat32
    ).clone(); // clone: from_blob doesn't own the Mat's memory
    return t;
}

void computeSuperPointTile(const cv::Mat& tile_gray, torch::jit::script::Module& model,
                            std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    cv::Mat gray_float;
    // Handle both 8-bit and 16-bit source images (many microscopy TIFFs are 16-bit).
    double norm_scale = (tile_gray.depth() == CV_16U) ? (1.0 / 65535.0) : (1.0 / 255.0);
    tile_gray.convertTo(gray_float, CV_32F, norm_scale);

    torch::Tensor input = grayMatToTensor(gray_float);

    c10::Dict<std::string, torch::Tensor> input_dict;
    input_dict.insert("image", input);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_dict);

    torch::NoGradGuard no_grad;
    auto output = model.forward(inputs).toGenericDict();

    torch::Tensor kp_tensor    = output.at("keypoints").toTensorVector()[0];       // [N,2] (x,y)
    torch::Tensor score_tensor = output.at("keypoint_scores").toTensorVector()[0]; // [N]
    torch::Tensor desc_tensor  = output.at("descriptors").toTensorVector()[0];     // [N,256]

    int n = (int)kp_tensor.size(0);
    keypoints.clear();
    keypoints.reserve(n);

    auto kp_acc = kp_tensor.accessor<float, 2>();
    auto score_acc = score_tensor.accessor<float, 1>();
    for (int i = 0; i < n; ++i) {
        // KeyPoint(x, y, size, angle, response)
        keypoints.emplace_back(kp_acc[i][0], kp_acc[i][1], 1.0f, -1.0f, score_acc[i]);
    }

    int desc_dim = (int)desc_tensor.size(1);
    descriptors = cv::Mat(n, desc_dim, CV_32F);
    auto desc_acc = desc_tensor.accessor<float, 2>();
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < desc_dim; ++j) {
            descriptors.at<float>(i, j) = desc_acc[i][j];
        }
    }
}

void computeSuperPoint(const cv::Mat& image, torch::jit::script::Module& model,
                        const SuperPointParameters& params,
                        std::vector<cv::KeyPoint>& keypoints, cv::Mat& descriptors)
{
    cv::Mat gray;
    if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = image; // already single-channel
    }

    auto detect_fn = [&](const cv::Mat& tile, std::vector<cv::KeyPoint>& kps, cv::Mat& desc) {
        computeSuperPointTile(tile, model, kps, desc);
    };

    std::vector<TiledKeypoint> tiled_kps;
    computeTiledKeypoints(gray, params.tile_size, params.tile_overlap, detect_fn, tiled_kps, descriptors);
    deduplicateKeypoints(tiled_kps, descriptors, KEYPOINT_DEDUP_RADIUS_PX);
    // NOTE: the currently scripted model (superpoint_batchsize1_scriptable.pt)
    // was exported with max_num_keypoints=None, so there's no per-tile cap
    // baked into the model itself. This global cap applies params.max_num_keypoints
    // across the whole (deduplicated) image instead.
    capKeypointsByScore(tiled_kps, descriptors, params.max_num_keypoints);

    keypoints.clear();
    keypoints.reserve(tiled_kps.size());
    for (auto& tk : tiled_kps) keypoints.push_back(tk.kp);
}

// ============================================================================
// Shared matching + RANSAC
// ============================================================================

void getFLANNMatches(const cv::Mat& descriptors1, const cv::Mat& descriptors2,
                      std::vector<std::vector<cv::DMatch>>& matches12,
                      std::vector<std::vector<cv::DMatch>>& matches21)
{
    cv::FlannBasedMatcher matcher(cv::makePtr<cv::flann::KDTreeIndexParams>(5),
                                   cv::makePtr<cv::flann::SearchParams>(50, 0, true));
    matcher.knnMatch(descriptors1, descriptors2, matches12, 2);
    matcher.knnMatch(descriptors2, descriptors1, matches21, 2);
}

// Lowe's ratio test in both directions, then a mutual cross-check: a match
// only survives if each descriptor is the other's best match in both directions.
std::vector<cv::DMatch> filterMatchesRatioAndCrossCheck(
    const std::vector<std::vector<cv::DMatch>>& matches12,
    const std::vector<std::vector<cv::DMatch>>& matches21,
    float lowe_ratio)
{
    std::vector<cv::DMatch> good12;
    for (auto& m : matches12) {
        if (m.size() < 2) continue;
        if (m[0].distance < lowe_ratio * m[1].distance) good12.push_back(m[0]);
    }

    std::vector<int> best_match_21(matches21.size(), -1);
    for (size_t idx2 = 0; idx2 < matches21.size(); ++idx2) {
        auto& m = matches21[idx2];
        if (m.size() < 2) continue;
        if (m[0].distance < lowe_ratio * m[1].distance) {
            best_match_21[idx2] = m[0].trainIdx; // index back into image1's descriptors
        }
    }

    std::vector<cv::DMatch> mutual_matches;
    for (auto& m : good12) {
        int idx1 = m.queryIdx;
        int idx2 = m.trainIdx;
        if (idx2 < (int)best_match_21.size() && best_match_21[idx2] == idx1) {
            mutual_matches.push_back(m);
        }
    }
    return mutual_matches;
}

cv::Mat estimateHomographyRANSAC(const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
                                  const std::vector<cv::DMatch>& matches,
                                  const RansacParameters& params,
                                  std::vector<cv::DMatch>& inliers, std::vector<cv::DMatch>& outliers)
{
    inliers.clear();
    outliers.clear();

    if (matches.size() < 4) {
        outliers = matches;
        return cv::Mat();
    }

    std::vector<cv::Point2f> pts1, pts2;
    pts1.reserve(matches.size());
    pts2.reserve(matches.size());
    for (auto& m : matches) {
        pts1.push_back(kp1[m.queryIdx].pt);
        pts2.push_back(kp2[m.trainIdx].pt);
    }

    std::vector<uchar> inlier_mask;
    cv::Mat H = cv::findHomography(pts1, pts2, cv::RANSAC, params.pixel_threshold,
                                    inlier_mask, params.max_iters, params.confidence);

    for (size_t i = 0; i < matches.size(); ++i) {
        if (inlier_mask[i]) inliers.push_back(matches[i]);
        else outliers.push_back(matches[i]);
    }
    return H;
}

// ============================================================================
// Output: visualizations, flicker GIF, summary log
// ============================================================================

void saveKeypointVisualization(const cv::Mat& image, const std::vector<cv::KeyPoint>& kps,
                                const std::string& out_path)
{
    cv::Mat out;
    cv::drawKeypoints(image, kps, out, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DEFAULT);
    cv::imwrite(out_path, out);
}

void saveMatchVisualization(const cv::Mat& img1, const cv::Mat& img2,
                             const std::vector<cv::KeyPoint>& kp1, const std::vector<cv::KeyPoint>& kp2,
                             const std::vector<cv::DMatch>& inliers, const std::vector<cv::DMatch>& outliers,
                             const std::string& out_path)
{
    cv::Mat vis;
    // Draw outliers first (red)...
    cv::drawMatches(img1, kp1, img2, kp2, outliers, vis, cv::Scalar(0, 0, 255), cv::Scalar(-1),
                     std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    // ...then inliers (green) drawn over the same canvas via DRAW_OVER_OUTIMG,
    // instead of blending two separately-rendered images.
    cv::drawMatches(img1, kp1, img2, kp2, inliers, vis, cv::Scalar(0, 255, 0), cv::Scalar(-1),
                     std::vector<char>(),
                     cv::DrawMatchesFlags::DRAW_OVER_OUTIMG | cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imwrite(out_path, vis);
}

// Opacity of img2 in the "blink" frame. 1.0 = fully opaque (old alternating
// behavior), lower values let img1 show through even while img2 is visible.
constexpr float FLICKER_OVERLAY_ALPHA = 0.5f;

cv::Mat normalizeToBGR8(const cv::Mat& mat)
{
    cv::Mat depth_fixed;
    if (mat.depth() == CV_16U) mat.convertTo(depth_fixed, CV_8U, 255.0 / 65535.0);
    else mat.convertTo(depth_fixed, CV_8U);

    cv::Mat bgr;
    if (depth_fixed.channels() == 1)      cv::cvtColor(depth_fixed, bgr, cv::COLOR_GRAY2BGR);
    else if (depth_fixed.channels() == 4) cv::cvtColor(depth_fixed, bgr, cv::COLOR_BGRA2BGR);
    else                                    bgr = depth_fixed;
    return bgr;
}

cv::Mat toRGBA8(const cv::Mat& mat)
{
    cv::Mat bgr = normalizeToBGR8(mat);
    cv::Mat rgba;
    cv::cvtColor(bgr, rgba, cv::COLOR_BGR2RGBA);
    return rgba;
}

void saveFlickerGIF(const cv::Mat& img1, const cv::Mat& img2, const cv::Mat& homography,
                     const std::string& out_path, uint32_t frame_delay_csec = 50)
{
    cv::Mat img2_aligned;
    if (!homography.empty()) {
        // homography maps img1 -> img2 (from findHomography(pts1, pts2, ...)).
        // To resample img2 into img1's coordinate frame we need the inverse.
        cv::warpPerspective(img2, img2_aligned, homography.inv(), img1.size());
    } else {
        cv::resize(img2, img2_aligned, img1.size());
    }

    cv::Mat img1_bgr8 = normalizeToBGR8(img1);
    cv::Mat img2_bgr8 = normalizeToBGR8(img2_aligned);

    // "Blink" frame: img1 stays fully visible, img2 is blended in at reduced
    // opacity on top of it -- so misalignment is visible as a translucent
    // ghost/double-edge rather than a hard cut between two separate images.
    cv::Mat blended;
    cv::addWeighted(img1_bgr8, 1.0 - FLICKER_OVERLAY_ALPHA, img2_bgr8, FLICKER_OVERLAY_ALPHA, 0.0, blended);

    cv::Mat frame_base = toRGBA8(img1_bgr8);   // img1 alone
    cv::Mat frame_blink = toRGBA8(blended);    // img1 + translucent img2 overlay

    GifWriter writer = {};
    GifBegin(&writer, out_path.c_str(), frame_base.cols, frame_base.rows, frame_delay_csec);
    GifWriteFrame(&writer, frame_base.data, frame_base.cols, frame_base.rows, frame_delay_csec);
    GifWriteFrame(&writer, frame_blink.data, frame_blink.cols, frame_blink.rows, frame_delay_csec);
    GifEnd(&writer);
}

struct PipelineResult
{
    std::string name;
    int num_keypoints1 = 0;
    int num_keypoints2 = 0;
    int num_raw_matches = 0;
    int num_inliers = 0;
    int num_outliers = 0;
    double detect_time_sec = 0.0;
    double match_time_sec = 0.0;
    double ransac_time_sec = 0.0;
};

void writeComparisonSummary(const std::string& out_path, const CLIArgs& args,
                             const PipelineResult& sift_result, const PipelineResult& sp_result)
{
    std::ofstream f(out_path);
    f << "=== SIFT vs SuperPoint Comparison Summary ===\n\n";

    f << "Input images:\n";
    f << "  img1: " << args.img1_path << "\n";
    f << "  img2: " << args.img2_path << "\n\n";

    f << "Preprocessing applied to img1:\n";
    f << "  rotate: " << args.rotate1_deg << " deg\n";
    f << "  flip: " << args.flip1 << "\n";
    f << "  brightness: "
      << (args.brightness1 ? std::to_string(*args.brightness1) : std::string("disabled")) << "\n\n";

    f << "Preprocessing applied to img2:\n";
    f << "  rotate: " << args.rotate2_deg << " deg\n";
    f << "  flip: " << args.flip2 << "\n";
    f << "  brightness: "
      << (args.brightness2 ? std::to_string(*args.brightness2) : std::string("disabled")) << "\n\n";

    f << "Rescale (both images): "
      << (args.target_mp ? std::to_string(*args.target_mp) + " MP target" : std::string("disabled")) << "\n\n";

    for (const auto& r : { sift_result, sp_result }) {
        f << "--- " << r.name << " ---\n";
        f << "  keypoints img1: " << r.num_keypoints1 << "\n";
        f << "  keypoints img2: " << r.num_keypoints2 << "\n";
        f << "  raw matches (ratio + cross-check): " << r.num_raw_matches << "\n";
        f << "  RANSAC inliers: " << r.num_inliers << "\n";
        f << "  RANSAC outliers: " << r.num_outliers << "\n";
        f << "  detection time: " << r.detect_time_sec << " s\n";
        f << "  matching time: " << r.match_time_sec << " s\n";
        f << "  RANSAC time: " << r.ransac_time_sec << " s\n\n";
    }
}

// ============================================================================
// main
// ============================================================================

int main(int argc, char** argv)
{
    CLIArgs args;
    try {
        args = parseArgs(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Argument error: " << e.what() << "\n" << USAGE;
        return 1;
    }

    cv::Mat img1_raw = cv::imread(args.img1_path, cv::IMREAD_UNCHANGED);
    cv::Mat img2_raw = cv::imread(args.img2_path, cv::IMREAD_UNCHANGED);
    if (img1_raw.empty() || img2_raw.empty()) {
        std::cerr << "Failed to load one or both images.\n";
        return 1;
    }
    std::cout << "img1 loaded: " << img1_raw.cols << "x" << img1_raw.rows
              << ", channels=" << img1_raw.channels() << ", depth=" << img1_raw.depth() << "\n";
    std::cout << "img2 loaded: " << img2_raw.cols << "x" << img2_raw.rows
              << ", channels=" << img2_raw.channels() << ", depth=" << img2_raw.depth() << "\n";

    cv::Mat img1 = preprocessImage(img1_raw, args.rotate1_deg, args.flip1, args.brightness1);
    cv::Mat img2 = preprocessImage(img2_raw, args.rotate2_deg, args.flip2, args.brightness2);

    if (args.target_mp) {
        img1 = rescaleToTargetMP(img1, *args.target_mp);
        img2 = rescaleToTargetMP(img2, *args.target_mp);
    }

    fs::path exe_dir = fs::absolute(fs::path(argv[0])).parent_path();
    fs::path results_base = exe_dir.parent_path().parent_path().parent_path().parent_path()
                             / "Results" / "C++" / "SiftvsSuperPoint";

    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm now_tm{};
#if defined(_WIN32)
    localtime_s(&now_tm, &now_c);
#else
    localtime_r(&now_c, &now_tm);
#endif
    std::ostringstream run_id;
    run_id << std::put_time(&now_tm, "%Y%m%d_%H%M%S");

    fs::path output_dir = results_base / run_id.str();
    fs::create_directories(output_dir);

    SIFTParameters sift_params;
    SuperPointParameters sp_params;
    RansacParameters ransac_params;

    // ---------------- SIFT pipeline ----------------
    PipelineResult sift_result;
    sift_result.name = "SIFT";

    std::vector<cv::KeyPoint> sift_kp1, sift_kp2;
    cv::Mat sift_desc1, sift_desc2;

    auto t0 = std::chrono::steady_clock::now();
    computeSIFT(img1, sift_params, sift_kp1, sift_desc1);
    computeSIFT(img2, sift_params, sift_kp2, sift_desc2);
    auto t1 = std::chrono::steady_clock::now();
    sift_result.detect_time_sec = std::chrono::duration<double>(t1 - t0).count();
    sift_result.num_keypoints1 = (int)sift_kp1.size();
    sift_result.num_keypoints2 = (int)sift_kp2.size();

    std::vector<std::vector<cv::DMatch>> sift_m12, sift_m21;
    t0 = std::chrono::steady_clock::now();
    getFLANNMatches(sift_desc1, sift_desc2, sift_m12, sift_m21);
    std::vector<cv::DMatch> sift_matches = filterMatchesRatioAndCrossCheck(sift_m12, sift_m21, LOWE_RATIO);
    t1 = std::chrono::steady_clock::now();
    sift_result.match_time_sec = std::chrono::duration<double>(t1 - t0).count();
    sift_result.num_raw_matches = (int)sift_matches.size();

    std::vector<cv::DMatch> sift_inliers, sift_outliers;
    t0 = std::chrono::steady_clock::now();
    cv::Mat sift_H = estimateHomographyRANSAC(sift_kp1, sift_kp2, sift_matches, ransac_params,
                                               sift_inliers, sift_outliers);
    t1 = std::chrono::steady_clock::now();
    sift_result.ransac_time_sec = std::chrono::duration<double>(t1 - t0).count();
    sift_result.num_inliers = (int)sift_inliers.size();
    sift_result.num_outliers = (int)sift_outliers.size();

    saveKeypointVisualization(img1, sift_kp1, (output_dir / "sift_keypoints_img1.png").string());
    saveKeypointVisualization(img2, sift_kp2, (output_dir / "sift_keypoints_img2.png").string());
    saveMatchVisualization(img1, img2, sift_kp1, sift_kp2, sift_inliers, sift_outliers,
                            (output_dir / "sift_matches.png").string());
    saveFlickerGIF(img1, img2, sift_H, (output_dir / "sift_flicker.gif").string());

    // ---------------- SuperPoint pipeline ----------------
    PipelineResult sp_result;
    sp_result.name = "SuperPoint";

    torch::jit::script::Module model;
    try {
        fs::path model_path = exe_dir / SUPERPOINT_MODEL_FILENAME;
        model = torch::jit::load(model_path.string());
        model.eval();
    } catch (const c10::Error& e) {
        std::cerr << "Failed to load SuperPoint model: " << e.what() << "\n";
        return 1;
    }

    std::vector<cv::KeyPoint> sp_kp1, sp_kp2;
    cv::Mat sp_desc1, sp_desc2;

    try {
        t0 = std::chrono::steady_clock::now();
        computeSuperPoint(img1, model, sp_params, sp_kp1, sp_desc1);
        computeSuperPoint(img2, model, sp_params, sp_kp2, sp_desc2);
        t1 = std::chrono::steady_clock::now();
    } catch (const c10::Error& e) {
        std::cerr << "SuperPoint inference failed (c10::Error): " << e.what() << "\n";
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "SuperPoint inference failed: " << e.what() << "\n";
        return 1;
    }
    sp_result.detect_time_sec = std::chrono::duration<double>(t1 - t0).count();
    sp_result.num_keypoints1 = (int)sp_kp1.size();
    sp_result.num_keypoints2 = (int)sp_kp2.size();

    std::vector<std::vector<cv::DMatch>> sp_m12, sp_m21;
    t0 = std::chrono::steady_clock::now();
    getFLANNMatches(sp_desc1, sp_desc2, sp_m12, sp_m21);
    std::vector<cv::DMatch> sp_matches = filterMatchesRatioAndCrossCheck(sp_m12, sp_m21, LOWE_RATIO);
    t1 = std::chrono::steady_clock::now();
    sp_result.match_time_sec = std::chrono::duration<double>(t1 - t0).count();
    sp_result.num_raw_matches = (int)sp_matches.size();

    std::vector<cv::DMatch> sp_inliers, sp_outliers;
    t0 = std::chrono::steady_clock::now();
    cv::Mat sp_H = estimateHomographyRANSAC(sp_kp1, sp_kp2, sp_matches, ransac_params,
                                             sp_inliers, sp_outliers);
    t1 = std::chrono::steady_clock::now();
    sp_result.ransac_time_sec = std::chrono::duration<double>(t1 - t0).count();
    sp_result.num_inliers = (int)sp_inliers.size();
    sp_result.num_outliers = (int)sp_outliers.size();

    saveKeypointVisualization(img1, sp_kp1, (output_dir / "superpoint_keypoints_img1.png").string());
    saveKeypointVisualization(img2, sp_kp2, (output_dir / "superpoint_keypoints_img2.png").string());
    saveMatchVisualization(img1, img2, sp_kp1, sp_kp2, sp_inliers, sp_outliers,
                            (output_dir / "superpoint_matches.png").string());
    saveFlickerGIF(img1, img2, sp_H, (output_dir / "superpoint_flicker.gif").string());

    writeComparisonSummary((output_dir / "comparison_summary.txt").string(), args, sift_result, sp_result);

    std::cout << "Done. Results written to: " << output_dir.string() << "\n";
    return 0;
}