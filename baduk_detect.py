import base64

import numpy as np

from board_geometry import (
    detect_grid_positions,
    find_board_quad,
    find_board_quad_candidates,
    order_points,
    score_grid_alignment,
    warp_board,
)
from kaya_corners import estimate_grid_in_warped, find_board_corners, inset_image_corners
from kaya_stones import classify_intersections

WARP_SIZE = 800
BOARD_INSET = 0.06

try:
    from moku_detect import (
        DEFAULT_THRESHOLD,
        detect_board_moku,
        detect_stones_on_warped,
        is_ready as moku_ready,
        suggest_corners_moku,
    )
except ImportError:
    DEFAULT_THRESHOLD = 0.035
    moku_ready = lambda: False

    def suggest_corners_moku(_img, threshold=DEFAULT_THRESHOLD):
        del threshold
        return None

    def detect_board_moku(*_args, **_kwargs):
        raise RuntimeError("moku unavailable")

    def detect_stones_on_warped(*_args, **_kwargs):
        raise RuntimeError("moku unavailable")


def _parse_user_corners(corners, width, height):
    if not corners or len(corners) != 4:
        return None
    quad = np.array(corners, dtype=np.float32).reshape(4, 2)
    quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)
    return order_points(quad)


def _warp_gray(img_bgr, quad):
    import cv2

    warped, _ = warp_board(img_bgr, quad, WARP_SIZE)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return warped, gray


def _preview_base64(warped_bgr):
    import cv2

    small = cv2.resize(warped_bgr, (320, 320), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _grid_corners_from_detected(gray, board_size):
    xs, ys, method = detect_grid_positions(gray, board_size)
    if method == "uniform":
        return estimate_grid_in_warped(WARP_SIZE), method
    grid_corners = np.array(
        [
            [xs[0], ys[0]],
            [xs[-1], ys[0]],
            [xs[-1], ys[-1]],
            [xs[0], ys[-1]],
        ],
        dtype=np.float32,
    )
    return grid_corners, method


def _score_quad_for_board(img_bgr, quad, board_size):
    try:
        _, gray = _warp_gray(img_bgr, quad)
        score, _ = score_grid_alignment(gray, board_size)
        return score
    except Exception:
        return -1.0


def _pick_best_quad(img_bgr, board_size):
    height, width = img_bgr.shape[:2]
    candidates = []

    def add(source, quad, detected):
        if quad is None:
            return
        key = tuple(np.round(quad.reshape(-1), 1))
        for item in candidates:
            if item["key"] == key:
                return
        candidates.append(
            {
                "key": key,
                "source": source,
                "quad": order_points(quad.astype(np.float32)),
                "detected": detected,
            }
        )

    if moku_ready():
        add("moku", suggest_corners_moku(img_bgr), True)
    add("saturation", find_board_corners(img_bgr), True)
    try:
        from opencv_series_detect import find_board_quad_hsv

        add("opencv-hsv", find_board_quad_hsv(img_bgr), True)
    except Exception:
        pass
    try:
        add("geometry", find_board_quad(img_bgr), True)
    except Exception:
        pass
    try:
        for quad in find_board_quad_candidates(img_bgr)[:4]:
            add("geometry-candidate", quad, True)
    except Exception:
        pass

    if not candidates:
        return inset_image_corners(width, height, BOARD_INSET), "fallback", False

    best = max(
        candidates,
        key=lambda item: _score_quad_for_board(img_bgr, item["quad"], board_size),
    )
    return best["quad"], best["source"], best["detected"]


def _moku_acceptable(result, board_size):
    if not result.get("cornersDetected"):
        return False
    black = len(result.get("black") or [])
    white = len(result.get("white") or [])
    if black + white == 0:
        return False
    conf = float(result.get("confidence") or 0)
    min_conf = 0.12 if board_size >= 19 else 0.10
    if conf < min_conf:
        return False
    if black + white > board_size * board_size * 0.85:
        return False
    return True


def _result_meta(img_bgr):
    height, width = img_bgr.shape[:2]
    return {"imageWidth": width, "imageHeight": height}


def _attach_meta(result, img_bgr):
    meta = _result_meta(img_bgr)
    stats = result.setdefault("stats", {})
    stats.update(meta)
    return result


def _detect_cv_with_quad(
    img_bgr,
    board_size,
    quad,
    corner_source,
    corners_detected,
    with_preview=False,
    threshold=DEFAULT_THRESHOLD,
):
    """Warp board, detect grid lines, classify stones; optionally refine with Moku on warp."""
    warped, gray = _warp_gray(img_bgr, quad)
    grid_corners, grid_method = _grid_corners_from_detected(gray, board_size)
    black, white, confidences = classify_intersections(gray, board_size, grid_corners)
    stone_source = "kmeans"
    detector = "cv"
    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

    if moku_ready():
        try:
            moku_result = detect_stones_on_warped(warped, board_size, threshold=threshold)
            moku_black = len(moku_result.get("black") or [])
            moku_white = len(moku_result.get("white") or [])
            moku_conf = float(moku_result.get("confidence") or 0)
            cv_count = len(black) + len(white)
            moku_count = moku_black + moku_white
            if moku_count > 0 and (
                moku_count > cv_count + 1
                or (moku_count >= cv_count and moku_conf > avg_conf + 0.03)
            ):
                black = moku_result["black"]
                white = moku_result["white"]
                avg_conf = moku_conf
                stone_source = "rtdetr-warp"
                detector = "hybrid"
        except Exception:
            pass

    result = {
        "ok": True,
        "boardSize": board_size,
        "black": black,
        "white": white,
        "confidence": avg_conf,
        "detector": detector,
        "pipeline": "classic-cv",
        "corners": quad.reshape(-1).tolist(),
        "cornersDetected": corners_detected,
        "stats": {
            "blackCount": len(black),
            "whiteCount": len(white),
            "cornerSource": corner_source,
            "stoneSource": stone_source,
            "gridMethod": grid_method,
            "warpSize": WARP_SIZE,
        },
    }
    if with_preview:
        preview = _preview_base64(warped)
        if preview:
            result["warpPreview"] = preview
    return _attach_meta(result, img_bgr)


def _detect_cv_auto(img_bgr, board_size, with_preview=False, threshold=DEFAULT_THRESHOLD):
    quad, corner_source, corners_detected = _pick_best_quad(img_bgr, board_size)
    return _detect_cv_with_quad(
        img_bgr,
        board_size,
        quad,
        corner_source,
        corners_detected,
        with_preview=with_preview,
        threshold=threshold,
    )


def _enrich_moku_result(result, img_bgr, with_preview=False):
    corners_flat = result.get("corners")
    if with_preview and corners_flat:
        quad = np.array(corners_flat, dtype=np.float32).reshape(4, 2)
        warped, _ = warp_board(img_bgr, quad, WARP_SIZE)
        preview = _preview_base64(warped)
        if preview:
            result["warpPreview"] = preview
    result["pipeline"] = "moku-rtdetr"
    stats = result.setdefault("stats", {})
    stats.setdefault("stoneSource", "rtdetr")
    stats.setdefault("cornerSource", "moku")
    stats.setdefault("warpSize", WARP_SIZE)
    return _attach_meta(result, img_bgr)


def _try_detect_moku(img_bgr, board_size, threshold, with_preview=False):
    thresholds = [threshold, max(threshold * 0.65, 0.018)]
    last = None
    for value in thresholds:
        try:
            result = detect_board_moku(img_bgr, board_size, threshold=value)
            last = result
            if _moku_acceptable(result, board_size):
                return _enrich_moku_result(result, img_bgr, with_preview=with_preview)
        except Exception:
            continue
    if last is not None and len(last.get("black") or []) + len(last.get("white") or []) > 0:
        return _enrich_moku_result(last, img_bgr, with_preview=with_preview)
    return None


def suggest_corners(img_bgr):
    """Suggest board corners: pick best among Moku, saturation CV, and geometry CV."""
    height, width = img_bgr.shape[:2]
    board_size = 19
    quad, source, detected = _pick_best_quad(img_bgr, board_size)
    return {
        "ok": True,
        "corners": quad.reshape(-1).tolist(),
        "cornersDetected": detected,
        "cornerSource": source,
        "imageWidth": width,
        "imageHeight": height,
    }


def detect_board_kaya(
    img_bgr,
    board_size=19,
    corners=None,
    with_preview=False,
    threshold=DEFAULT_THRESHOLD,
):
    """
    Kaya-style pipeline:
    1. Manual corners -> grid-aligned CV (+ Moku on warp when available)
    2. Auto -> Moku RT-DETR with quality gate and threshold retry
    3. Fallback -> multi-source CV corner pick + grid-aligned k-means
    """
    height, width = img_bgr.shape[:2]
    user_quad = _parse_user_corners(corners, width, height)
    if user_quad is not None:
        return _detect_cv_with_quad(
            img_bgr,
            board_size,
            user_quad,
            "manual",
            True,
            with_preview=with_preview,
            threshold=threshold,
        )

    if moku_ready():
        moku_result = _try_detect_moku(
            img_bgr,
            board_size,
            threshold,
            with_preview=with_preview,
        )
        if moku_result is not None and _moku_acceptable(moku_result, board_size):
            return moku_result

    return _detect_cv_auto(
        img_bgr,
        board_size,
        with_preview=with_preview,
        threshold=threshold,
    )


detect_board_baduk = detect_board_kaya
