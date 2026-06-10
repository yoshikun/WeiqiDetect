import base64

import numpy as np

from board_geometry import order_points, warp_board
from kaya_corners import estimate_grid_in_warped, find_board_corners, inset_image_corners
from kaya_stones import classify_intersections

WARP_SIZE = 800

try:
    from moku_detect import is_ready as moku_ready, suggest_corners_moku
except ImportError:
    moku_ready = lambda: False

    def suggest_corners_moku(_img):
        return None


def _parse_user_corners(corners, width, height):
    if not corners or len(corners) != 4:
        return None
    quad = np.array(corners, dtype=np.float32).reshape(4, 2)
    quad[:, 0] = np.clip(quad[:, 0], 0, width - 1)
    quad[:, 1] = np.clip(quad[:, 1], 0, height - 1)
    return order_points(quad)


def suggest_corners(img_bgr):
    height, width = img_bgr.shape[:2]
    quad = find_board_corners(img_bgr)
    source = "cv"
    if quad is None and moku_ready():
        quad = suggest_corners_moku(img_bgr)
        source = "moku"
    if quad is None:
        quad = inset_image_corners(width, height, 0.08)
        source = "fallback"
    return {
        "ok": True,
        "corners": quad.reshape(-1).tolist(),
        "cornersDetected": source != "fallback",
        "cornerSource": source,
        "imageWidth": width,
        "imageHeight": height,
    }


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


def detect_board_baduk(img_bgr, board_size=19, corners=None, with_preview=False):
    height, width = img_bgr.shape[:2]
    user_quad = _parse_user_corners(corners, width, height)

    if user_quad is not None:
        quad = user_quad
        corner_source = "manual"
        corners_detected = True
    else:
        quad = find_board_corners(img_bgr)
        corner_source = "cv"
        corners_detected = quad is not None
        if quad is None and moku_ready():
            quad = suggest_corners_moku(img_bgr)
            corner_source = "moku"
            corners_detected = quad is not None
        if quad is None:
            quad = inset_image_corners(width, height, 0.08)
            corner_source = "fallback"
            corners_detected = False

    warped, gray = _warp_gray(img_bgr, quad)
    grid_corners = estimate_grid_in_warped(WARP_SIZE)
    black, white, confidences = classify_intersections(gray, board_size, grid_corners)
    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

    result = {
        "ok": True,
        "boardSize": board_size,
        "black": black,
        "white": white,
        "confidence": avg_conf,
        "detector": "baduk",
        "corners": quad.reshape(-1).tolist(),
        "cornersDetected": corners_detected,
        "stats": {
            "blackCount": len(black),
            "whiteCount": len(white),
            "cornerSource": corner_source,
            "warpSize": WARP_SIZE,
        },
    }
    if with_preview:
        preview = _preview_base64(warped)
        if preview:
            result["warpPreview"] = preview
    return result
