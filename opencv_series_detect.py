"""
Classic OpenCV Go board recognition (Zhihu/CSDN series style).

Pipeline:
1. HSV wood-color mask + contour to locate board region
2. Perspective warp to square grid
3. Per-cell black/white pixel ratio on fixed grid (no deep model)
"""

import base64

import numpy as np

from board_geometry import order_points, warp_board

WARP_SIZE = 800
BOARD_INSET = 0.06


def _preview_base64(warped_bgr):
    import cv2

    small = cv2.resize(warped_bgr, (320, 320), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", small, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return None
    return base64.b64encode(buf.tobytes()).decode("ascii")


def find_board_quad_hsv(img_bgr):
    """Locate board via HSV wood mask + largest contour quad."""
    import cv2

    height, width = img_bgr.shape[:2]
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (10, 0, 0), (40, 255, 255))
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    gray = cv2.cvtColor(cv2.bitwise_and(img_bgr, img_bgr, mask=mask), cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < width * height * 0.08:
        return None

    peri = cv2.arcLength(contour, True)
    for eps_ratio in (0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(contour, eps_ratio * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_points(approx.reshape(4, 2).astype(np.float32))

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    return order_points(box.astype(np.float32))


def _black_ratio(patch_bgr):
    import cv2

    gray = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2GRAY)
    return float((gray < 85).mean())


def _white_ratio(patch_bgr):
    import cv2

    hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask = (sat < 55) & (val > 165)
    return float(mask.mean())


def classify_grid_cells(warped_bgr, board_size):
    """Per-cell occupancy via black/white area ratio (classic grid scan)."""
    import cv2

    size = board_size * 38
    warped = cv2.resize(warped_bgr, (size, size), interpolation=cv2.INTER_AREA)
    cell = size // board_size
    stone_d = max(8, int(cell * 0.95))

    black = []
    white = []
    confidences = []

    for row in range(board_size):
        for col in range(board_size):
            x0 = col * cell
            y0 = row * cell
            patch = warped[y0 : y0 + stone_d, x0 : x0 + stone_d]
            if patch.size == 0:
                continue

            black_ratio = _black_ratio(patch)
            if black_ratio > 0.45:
                black.append([col, row])
                confidences.append(min(1.0, black_ratio))
                continue

            white_ratio = _white_ratio(patch)
            if white_ratio > 0.12:
                white.append([col, row])
                confidences.append(min(1.0, white_ratio * 2.0))

    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0
    return black, white, avg_conf


def detect_board_opencv_series(img_bgr, board_size=19, corners=None, with_preview=False):
    """Full detect using classic OpenCV grid pipeline."""
    from kaya_corners import inset_image_corners

    height, width = img_bgr.shape[:2]
    quad = None
    corner_source = "opencv-hsv"
    corners_detected = False

    if corners and len(corners) == 4:
        quad = order_points(np.array(corners, dtype=np.float32).reshape(4, 2))
        corner_source = "manual"
        corners_detected = True
    else:
        quad = find_board_quad_hsv(img_bgr)
        if quad is not None:
            corners_detected = True
        else:
            quad = inset_image_corners(width, height, BOARD_INSET)
            corner_source = "fallback"
            corners_detected = False

    warped, _ = warp_board(img_bgr, quad, WARP_SIZE)
    black, white, avg_conf = classify_grid_cells(warped, board_size)

    result = {
        "ok": True,
        "boardSize": board_size,
        "black": black,
        "white": white,
        "confidence": avg_conf,
        "detector": "opencv-series",
        "pipeline": "opencv-hsv+grid-ratio",
        "corners": quad.reshape(-1).tolist(),
        "cornersDetected": corners_detected,
        "stats": {
            "blackCount": len(black),
            "whiteCount": len(white),
            "cornerSource": corner_source,
            "stoneSource": "grid-ratio",
            "gridMethod": "fixed-cell",
            "warpSize": WARP_SIZE,
            "imageWidth": width,
            "imageHeight": height,
        },
    }
    if with_preview:
        preview = _preview_base64(warped)
        if preview:
            result["warpPreview"] = preview
    return result
