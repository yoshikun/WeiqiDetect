import numpy as np

from board_geometry import order_points

BOARD_INSET = 0.06


def _compute_saturation(bgr):
    bgr = bgr.astype(np.float32)
    mx = bgr.max(axis=2)
    mn = bgr.min(axis=2)
    sat = np.zeros_like(mx)
    np.divide(mx - mn, mx, out=sat, where=mx > 0)
    return sat


def _board_mask(bgr, gray, sat_threshold=0.1, bright_max=235, bright_min=35, dilate_radius=5):
    import cv2

    sat = _compute_saturation(bgr)
    mask = ((sat > sat_threshold) & (gray < bright_max) & (gray > bright_min)).astype(np.uint8)
    if dilate_radius <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_radius * 2 + 1, dilate_radius * 2 + 1))
    return cv2.dilate(mask, kernel)


def _quad_from_mask(mask, width, height):
    tl_score = float("inf")
    tr_score = float("-inf")
    br_score = float("-inf")
    bl_score = float("inf")
    tl = tr = br = bl = None
    boundary_count = 0

    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if not mask[y, x]:
                continue
            if mask[y - 1, x] and mask[y + 1, x] and mask[y, x - 1] and mask[y, x + 1]:
                continue
            boundary_count += 1
            total = x + y
            diff = x - y
            if total < tl_score:
                tl_score = total
                tl = (x, y)
            if total > br_score:
                br_score = total
                br = (x, y)
            if diff > tr_score:
                tr_score = diff
                tr = (x, y)
            if diff < bl_score:
                bl_score = diff
                bl = (x, y)

    if boundary_count < 20 or None in (tl, tr, br, bl):
        return None

    area = _quad_area([tl, tr, br, bl])
    if area < width * height * 0.05:
        return None
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _quad_area(points):
    pts = np.array(points, dtype=np.float32)
    w_top = np.linalg.norm(pts[1] - pts[0])
    w_bottom = np.linalg.norm(pts[2] - pts[3])
    h_left = np.linalg.norm(pts[3] - pts[0])
    h_right = np.linalg.norm(pts[2] - pts[1])
    return (w_top * h_left + w_bottom * h_right) * 0.25


def inset_image_corners(width, height, fraction=0.05):
    margin = min(width, height) * fraction
    return order_points(
        np.array(
            [
                [margin, margin],
                [width - 1 - margin, margin],
                [width - 1 - margin, height - 1 - margin],
                [margin, height - 1 - margin],
            ],
            dtype=np.float32,
        )
    )


def estimate_grid_in_warped(size):
    margin = size * BOARD_INSET
    return np.array(
        [
            [margin, margin],
            [size - margin, margin],
            [size - margin, size - margin],
            [margin, size - margin],
        ],
        dtype=np.float32,
    )


def find_board_corners(img_bgr, max_dim=600):
    import cv2

    height, width = img_bgr.shape[:2]
    scale = min(1.0, max_dim / max(height, width))
    if scale < 1.0:
        small = cv2.resize(img_bgr, (int(width * scale), int(height * scale)))
        scale_factor = width / small.shape[1]
    else:
        small = img_bgr
        scale_factor = 1.0

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mask = _board_mask(small, gray)
    quad = _quad_from_mask(mask, small.shape[1], small.shape[0])
    if quad is None:
        return None
    return order_points(quad * scale_factor)
