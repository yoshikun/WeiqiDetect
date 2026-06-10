def order_points(pts):
    import numpy as np

    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    rect[0] = pts[s.argmin()]
    rect[2] = pts[s.argmax()]
    rect[1] = pts[diff.argmin()]
    rect[3] = pts[diff.argmax()]
    return rect


def _score_quad(quad, width, height):
    import cv2
    import numpy as np

    area = cv2.contourArea(quad)
    img_area = float(width * height)
    if area < img_area * 0.08 or area > img_area * 0.99:
        return -1.0

    w_top = np.linalg.norm(quad[1] - quad[0])
    w_bottom = np.linalg.norm(quad[2] - quad[3])
    h_left = np.linalg.norm(quad[3] - quad[0])
    h_right = np.linalg.norm(quad[2] - quad[1])
    if min(w_top, w_bottom, h_left, h_right) < 1:
        return -1.0

    ratio = max(w_top, w_bottom) / max(min(w_top, w_bottom), 1.0)
    ratio_h = max(h_left, h_right) / max(min(h_left, h_right), 1.0)
    if ratio > 1.8 or ratio_h > 1.8:
        return -1.0

    aspect = (w_top + w_bottom) * 0.5 / max((h_left + h_right) * 0.5, 1.0)
    aspect_score = 1.0 - min(abs(aspect - 1.0), 1.0)
    area_score = min(1.0, area / (img_area * 0.75))
    return area_score * 0.65 + aspect_score * 0.35


def _quad_from_contour(contour, scale_factor):
    import cv2
    import numpy as np

    peri = cv2.arcLength(contour, True)
    for eps in (0.015, 0.02, 0.03, 0.04, 0.05):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return order_points(approx.reshape(4, 2).astype(np.float32) * scale_factor)
    return None


def _quad_from_mask(mask, scale_factor):
    import cv2
    import numpy as np

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    quad = _quad_from_contour(contour, scale_factor)
    if quad is not None:
        return quad

    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect) * scale_factor
    return order_points(box.astype(np.float32))


def _build_board_mask(small):
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    wood1 = cv2.inRange(hsv, (8, 20, 50), (35, 200, 255))
    wood2 = cv2.inRange(hsv, (0, 0, 60), (25, 80, 220))
    mask = cv2.bitwise_or(wood1, wood2)

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 30, 120)
    mask = cv2.bitwise_or(mask, edges)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _quad_from_edges(small, scale_factor):
    import cv2

    mask = _build_board_mask(small)
    return _quad_from_mask(mask, scale_factor)


def _quad_from_largest_edge_contour(small, scale_factor):
    import cv2

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 25, 100)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_area = small.shape[0] * small.shape[1]
    best = None
    best_score = -1.0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < img_area * 0.1:
            continue
        quad = _quad_from_contour(contour, scale_factor)
        if quad is None:
            continue
        score = _score_quad(quad, small.shape[1] * scale_factor, small.shape[0] * scale_factor)
        if score > best_score:
            best_score = score
            best = quad
    return best


def _fallback_quad(width, height):
    import numpy as np

    margin = 0.04
    return order_points(
        np.array(
            [
                [width * margin, height * margin],
                [width * (1 - margin), height * margin],
                [width * (1 - margin), height * (1 - margin)],
                [width * margin, height * (1 - margin)],
            ],
            dtype=np.float32,
        )
    )


def find_board_quad(img):
    import cv2

    height, width = img.shape[:2]
    scale = min(1.0, 1000.0 / max(height, width))
    small = cv2.resize(img, (int(width * scale), int(height * scale))) if scale < 1 else img
    scale_factor = width / small.shape[1] if scale < 1 else 1.0

    candidates = []
    for finder in (_quad_from_largest_edge_contour, _quad_from_edges):
        quad = finder(small, scale_factor)
        if quad is None:
            continue
        score = _score_quad(quad, width, height)
        if score > 0:
            candidates.append((score, quad))

    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    return _fallback_quad(width, height)


def warp_board(img, quad, size=960):
    import cv2
    import numpy as np

    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img, matrix, (size, size), flags=cv2.INTER_LINEAR)
    return warped, matrix


def _smooth_1d(values, window=9):
    import numpy as np

    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(values, kernel, mode="same")


def _pick_peaks(profile, count):
    import numpy as np

    smoothed = _smooth_1d(profile.astype(np.float32), max(5, len(profile) // 80 | 1))
    if smoothed.max() <= 0:
        return None

    candidates = []
    for idx in range(1, len(smoothed) - 1):
        if smoothed[idx] >= smoothed[idx - 1] and smoothed[idx] >= smoothed[idx + 1]:
            candidates.append((float(smoothed[idx]), idx))

    if len(candidates) < count:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    strong = candidates[: max(count * 3, count + 4)]
    strong.sort(key=lambda item: item[1])
    if len(strong) < count:
        return None

    positions = np.array([item[1] for item in strong], dtype=np.float32)
    if len(positions) == count:
        return positions

    step = (len(profile) - 1) / max(count - 1, 1)
    target = np.array([step * i for i in range(count)], dtype=np.float32)
    chosen = []
    used = set()
    for target_pos in target:
        best_idx = None
        best_dist = 1e9
        for pos in positions:
            pos_int = int(pos)
            if pos_int in used:
                continue
            dist = abs(pos - target_pos)
            if dist < best_dist:
                best_dist = dist
                best_idx = pos_int
        if best_idx is None:
            return None
        used.add(best_idx)
        chosen.append(best_idx)

    if len(chosen) != count:
        return None
    return np.array(chosen, dtype=np.float32)


def _grid_positions_from_projection(gray, board_size, axis):
    import numpy as np

    inverted = 255 - gray
    profile = inverted.sum(axis=axis).astype(np.float32)
    return _pick_peaks(profile, board_size)


def _grid_positions_from_hough(gray, board_size):
    import cv2
    import numpy as np

    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 40, 130)
    min_len = max(gray.shape[0], gray.shape[1]) // (board_size + 1)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(40, min_len // 2),
        minLineLength=min_len,
        maxLineGap=12,
    )
    if lines is None:
        return None, None

    xs = []
    ys = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        dx = x2 - x1
        dy = y2 - y1
        length = float(np.hypot(dx, dy))
        if length < min_len * 0.6:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle <= 20 or angle >= 160:
            ys.append((y1 + y2) * 0.5)
        elif 70 <= angle <= 110:
            xs.append((x1 + x2) * 0.5)

    if len(xs) < board_size or len(ys) < board_size:
        return None, None

    def cluster(values, count):
        values = np.array(sorted(values), dtype=np.float32)
        if len(values) < count:
            return None
        groups = []
        current = [values[0]]
        gap = max(8.0, (values[-1] - values[0]) / (count * 1.5))
        for value in values[1:]:
            if value - current[-1] <= gap:
                current.append(value)
            else:
                groups.append(float(np.mean(current)))
                current = [value]
        groups.append(float(np.mean(current)))
        if len(groups) < count:
            return None
        groups = np.array(groups, dtype=np.float32)
        if len(groups) == count:
            return groups
        step = (len(groups) - 1) / max(count - 1, 1)
        indices = np.round(np.arange(count) * step).astype(int)
        return groups[indices]

    return cluster(xs, board_size), cluster(ys, board_size)


def detect_grid_positions(gray, board_size):
    xs, ys = _grid_positions_from_hough(gray, board_size)
    if xs is not None and ys is not None:
        return xs, ys, "hough"

    xs = _grid_positions_from_projection(gray, board_size, axis=0)
    ys = _grid_positions_from_projection(gray, board_size, axis=1)
    if xs is not None and ys is not None:
        return xs, ys, "projection"

    height, width = gray.shape[:2]
    xs = np.array([(i + 0.5) * width / board_size for i in range(board_size)], dtype=np.float32)
    ys = np.array([(i + 0.5) * height / board_size for i in range(board_size)], dtype=np.float32)
    return xs, ys, "uniform"


def intersection_points(board_size, width, height, xs=None, ys=None):
    import numpy as np

    if xs is None or ys is None:
        xs = np.array([(i + 0.5) * width / board_size for i in range(board_size)], dtype=np.float32)
        ys = np.array([(i + 0.5) * height / board_size for i in range(board_size)], dtype=np.float32)

    points = []
    for gy, py in enumerate(ys):
        for gx, px in enumerate(xs):
            points.append((gx, gy, int(round(px)), int(round(py))))
    return points


def crop_square_patch(img, px, py, patch_size):
    import cv2

    height, width = img.shape[:2]
    half = patch_size // 2
    x0 = px - half
    y0 = py - half
    x1 = x0 + patch_size
    y1 = y0 + patch_size

    pad_left = max(0, -x0)
    pad_top = max(0, -y0)
    pad_right = max(0, x1 - width)
    pad_bottom = max(0, y1 - height)

    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(width, x1)
    y1 = min(height, y1)

    patch = img[y0:y1, x0:x1]
    if pad_left or pad_top or pad_right or pad_bottom:
        patch = cv2.copyMakeBorder(
            patch,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_REFLECT_101,
        )

    if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
        patch = cv2.resize(patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    return patch
