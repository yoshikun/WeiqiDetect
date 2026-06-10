import numpy as np


def uniform_grid(board_size, width, height):
    xs = np.array([(i + 0.5) * width / board_size for i in range(board_size)], dtype=np.float32)
    ys = np.array([(i + 0.5) * height / board_size for i in range(board_size)], dtype=np.float32)
    return xs, ys


def snap_to_grid(cx, cy, xs, ys):
    gx = int(np.argmin(np.abs(xs - cx)))
    gy = int(np.argmin(np.abs(ys - cy)))
    dist = float(np.hypot(xs[gx] - cx, ys[gy] - cy))
    cell = float(np.mean(np.diff(xs))) if len(xs) > 1 else 1.0
    if dist > cell * 0.55:
        return None, None, dist
    return gx, gy, dist


def classify_stone_color(bgr, cx, cy, radius, wood_level):
    import cv2

    height, width = bgr.shape[:2]
    r = max(3, int(radius * 0.65))
    x0 = max(0, int(cx) - r)
    y0 = max(0, int(cy) - r)
    x1 = min(width, int(cx) + r + 1)
    y1 = min(height, int(cy) + r + 1)
    patch = bgr[y0:y1, x0:x1]
    if patch.size == 0:
        return None, 0.0

    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    median = float(np.median(gray))
    mean_bgr = patch.reshape(-1, 3).mean(axis=0)
    brightness = float(gray.mean())

    dark_cut = wood_level - max(22, 0.26 * (255 - wood_level))
    light_cut = wood_level + max(22, 0.26 * wood_level)

    if median < dark_cut and brightness < wood_level - 10:
        conf = min(1.0, (wood_level - median) / 70.0 + 0.35)
        return "black", conf
    if median > light_cut and brightness > wood_level + 10:
        conf = min(1.0, (median - wood_level) / 70.0 + 0.35)
        return "white", conf
    if float(mean_bgr.mean()) < wood_level - 18:
        return "black", 0.45
    if float(mean_bgr.mean()) > wood_level + 18:
        return "white", 0.45
    return None, 0.0


def find_stone_circles(gray, board_size):
    import cv2

    height, width = gray.shape[:2]
    cell = min(height, width) / board_size
    min_r = max(4, int(cell * 0.22))
    max_r = max(min_r + 2, int(cell * 0.50))
    min_dist = max(8, int(cell * 0.65))

    blur = cv2.GaussianBlur(gray, (9, 9), 1.5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dist,
        param1=80,
        param2=22,
        minRadius=min_r,
        maxRadius=max_r,
    )
    if circles is None:
        return []

    result = []
    for cx, cy, radius in np.round(circles[0]).astype(int):
        result.append((int(cx), int(cy), int(radius)))
    return result


def detect_stones_by_circles(warped, gray, board_size, wood_level, xs=None, ys=None):
    height, width = gray.shape[:2]
    if xs is None or ys is None:
        xs, ys = uniform_grid(board_size, width, height)

    circles = find_stone_circles(gray, board_size)
    assignments = {}

    for cx, cy, radius in circles:
        gx, gy, dist = snap_to_grid(cx, cy, xs, ys)
        if gx is None:
            continue
        stone, conf = classify_stone_color(warped, cx, cy, radius, wood_level)
        if not stone:
            continue
        conf = conf * max(0.5, 1.0 - dist / max(float(np.mean(np.diff(xs))), 1.0))
        key = (gx, gy)
        if key not in assignments or conf > assignments[key][1]:
            assignments[key] = (stone, conf)

    black = []
    white = []
    confidences = []
    for (gx, gy), (stone, conf) in assignments.items():
        if conf < 0.42:
            continue
        if stone == "black":
            black.append([gx, gy])
        else:
            white.append([gx, gy])
        confidences.append(conf)

    return black, white, confidences, len(circles)
