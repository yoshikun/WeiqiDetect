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


def find_board_quad(img):
    import cv2
    import numpy as np

    height, width = img.shape[:2]
    scale = min(1.0, 900.0 / max(height, width))
    small = cv2.resize(img, (int(width * scale), int(height * scale))) if scale < 1 else img
    scale_factor = width / small.shape[1] if scale < 1 else 1.0

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 35, 110)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    img_area = small.shape[0] * small.shape[1]
    best = None
    best_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < img_area * 0.12 or area > img_area * 0.98:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) != 4:
            continue
        if area > best_area:
            best_area = area
            best = approx.reshape(4, 2).astype(np.float32)

    if best is not None:
        return order_points(best * scale_factor)

    margin = 0.06
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


def warp_board(img, quad, size=900):
    import cv2
    import numpy as np

    dst = np.array(
        [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(quad, dst)
    return cv2.warpPerspective(img, matrix, (size, size))


def intersection_points(board_size, width, height):
    points = []
    for gy in range(board_size):
        for gx in range(board_size):
            px = int((gx + 0.5) * width / board_size)
            py = int((gy + 0.5) * height / board_size)
            points.append((gx, gy, px, py))
    return points


def crop_square_patch(img, px, py, patch_size):
    import cv2
    import numpy as np

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
