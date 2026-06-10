import base64

import requests
from flask import Flask, jsonify, request

app = Flask(__name__)
VERSION = "1.2.0"


def health_payload():
    return {"ok": True, "service": "weiqi-detect", "version": VERSION}


@app.route("/", methods=["GET", "POST"])
def root():
    return jsonify(health_payload())


@app.route("/health", methods=["GET", "POST"])
@app.route("/health/", methods=["GET", "POST"])
def health():
    return jsonify(health_payload())


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
    if scale < 1:
        scale_factor = width / small.shape[1]
    else:
        scale_factor = 1.0

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


def estimate_wood_level(gray):
    import numpy as np

    height, width = gray.shape[:2]
    strip = max(6, int(min(height, width) * 0.02))
    samples = np.concatenate(
        [
            gray[:strip, :].reshape(-1),
            gray[-strip:, :].reshape(-1),
            gray[:, :strip].reshape(-1),
            gray[:, -strip:].reshape(-1),
        ]
    )
    return float(np.median(samples))


def extract_patch(gray, px, py, radius):
    import numpy as np

    height, width = gray.shape[:2]
    x0 = max(0, px - radius)
    y0 = max(0, py - radius)
    x1 = min(width, px + radius + 1)
    y1 = min(height, py + radius + 1)
    patch = gray[y0:y1, x0:x1]
    if patch.size == 0:
        return patch

    yy, xx = np.ogrid[-(py - y0) : y1 - py, -(px - x0) : x1 - px]
    mask = xx * xx + yy * yy <= radius * radius
    return patch[mask]


def classify_intersection(gray, px, py, radius, wood_level):
    import numpy as np

    patch = extract_patch(gray, px, py, radius)
    if patch.size < 9:
        return None, 0.0

    median = float(np.median(patch))
    dark_cut = wood_level - max(18, 0.22 * (255 - wood_level))
    light_cut = wood_level + max(18, 0.22 * wood_level)
    dark_frac = float(np.mean(patch < dark_cut))
    light_frac = float(np.mean(patch > light_cut))

    if dark_frac > 0.22 and median < wood_level - 12:
        confidence = min(1.0, dark_frac * 1.4 + (wood_level - median) / 80)
        return "black", confidence
    if light_frac > 0.22 and median > wood_level + 12:
        confidence = min(1.0, light_frac * 1.4 + (median - wood_level) / 80)
        return "white", confidence
    return None, 0.0


def detect_board(img, board_size=19):
    import cv2

    quad = find_board_quad(img)
    warped = warp_board(img, quad, 900)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    wood_level = estimate_wood_level(gray)

    height, width = gray.shape[:2]
    radius = max(4, int(min(height, width) / (board_size * 2.4)))

    black = []
    white = []
    confidences = []

    for gy in range(board_size):
        for gx in range(board_size):
            px = int((gx + 0.5) * width / board_size)
            py = int((gy + 0.5) * height / board_size)
            stone, confidence = classify_intersection(gray, px, py, radius, wood_level)
            if stone == "black":
                black.append([gx, gy])
                confidences.append(confidence)
            elif stone == "white":
                white.append([gx, gy])
                confidences.append(confidence)

    confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0.35

    return {
        "ok": True,
        "boardSize": board_size,
        "black": black,
        "white": white,
        "confidence": confidence,
        "stats": {
            "blackCount": len(black),
            "whiteCount": len(white),
            "woodLevel": round(wood_level, 1),
        },
    }


def decode_image(payload):
    import cv2
    import numpy as np

    image_url = payload.get("imageUrl")
    if image_url:
        try:
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            data = np.frombuffer(resp.content, dtype=np.uint8)
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            return None

    image_b64 = payload.get("image")
    if not image_b64:
        return None

    try:
        raw = base64.b64decode(image_b64)
        data = np.frombuffer(raw, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


@app.route("/api/v1/detect", methods=["POST"])
def detect():
    payload = request.get_json(silent=True) or {}
    board_size = int(payload.get("boardSize") or 19)

    if board_size not in (9, 13, 19):
        return jsonify({"ok": False, "error": "unsupported boardSize"}), 400

    if not payload.get("imageUrl") and not payload.get("image"):
        return jsonify({"ok": False, "error": "missing image"}), 400

    img = decode_image(payload)
    if img is None:
        return jsonify({"ok": False, "error": "invalid image"}), 400

    try:
        result = detect_board(img, board_size)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"detect failed: {exc}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
