import base64

import cv2
import numpy as np
import requests
from flask import Flask, jsonify, request

app = Flask(__name__)


def detect_board(img, board_size=19):
    height, width = img.shape[:2]
    margin = 0.08
    x0 = int(width * margin)
    y0 = int(height * margin)
    x1 = int(width * (1 - margin))
    y1 = int(height * (1 - margin))
    crop = img[y0:y1, x0:x1]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop_h, crop_w = gray.shape

    black = []
    white = []
    radius = max(3, min(crop_w, crop_h) // (board_size * 2))

    for y in range(board_size):
        for x in range(board_size):
            px = int((x + 0.5) * crop_w / board_size)
            py = int((y + 0.5) * crop_h / board_size)
            patch = gray[
                max(0, py - radius):min(crop_h, py + radius),
                max(0, px - radius):min(crop_w, px + radius),
            ]
            if patch.size == 0:
                continue
            avg = float(patch.mean())
            if avg < 95:
                black.append([x, y])
            elif avg > 175:
                white.append([x, y])

    return {
        "ok": True,
        "boardSize": board_size,
        "black": black,
        "white": white,
        "confidence": 0.5,
    }


def decode_image(payload):
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


@app.get("/health")
def health():
    return jsonify({"ok": True, "service": "weiqi-detect"})


@app.post("/api/v1/detect")
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

    result = detect_board(img, board_size)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
