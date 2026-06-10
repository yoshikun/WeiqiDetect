import base64

import requests
from flask import Flask, jsonify, request

from cnn_classifier import is_ready, model_info
from detector import detect_board

app = Flask(__name__)
VERSION = "2.0.0"


def health_payload():
    info = model_info()
    return {
        "ok": True,
        "service": "weiqi-detect",
        "version": VERSION,
        "detector": "cnn" if info.get("ready") else "missing-model",
        "modelValAccuracy": info.get("valAccuracy"),
    }


@app.route("/", methods=["GET", "POST"])
def root():
    return jsonify(health_payload())


@app.route("/health", methods=["GET", "POST"])
@app.route("/health/", methods=["GET", "POST"])
def health():
    return jsonify(health_payload())


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
    threshold = float(payload.get("threshold") or 0.52)

    if board_size not in (9, 13, 19):
        return jsonify({"ok": False, "error": "unsupported boardSize"}), 400

    if not payload.get("imageUrl") and not payload.get("image"):
        return jsonify({"ok": False, "error": "missing image"}), 400

    if not is_ready():
        return jsonify({"ok": False, "error": "CNN model not loaded"}), 503

    img = decode_image(payload)
    if img is None:
        return jsonify({"ok": False, "error": "invalid image"}), 400

    try:
        result = detect_board(img, board_size, threshold=threshold)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"detect failed: {exc}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
