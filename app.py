from flask import Flask, jsonify, request

from detector import detect_board
from image_decode import decode_image

app = Flask(__name__)
VERSION = "3.0.0"


def health_payload():
    return {
        "ok": True,
        "service": "weiqi-detect",
        "version": VERSION,
        "detector": "circles",
    }


@app.route("/", methods=["GET", "POST"])
def root():
    return jsonify(health_payload())


@app.route("/health", methods=["GET", "POST"])
@app.route("/health/", methods=["GET", "POST"])
def health():
    return jsonify(health_payload())


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
