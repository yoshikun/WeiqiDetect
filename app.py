from flask import Flask, jsonify, request

from baduk_detect import detect_board_kaya, suggest_corners
from image_decode import decode_image
from moku_detect import is_ready as moku_ready
from opencv_series_detect import detect_board_opencv_series

app = Flask(__name__)
VERSION = "6.1.0"


def health_payload():
    return {
        "ok": True,
        "service": "weiqi-detect",
        "version": VERSION,
        "detector": "kaya",
        "pipeline": "moku-rtdetr+cv",
        "mokuReady": moku_ready(),
    }


@app.route("/", methods=["GET", "POST"])
def root():
    return jsonify(health_payload())


@app.route("/health", methods=["GET", "POST"])
@app.route("/health/", methods=["GET", "POST"])
def health():
    return jsonify(health_payload())


@app.route("/api/v1/corners", methods=["POST"])
def corners():
    payload = request.get_json(silent=True) or {}
    if not payload.get("imageUrl") and not payload.get("image"):
        return jsonify({"ok": False, "error": "missing image"}), 400

    img = decode_image(payload)
    if img is None:
        return jsonify({"ok": False, "error": "invalid image"}), 400

    try:
        return jsonify(suggest_corners(img))
    except Exception as exc:
        return jsonify({"ok": False, "error": f"corners failed: {exc}"}), 500


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

    corners = payload.get("corners")
    with_preview = bool(payload.get("withPreview"))
    threshold = float(payload.get("threshold") or 0.035)
    pipeline = (payload.get("pipeline") or "kaya").strip().lower()

    try:
        if pipeline in ("opencv-series", "opencv", "opencv_series"):
            result = detect_board_opencv_series(
                img,
                board_size,
                corners=corners,
                with_preview=with_preview,
            )
        else:
            result = detect_board_kaya(
                img,
                board_size,
                corners=corners,
                with_preview=with_preview,
                threshold=threshold,
            )
    except Exception as exc:
        return jsonify({"ok": False, "error": f"detect failed: {exc}"}), 500

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
