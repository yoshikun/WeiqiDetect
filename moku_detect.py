"""Kaya Moku RT-DETR board detection (ported from kaya-go/kaya board-recognition)."""

from __future__ import annotations

import os

import numpy as np

INPUT_SIZE = 640
NUM_QUERIES = 300
NUM_CLASSES = 3

CLASS_BLACK_STONE = 0
CLASS_WHITE_STONE = 1
CLASS_BOARD_CORNER = 2

DEFAULT_THRESHOLD = 0.035
CORNER_MIN_THRESHOLD = 0.005
WARP_OUTPUT_SIZE = 800

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "moku")
MODEL_PATH = os.path.join(MODEL_DIR, "model.onnx")

_session = None


def is_ready() -> bool:
    return os.path.exists(MODEL_PATH)


def _get_session():
    global _session
    if _session is not None:
        return _session
    if not is_ready():
        raise FileNotFoundError(f"Moku ONNX model not found: {MODEL_PATH}")
    import onnxruntime as ort

    _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _session


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def _order_corners(points):
    pts = np.array(points, dtype=np.float64)
    cx = pts[:, 0].mean()
    cy = pts[:, 1].mean()
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    order = np.argsort(angles)
    sorted_pts = pts[order]
    sums = sorted_pts[:, 0] + sorted_pts[:, 1]
    tl_idx = int(np.argmin(sums))
    return np.roll(sorted_pts, -tl_idx, axis=0)


def _inset_image_corners(width, height, fraction=0.05):
    margin = min(width, height) * fraction
    return np.array(
        [
            [margin, margin],
            [width - 1 - margin, margin],
            [width - 1 - margin, height - 1 - margin],
            [margin, height - 1 - margin],
        ],
        dtype=np.float32,
    )


def _are_corners_degenerate(corners, width, height, min_fraction=0.02):
    xs = corners[:, 0]
    ys = corners[:, 1]
    bbox_area = (xs.max() - xs.min()) * (ys.max() - ys.min())
    return bbox_area < width * height * min_fraction


def _spread_collapsed_corners(corners, width, height, min_dist_fraction=0.05):
    diag = float(np.hypot(width, height))
    min_dist = diag * min_dist_fraction
    for i in range(4):
        for j in range(i + 1, 4):
            if np.linalg.norm(corners[i] - corners[j]) < min_dist:
                return _inset_image_corners(width, height, 0.05), True
    return corners, False


def _compute_homography(src, dst):
    import cv2

    matrix = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    return matrix


def _preprocess(img_bgr):
    import cv2

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    tensor = resized.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[None, ...]
    return tensor, width, height


def _decode_detections(logits, pred_boxes, width, height, threshold):
    stones = []
    corner_candidates = []

    for q in range(NUM_QUERIES):
        logit_base = q * NUM_CLASSES
        box_base = q * 4

        best_class = 0
        best_score = -1.0
        for c in range(NUM_CLASSES):
            score = _sigmoid(float(logits[logit_base + c]))
            if score > best_score:
                best_score = score
                best_class = c

        min_score = CORNER_MIN_THRESHOLD if best_class == CLASS_BOARD_CORNER else threshold
        if best_score < min_score:
            continue

        cx = float(pred_boxes[box_base]) * width
        cy = float(pred_boxes[box_base + 1]) * height
        det = {"cx": cx, "cy": cy, "class_id": best_class, "score": best_score}
        if best_class == CLASS_BOARD_CORNER:
            corner_candidates.append(det)
        else:
            stones.append(det)

    corner_candidates.sort(key=lambda item: item["score"], reverse=True)

    dedupe_min_dist = float(np.hypot(width, height)) * 0.05
    i = 0
    while i < len(corner_candidates):
        j = i + 1
        while j < len(corner_candidates):
            dist = np.hypot(
                corner_candidates[i]["cx"] - corner_candidates[j]["cx"],
                corner_candidates[i]["cy"] - corner_candidates[j]["cy"],
            )
            if dist < dedupe_min_dist:
                corner_candidates.pop(j)
            else:
                j += 1
        i += 1

    return stones, corner_candidates


def _infer_corners_from_candidates(corner_candidates, width, height):
    if len(corner_candidates) < 2:
        return _inset_image_corners(width, height, 0.05), False

    if len(corner_candidates) == 2:
        p1 = np.array([corner_candidates[0]["cx"], corner_candidates[0]["cy"]], dtype=np.float64)
        p2 = np.array([corner_candidates[1]["cx"], corner_candidates[1]["cy"]], dtype=np.float64)
        mx, my = (p1 + p2) / 2
        dx, dy = p2 - p1
        hdx, hdy = dx / 2, dy / 2
        candidates = [
            np.array([p1, [mx + hdy, my - hdx], p2, [mx - hdy, my + hdx]], dtype=np.float64),
            np.array([p1, p2, p2 + np.array([-dy, dx]), p1 + np.array([-dy, dx])], dtype=np.float64),
            np.array([p1, p2, p2 + np.array([dy, -dx]), p1 + np.array([dy, -dx])], dtype=np.float64),
        ]
        best_quad = candidates[0]
        best_score = -1e18
        for quad in candidates:
            inside = 0
            margin_sum = 0.0
            for pt in quad:
                mxv = min(pt[0], width - pt[0])
                myv = min(pt[1], height - pt[1])
                if mxv >= 0 and myv >= 0:
                    inside += 1
                margin_sum += mxv + myv
            score = inside * 1e6 + margin_sum
            if score > best_score:
                best_score = score
                best_quad = quad
        return _order_corners(best_quad).astype(np.float32), True

    if len(corner_candidates) == 3:
        pts = [np.array([d["cx"], d["cy"]], dtype=np.float64) for d in corner_candidates]
        best_quad = None
        best_score = float("inf")
        for diag in range(3):
            a = pts[diag]
            b = pts[(diag + 1) % 3]
            c = pts[(diag + 2) % 3]
            p4 = b + c - a
            quad = np.array([a, b, c, p4], dtype=np.float64)
            d1 = np.linalg.norm(a - p4)
            d2 = np.linalg.norm(b - c)
            score = abs(d1 - d2)
            if score < best_score:
                best_score = score
                best_quad = quad
        return _order_corners(best_quad).astype(np.float32), True

    top4 = corner_candidates[:4]
    points = [[d["cx"], d["cy"]] for d in top4]
    return _order_corners(points).astype(np.float32), True


def _map_stones_to_grid(stones, corners, board_size):
    import cv2

    dst = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    matrix = _compute_homography(corners, dst)

    black = []
    white = []
    confidences = []
    occupied = set()

    sorted_stones = sorted(stones, key=lambda item: item["score"], reverse=True)
    for det in sorted_stones:
        point = np.array([[[det["cx"], det["cy"]]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(point, matrix)[0, 0]
        col = int(np.clip(round(float(mapped[0]) * (board_size - 1)), 0, board_size - 1))
        row = int(np.clip(round(float(mapped[1]) * (board_size - 1)), 0, board_size - 1))
        key = (col, row)
        if key in occupied:
            continue
        occupied.add(key)
        confidences.append(det["score"])
        if det["class_id"] == CLASS_BLACK_STONE:
            black.append([col, row])
        else:
            white.append([col, row])

    return black, white, confidences


def suggest_corners_moku(img_bgr, threshold=DEFAULT_THRESHOLD):
    if not is_ready():
        return None
    session = _get_session()
    tensor, width, height = _preprocess(img_bgr)
    outputs = session.run(None, {"pixel_values": tensor})
    output_names = [item.name for item in session.get_outputs()]
    output_map = dict(zip(output_names, outputs))
    logits = output_map.get("logits", outputs[0]).reshape(-1)
    pred_boxes = output_map.get("pred_boxes", outputs[1]).reshape(-1)
    _stones, corner_candidates = _decode_detections(logits, pred_boxes, width, height, threshold)
    if len(corner_candidates) < 2:
        return None
    corners, _detected = _infer_corners_from_candidates(corner_candidates, width, height)
    return corners


def detect_board_moku(img_bgr, board_size=19, threshold=DEFAULT_THRESHOLD):
    session = _get_session()
    tensor, width, height = _preprocess(img_bgr)

    outputs = session.run(None, {"pixel_values": tensor})
    output_names = [item.name for item in session.get_outputs()]
    output_map = dict(zip(output_names, outputs))

    logits = output_map.get("logits", outputs[0]).reshape(-1)
    pred_boxes = output_map.get("pred_boxes", outputs[1]).reshape(-1)

    stones, corner_candidates = _decode_detections(logits, pred_boxes, width, height, threshold)
    corners, corners_detected = _infer_corners_from_candidates(corner_candidates, width, height)

    if _are_corners_degenerate(corners, width, height):
        corners = _inset_image_corners(width, height, 0.05)
        corners_detected = False

    corners, _collapsed = _spread_collapsed_corners(corners, width, height)

    black, white, confidences = _map_stones_to_grid(stones, corners, board_size)
    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.0

    return {
        "ok": True,
        "boardSize": board_size,
        "black": black,
        "white": white,
        "confidence": avg_conf,
        "detector": "moku",
        "stats": {
            "blackCount": len(black),
            "whiteCount": len(white),
            "cornerCount": min(len(corner_candidates), 4),
            "cornersDetected": corners_detected,
            "rawStoneCount": len(stones),
            "threshold": threshold,
            "model": "moku-v3",
        },
    }
