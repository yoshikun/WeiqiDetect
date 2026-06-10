from board_geometry import crop_square_patch, find_board_quad, intersection_points, warp_board
from cnn_classifier import is_ready, model_info, predict_patches


def detect_board_heuristic(warped, board_size):
    import cv2
    import numpy as np

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    height, width = gray.shape[:2]
    strip = max(6, int(min(height, width) * 0.02))
    wood = float(np.median(np.concatenate([
        gray[:strip, :].reshape(-1),
        gray[-strip:, :].reshape(-1),
        gray[:, :strip].reshape(-1),
        gray[:, -strip:].reshape(-1),
    ])))
    radius = max(4, int(min(height, width) / (board_size * 2.4)))

    black = []
    white = []
    confidences = []

    for gy in range(board_size):
        for gx in range(board_size):
            px = int((gx + 0.5) * width / board_size)
            py = int((gy + 0.5) * height / board_size)
            x0 = max(0, px - radius)
            y0 = max(0, py - radius)
            x1 = min(width, px + radius + 1)
            y1 = min(height, py + radius + 1)
            patch = gray[y0:y1, x0:x1]
            if patch.size < 9:
                continue
            median = float(np.median(patch))
            dark_cut = wood - max(18, 0.22 * (255 - wood))
            light_cut = wood + max(18, 0.22 * wood)
            dark_frac = float(np.mean(patch < dark_cut))
            light_frac = float(np.mean(patch > light_cut))
            if dark_frac > 0.22 and median < wood - 12:
                black.append([gx, gy])
                confidences.append(min(1.0, dark_frac))
            elif light_frac > 0.22 and median > wood + 12:
                white.append([gx, gy])
                confidences.append(min(1.0, light_frac))

    return black, white, confidences, "heuristic"


def detect_board(img, board_size=19, threshold=0.52):
    quad = find_board_quad(img)
    warped = warp_board(img, quad, 900)
    height, width = warped.shape[:2]
    patch_size = max(48, int(min(height, width) / board_size * 1.35))
    points = intersection_points(board_size, width, height)

    if is_ready():
        patches = [crop_square_patch(warped, px, py, patch_size) for _, _, px, py in points]
        predictions = predict_patches(patches, threshold=threshold)
        black = []
        white = []
        confidences = []
        for (gx, gy, _, _), (stone, confidence) in zip(points, predictions):
            if stone == "black":
                black.append([gx, gy])
                confidences.append(confidence)
            elif stone == "white":
                white.append([gx, gy])
                confidences.append(confidence)
        detector = "cnn"
        info = model_info()
        stats = {
            "blackCount": len(black),
            "whiteCount": len(white),
            "modelValAccuracy": info.get("valAccuracy"),
            "patchSize": info.get("patchSize"),
        }
    else:
        black, white, confidences, detector = detect_board_heuristic(warped, board_size)
        stats = {
            "blackCount": len(black),
            "whiteCount": len(white),
            "modelValAccuracy": None,
            "patchSize": patch_size,
        }

    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.35

    return {
        "ok": True,
        "boardSize": board_size,
        "black": black,
        "white": white,
        "confidence": avg_conf,
        "detector": detector,
        "stats": stats,
    }
