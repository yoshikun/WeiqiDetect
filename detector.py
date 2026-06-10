from board_geometry import crop_square_patch, find_board_quad, intersection_points, warp_board
from cnn_classifier import is_ready, model_info, predict_patches


def detect_board(img, board_size=19, threshold=0.52):
    import cv2

    if not is_ready():
        raise RuntimeError("CNN model is not available")

    quad = find_board_quad(img)
    warped = warp_board(img, quad, 900)
    height, width = warped.shape[:2]
    patch_size = max(48, int(min(height, width) / board_size * 1.35))

    points = intersection_points(board_size, width, height)
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

    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.35
    info = model_info()

    return {
        "ok": True,
        "boardSize": board_size,
        "black": black,
        "white": white,
        "confidence": avg_conf,
        "detector": "cnn",
        "stats": {
            "blackCount": len(black),
            "whiteCount": len(white),
            "modelValAccuracy": info.get("valAccuracy"),
            "patchSize": info.get("patchSize"),
        },
    }
