from board_geometry import (
    crop_square_patch,
    detect_grid_positions,
    find_board_quad_candidates,
    intersection_points,
    score_grid_alignment,
    warp_board,
)
from cnn_classifier import is_ready, model_info, predict_patches
from stone_fusion import (
    estimate_wood_level,
    filter_stone_list,
    fuse_prediction,
    heuristic_stone_at,
)


def _pick_best_warp(img, board_size):
    import cv2

    best = None
    best_score = -1.0
    for quad in find_board_quad_candidates(img):
        warped, _ = warp_board(img, quad, 1024)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        score, method = score_grid_alignment(gray, board_size)
        if score > best_score:
            best_score = score
            best = (warped, gray, method, score)
    if best is None:
        from board_geometry import find_board_quad

        quad = find_board_quad(img)
        warped, _ = warp_board(img, quad, 1024)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        score, method = score_grid_alignment(gray, board_size)
        best = (warped, gray, method, score)
    return best


def detect_board(img, board_size=19, threshold=0.58):
    warped, gray, pregrid_method, grid_score = _pick_best_warp(img, board_size)
    height, width = gray.shape[:2]

    xs, ys, grid_method = detect_grid_positions(gray, board_size)
    patch_size = max(48, int(min(height, width) / board_size * 1.2))
    radius = max(4, int(min(height, width) / (board_size * 2.6)))
    wood_level = estimate_wood_level(gray)
    points = intersection_points(board_size, width, height, xs, ys)

    black = []
    white = []
    confidences = []
    conf_map = {}

    if is_ready():
        patches = [crop_square_patch(warped, px, py, patch_size) for _, _, px, py in points]
        predictions = predict_patches(patches, threshold=threshold)
        for (gx, gy, px, py), pred in zip(points, predictions):
            cnn_stone, cnn_conf, _ = pred
            heur_stone, heur_conf = heuristic_stone_at(gray, px, py, radius, wood_level)
            stone, confidence = fuse_prediction(
                (cnn_stone, cnn_conf, None),
                (heur_stone, heur_conf),
            )
            if stone == "black":
                black.append([gx, gy])
                confidences.append(confidence)
                conf_map[(gx, gy)] = confidence
            elif stone == "white":
                white.append([gx, gy])
                confidences.append(confidence)
                conf_map[(gx, gy)] = confidence
        detector = "cnn+fusion"
        info = model_info()
        black = filter_stone_list(black, conf_map, board_size, min_conf=0.52)
        white = filter_stone_list(white, conf_map, board_size, min_conf=0.52)
        stats = {
            "blackCount": len(black),
            "whiteCount": len(white),
            "modelValAccuracy": info.get("valAccuracy"),
            "patchSize": info.get("patchSize"),
            "gridMethod": grid_method,
            "gridScore": round(grid_score, 3),
            "pregridMethod": pregrid_method,
        }
    else:
        for gx, gy, px, py in points:
            stone, confidence = heuristic_stone_at(gray, px, py, radius, wood_level)
            if stone == "black":
                black.append([gx, gy])
                confidences.append(confidence)
                conf_map[(gx, gy)] = confidence
            elif stone == "white":
                white.append([gx, gy])
                confidences.append(confidence)
                conf_map[(gx, gy)] = confidence
        black = filter_stone_list(black, conf_map, board_size, min_conf=0.55)
        white = filter_stone_list(white, conf_map, board_size, min_conf=0.55)
        detector = "heuristic"
        stats = {
            "blackCount": len(black),
            "whiteCount": len(white),
            "modelValAccuracy": None,
            "patchSize": patch_size,
            "gridMethod": grid_method,
            "gridScore": round(grid_score, 3),
            "pregridMethod": pregrid_method,
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
