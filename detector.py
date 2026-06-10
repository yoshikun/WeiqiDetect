from board_geometry import (
    detect_grid_positions,
    find_board_quad_candidates,
    score_grid_alignment,
    warp_board,
)
from circle_detect import detect_stones_by_circles, uniform_grid
from stone_fusion import estimate_wood_level


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
    del threshold  # circle pipeline uses internal thresholds

    warped, gray, pregrid_method, grid_score = _pick_best_warp(img, board_size)
    height, width = gray.shape[:2]
    wood_level = estimate_wood_level(gray)

    xs, ys, grid_method = detect_grid_positions(gray, board_size)
    if grid_score < 0.35:
        xs, ys = uniform_grid(board_size, width, height)
        grid_method = "uniform-fallback"

    black, white, confidences, circle_count = detect_stones_by_circles(
        warped, gray, board_size, wood_level, xs, ys
    )

    avg_conf = round(sum(confidences) / len(confidences), 3) if confidences else 0.35

    return {
        "ok": True,
        "boardSize": board_size,
        "black": black,
        "white": white,
        "confidence": avg_conf,
        "detector": "circles",
        "stats": {
            "blackCount": len(black),
            "whiteCount": len(white),
            "gridMethod": grid_method,
            "gridScore": round(grid_score, 3),
            "pregridMethod": pregrid_method,
            "circleCount": circle_count,
            "woodLevel": round(wood_level, 1),
        },
    }
