import numpy as np


def _sample_disc(gray, cx, cy, radius, width, height):
    x0 = max(0, int(np.ceil(cx - radius)))
    x1 = min(width - 1, int(np.floor(cx + radius)))
    y0 = max(0, int(np.ceil(cy - radius)))
    y1 = min(height - 1, int(np.floor(cy + radius)))
    r2 = radius * radius
    values = []
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                values.append(float(gray[y, x]))
    return float(np.mean(values)) if values else 0.0


def _sample_variance(gray, cx, cy, radius, width, height):
    x0 = max(0, int(np.ceil(cx - radius)))
    x1 = min(width - 1, int(np.floor(cx + radius)))
    y0 = max(0, int(np.ceil(cy - radius)))
    y1 = min(height - 1, int(np.floor(cy + radius)))
    r2 = radius * radius
    values = []
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                values.append(float(gray[y, x]))
    if len(values) < 2:
        return 0.0
    arr = np.array(values, dtype=np.float32)
    return float(np.std(arr))


def _kmeans3(values):
    if len(values) < 3:
        return 0.0, 0.0, 0.0
    sorted_vals = sorted(values)
    c0 = sorted_vals[int(len(sorted_vals) * 0.1)]
    c1 = sorted_vals[int(len(sorted_vals) * 0.5)]
    c2 = sorted_vals[int(len(sorted_vals) * 0.9)]
    for _ in range(20):
        s0 = s1 = s2 = 0.0
        n0 = n1 = n2 = 0
        for value in values:
            d0 = abs(value - c0)
            d1 = abs(value - c1)
            d2 = abs(value - c2)
            if d0 <= d1 and d0 <= d2:
                s0 += value
                n0 += 1
            elif d1 <= d2:
                s1 += value
                n1 += 1
            else:
                s2 += value
                n2 += 1
        new_c0 = s0 / n0 if n0 else c0
        new_c1 = s1 / n1 if n1 else c1
        new_c2 = s2 / n2 if n2 else c2
        if abs(new_c0 - c0) + abs(new_c1 - c1) + abs(new_c2 - c2) < 0.5:
            break
        c0, c1, c2 = new_c0, new_c1, new_c2
    return tuple(sorted((c0, c1, c2)))


def _grid_point(col, row, board_size, grid_corners, cell_size):
    if grid_corners is not None:
        u = col / max(board_size - 1, 1)
        v = row / max(board_size - 1, 1)
        tl, tr, br, bl = grid_corners
        cx = (1 - u) * (1 - v) * tl[0] + u * (1 - v) * tr[0] + u * v * br[0] + (1 - u) * v * bl[0]
        cy = (1 - u) * (1 - v) * tl[1] + u * (1 - v) * tr[1] + u * v * br[1] + (1 - u) * v * bl[1]
        return cx, cy
    return col * cell_size, row * cell_size


def classify_intersections(gray, board_size, grid_corners=None):
    height, width = gray.shape[:2]
    if grid_corners is not None:
        tl, tr, _, bl = grid_corners
        grid_w = float(np.hypot(tr[0] - tl[0], tr[1] - tl[1]))
        grid_h = float(np.hypot(bl[0] - tl[0], bl[1] - tl[1]))
        cell_size = (grid_w + grid_h) / (2 * max(board_size - 1, 1))
    else:
        cell_size = (width - 1) / max(board_size - 1, 1)

    disc_radius = cell_size * 0.35
    var_radius = cell_size * 0.35
    total = board_size * board_size
    brightness = np.zeros(total, dtype=np.float32)
    variances = np.zeros(total, dtype=np.float32)

    for row in range(board_size):
        for col in range(board_size):
            cx, cy = _grid_point(col, row, board_size, grid_corners, cell_size)
            idx = row * board_size + col
            brightness[idx] = _sample_disc(gray, cx, cy, disc_radius, width, height)
            variances[idx] = _sample_variance(gray, cx, cy, var_radius, width, height)

    relative = np.zeros(total, dtype=np.float32)
    ring = 3
    for row in range(board_size):
        for col in range(board_size):
            neighbors = []
            for dr in range(-ring, ring + 1):
                for dc in range(-ring, ring + 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < board_size and 0 <= nc < board_size:
                        neighbors.append(brightness[nr * board_size + nc])
            neighbors.sort()
            local_median = neighbors[len(neighbors) // 2] if neighbors else brightness[row * board_size + col]
            relative[row * board_size + col] = brightness[row * board_size + col] - local_median

    black_c, board_c, white_c = _kmeans3(relative.tolist())
    black_boundary = (black_c + board_c) / 2
    white_boundary = (board_c + white_c) / 2
    total_spread = white_c - black_c
    has_black = total_spread > 5 and board_c - black_c > total_spread * 0.15
    has_white = total_spread > 5 and white_c - board_c > total_spread * 0.15
    median_var = float(np.median(variances))

    black = []
    white = []
    confidences = []
    for row in range(board_size):
        for col in range(board_size):
            idx = row * board_size + col
            rel = relative[idx]
            high_var = variances[idx] > median_var * 3
            is_edge = row in (0, board_size - 1) or col in (0, board_size - 1)
            margin = total_spread * 0.1 if is_edge else 0.0
            if has_black and rel < black_boundary - margin and (not high_var or rel < black_c * 0.5):
                black.append([col, row])
                confidences.append(min(1.0, abs(rel - black_boundary) / max(total_spread, 1.0) + 0.4))
            elif has_white and rel > white_boundary + margin and (not high_var or rel > white_c * 0.5):
                white.append([col, row])
                confidences.append(min(1.0, abs(rel - white_boundary) / max(total_spread, 1.0) + 0.4))

    return black, white, confidences
