
def estimate_wood_level(gray):
    import numpy as np

    height, width = gray.shape[:2]
    strip = max(6, int(min(height, width) * 0.02))
    samples = np.concatenate(
        [
            gray[:strip, :].reshape(-1),
            gray[-strip:, :].reshape(-1),
            gray[:, :strip].reshape(-1),
            gray[:, -strip:].reshape(-1),
        ]
    )
    return float(np.median(samples))


def heuristic_stone_at(gray, px, py, radius, wood_level):
    import numpy as np

    height, width = gray.shape[:2]
    x0 = max(0, px - radius)
    y0 = max(0, py - radius)
    x1 = min(width, px + radius + 1)
    y1 = min(height, py + radius + 1)
    patch = gray[y0:y1, x0:x1]
    if patch.size < 9:
        return None, 0.0

    yy, xx = np.ogrid[-(py - y0) : y1 - py, -(px - x0) : x1 - px]
    mask = xx * xx + yy * yy <= radius * radius
    if mask.shape != patch.shape:
        values = patch.reshape(-1)
    else:
        values = patch[mask]
    if values.size < 9:
        values = patch.reshape(-1)

    median = float(np.median(values))
    dark_cut = wood_level - max(20, 0.24 * (255 - wood_level))
    light_cut = wood_level + max(20, 0.24 * wood_level)
    dark_frac = float(np.mean(values < dark_cut))
    light_frac = float(np.mean(values > light_cut))

    if dark_frac > 0.28 and median < wood_level - 14:
        return "black", min(1.0, dark_frac + (wood_level - median) / 100.0)
    if light_frac > 0.28 and median > wood_level + 14:
        return "white", min(1.0, light_frac + (median - wood_level) / 100.0)
    return None, max(dark_frac, light_frac)


def fuse_prediction(cnn_result, heur_result):
    cnn_stone, cnn_conf, probs = cnn_result
    heur_stone, heur_conf = heur_result

    if cnn_stone and heur_stone:
        if cnn_stone == heur_stone:
            return cnn_stone, min(1.0, (cnn_conf + heur_conf) * 0.55)
        return None, max(cnn_conf, heur_conf) * 0.5

    if cnn_stone:
        if cnn_conf >= 0.72:
            return cnn_stone, cnn_conf
        return None, cnn_conf

    if heur_stone and heur_conf >= 0.62:
        return heur_stone, heur_conf * 0.9

    return None, max(cnn_conf, heur_conf)


def remove_isolated_points(stones, board_size, radius=2, min_neighbors=1):
    if len(stones) <= 1:
        return stones

    kept = []
    stone_set = {tuple(point) for point in stones}
    for point in stones:
        x, y = point
        neighbors = 0
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                if dx == 0 and dy == 0:
                    continue
                if (x + dx, y + dy) in stone_set:
                    neighbors += 1
        if neighbors >= min_neighbors:
            kept.append(point)
    return kept


def filter_stone_list(stones, conf_map, board_size, min_conf=0.55):
    filtered = []
    stone_set = set()
    for point in stones:
        key = tuple(point)
        conf = conf_map.get(key, 0.0)
        if conf >= min_conf:
            filtered.append(point)
            stone_set.add(key)

    kept = []
    for point in filtered:
        key = tuple(point)
        conf = conf_map.get(key, 0.0)
        if conf >= 0.68:
            kept.append(point)
            continue
        neighbors = 0
        x, y = point
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                if dx == 0 and dy == 0:
                    continue
                if (x + dx, y + dy) in stone_set:
                    neighbors += 1
        if neighbors >= 1:
            kept.append(point)
    return kept
