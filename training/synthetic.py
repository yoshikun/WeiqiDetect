import random

import cv2
import numpy as np

PATCH_SIZE = 64
CLASS_EMPTY = 0
CLASS_BLACK = 1
CLASS_WHITE = 2


def _rand_wood_color():
    base = random.randint(95, 185)
    drift = random.randint(-18, 18)
    b = max(40, min(220, base + random.randint(-10, 10)))
    g = max(40, min(230, base + drift))
    r = max(40, min(230, base + random.randint(-8, 16)))
    return int(b), int(g), int(r)


def _draw_grid(img, wood, cell):
    line = tuple(max(0, c - random.randint(35, 75)) for c in wood)
    thickness = random.choice([1, 1, 2])
    center = PATCH_SIZE // 2
    cv2.line(img, (0, center), (PATCH_SIZE, center), line, thickness, cv2.LINE_AA)
    cv2.line(img, (center, 0), (center, PATCH_SIZE), line, thickness, cv2.LINE_AA)
    if random.random() < 0.45:
        offset = random.randint(-cell // 2, cell // 2)
        y = max(0, min(PATCH_SIZE - 1, center + offset))
        cv2.line(img, (0, y), (PATCH_SIZE, y), line, thickness, cv2.LINE_AA)
    if random.random() < 0.45:
        offset = random.randint(-cell // 2, cell // 2)
        x = max(0, min(PATCH_SIZE - 1, center + offset))
        cv2.line(img, (x, 0), (x, PATCH_SIZE), line, thickness, cv2.LINE_AA)


def _draw_stone(img, label, cell):
    center = (PATCH_SIZE // 2, PATCH_SIZE // 2)
    radius = max(8, int(cell * random.uniform(0.34, 0.48)))
    if label == CLASS_BLACK:
        cv2.circle(img, center, radius, (25, 25, 25), -1, cv2.LINE_AA)
        cv2.circle(img, center, radius, (8, 8, 8), 1, cv2.LINE_AA)
        highlight = (center[0] - radius // 3, center[1] - radius // 3)
        cv2.circle(img, highlight, max(2, radius // 5), (70, 70, 70), -1, cv2.LINE_AA)
    elif label == CLASS_WHITE:
        cv2.circle(img, center, radius, (245, 245, 245), -1, cv2.LINE_AA)
        cv2.circle(img, center, radius, (180, 180, 180), 1, cv2.LINE_AA)
        highlight = (center[0] - radius // 3, center[1] - radius // 3)
        cv2.circle(img, highlight, max(2, radius // 4), (255, 255, 255), -1, cv2.LINE_AA)


def _augment(img):
    import cv2
    import numpy as np

    if random.random() < 0.65:
        h, w = img.shape[:2]
        src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
        jitter = random.randint(4, 12)
        dst = np.float32(
            [
                [random.randint(0, jitter), random.randint(0, jitter)],
                [w - 1 - random.randint(0, jitter), random.randint(0, jitter)],
                [w - 1 - random.randint(0, jitter), h - 1 - random.randint(0, jitter)],
                [random.randint(0, jitter), h - 1 - random.randint(0, jitter)],
            ]
        )
        matrix = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)

    if random.random() < 0.45:
        offset_x = random.randint(-6, 6)
        offset_y = random.randint(-6, 6)
        matrix = np.float32([[1, 0, offset_x], [0, 1, offset_y]])
        img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT_101)

    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.2:
        img = cv2.flip(img, 0)

    alpha = random.uniform(0.8, 1.2)
    beta = random.randint(-22, 22)
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    if random.random() < 0.4:
        noise = np.random.normal(0, random.uniform(2, 10), img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), random.uniform(0.2, 1.2))

    if random.random() < 0.25:
        quality = random.randint(30, 85)
        ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if ok:
            img = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    if random.random() < 0.2:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-8, 8)) % 180
        hsv[:, :, 1] *= random.uniform(0.8, 1.2)
        hsv[:, :, 2] *= random.uniform(0.85, 1.15)
        img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    return img


def render_patch(label, patch_size=PATCH_SIZE):
    wood = _rand_wood_color()
    cell = patch_size // 2 + random.randint(-4, 4)
    img = np.full((patch_size, patch_size, 3), wood, dtype=np.uint8)
    _draw_grid(img, wood, cell)
    if label in (CLASS_BLACK, CLASS_WHITE):
        _draw_stone(img, label, cell)
    return _augment(img)


def generate_batch(count_per_class=4000, patch_size=PATCH_SIZE):
    images = []
    labels = []
    for label in (CLASS_EMPTY, CLASS_BLACK, CLASS_WHITE):
        for _ in range(count_per_class):
            images.append(render_patch(label, patch_size))
            labels.append(label)
    indices = list(range(len(images)))
    random.shuffle(indices)
    images = [images[i] for i in indices]
    labels = [labels[i] for i in indices]
    return np.stack(images), np.array(labels, dtype=np.int64)
