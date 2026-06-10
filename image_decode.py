import base64

import cv2
import numpy as np
import requests
from PIL import Image, ImageOps


def _bytes_to_bgr(raw):
    image = Image.open(__import__("io").BytesIO(raw))
    image = ImageOps.exif_transpose(image)
    if image.mode != "RGB":
        image = image.convert("RGB")
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def decode_image(payload):
    image_url = payload.get("imageUrl")
    if image_url:
        try:
            resp = requests.get(image_url, timeout=30)
            resp.raise_for_status()
            return _bytes_to_bgr(resp.content)
        except Exception:
            return None

    image_b64 = payload.get("image")
    if not image_b64:
        return None

    try:
        return _bytes_to_bgr(base64.b64decode(image_b64))
    except Exception:
        return None
