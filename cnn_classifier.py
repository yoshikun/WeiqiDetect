import json
import os

import numpy as np

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
MODEL_PATH = os.path.join(MODEL_DIR, "stone_cls.onnx")
META_PATH = os.path.join(MODEL_DIR, "stone_cls.meta.json")
PATCH_SIZE = 64
CLASS_EMPTY = 0
CLASS_BLACK = 1
CLASS_WHITE = 2

_session = None
_input_name = None
_output_name = None
_meta = None


def _load_meta():
    global _meta
    if _meta is not None:
        return _meta
    if os.path.exists(META_PATH):
        with open(META_PATH, "r", encoding="utf-8") as handle:
            _meta = json.load(handle)
    else:
        _meta = {"patchSize": PATCH_SIZE, "classes": ["empty", "black", "white"]}
    return _meta


def is_ready():
    return os.path.exists(MODEL_PATH)


def _get_session():
    global _session, _input_name, _output_name
    if _session is not None:
        return _session
    if not is_ready():
        raise FileNotFoundError(f"CNN model not found: {MODEL_PATH}")
    import onnxruntime as ort

    _session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    _input_name = _session.get_inputs()[0].name
    _output_name = _session.get_outputs()[0].name
    return _session


def preprocess_patch(patch_bgr):
    meta = _load_meta()
    patch_size = int(meta.get("patchSize") or PATCH_SIZE)
    if patch_bgr.shape[0] != patch_size or patch_bgr.shape[1] != patch_size:
        import cv2

        patch_bgr = cv2.resize(patch_bgr, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    tensor = patch_bgr.astype(np.float32) / 255.0
    tensor = tensor.transpose(2, 0, 1)[None, ...]
    return tensor


def predict_patches(patches_bgr, threshold=0.58, margin=0.14):
    if not patches_bgr:
        return []

    session = _get_session()
    batch = np.concatenate([preprocess_patch(patch) for patch in patches_bgr], axis=0)
    logits = session.run([_output_name], {_input_name: batch})[0]

    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)

    results = []
    for prob in probs:
        empty_p = float(prob[CLASS_EMPTY])
        black_p = float(prob[CLASS_BLACK])
        white_p = float(prob[CLASS_WHITE])
        ordered = sorted(
            [(CLASS_BLACK, black_p, "black"), (CLASS_WHITE, white_p, "white"), (CLASS_EMPTY, empty_p, None)],
            key=lambda item: item[1],
            reverse=True,
        )
        best_label, best_p, best_name = ordered[0]
        second_p = ordered[1][1]

        if best_name is None:
            results.append((None, empty_p, prob))
            continue

        if best_p < threshold or (best_p - second_p) < margin:
            results.append((None, best_p, prob))
            continue

        results.append((best_name, best_p, prob))

    return results


def model_info():
    meta = _load_meta()
    return {
        "ready": is_ready(),
        "path": MODEL_PATH,
        "patchSize": meta.get("patchSize", PATCH_SIZE),
        "valAccuracy": meta.get("valAccuracy"),
    }
