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


def predict_patches(patches_bgr, threshold=0.52):
    if not patches_bgr:
        return []

    session = _get_session()
    batch = np.concatenate([preprocess_patch(patch) for patch in patches_bgr], axis=0)
    logits = session.run([_output_name], {_input_name: batch})[0]

    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)

    results = []
    for prob in probs:
        label = int(prob.argmax())
        confidence = float(prob[label])
        if label == CLASS_BLACK and confidence >= threshold:
            results.append(("black", confidence))
        elif label == CLASS_WHITE and confidence >= threshold:
            results.append(("white", confidence))
        else:
            results.append((None, confidence if label == CLASS_EMPTY else confidence * 0.8))
    return results


def model_info():
    meta = _load_meta()
    return {
        "ready": is_ready(),
        "path": MODEL_PATH,
        "patchSize": meta.get("patchSize", PATCH_SIZE),
        "valAccuracy": meta.get("valAccuracy"),
    }
