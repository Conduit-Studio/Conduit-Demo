"""Conduit Run Code step: REAL YOLO-pose keypoint inference on one image, ONNX runtime.

Runs an ONNX-exported YOLO-pose model on the image PIXELS with `onnxruntime` (no torch,
no ultralytics at runtime — only onnxruntime + numpy + Pillow, ~105 MB, fits a Lambda
container image). Returns the annotated overlay PNG and a predicted-labels file.

The model `yolo11n-pose.onnx` is bundled next to this file (exported once via
`from ultralytics import YOLO; YOLO('yolo11n-pose.pt').export(format='onnx', imgsz=640)`),
or point `YOLO_POSE_ONNX` at another path.

Input port:
    image: s3-ref — the image (Conduit downloads it; the code gets a local path)
Output ports:
    overlay: s3-ref — annotated PNG with the detected pose (Conduit uploads it)
    labels:  s3-ref — predicted keypoints as a YOLO-pose .txt (saved back to S3)
    summary: json    — person count + per-detection confidence
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

ONNX_PATH = os.environ.get("YOLO_POSE_ONNX", str(Path(__file__).resolve().parent / "yolo11n-pose.onnx"))
IMGSZ = 640
CONF_THRES = 0.25
IOU_THRES = 0.45

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
SKELETON = [
    ("left_ankle", "left_knee"), ("left_knee", "left_hip"),
    ("right_ankle", "right_knee"), ("right_knee", "right_hip"),
    ("left_hip", "right_hip"), ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"), ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"), ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"), ("right_elbow", "right_wrist"),
    ("nose", "left_eye"), ("nose", "right_eye"),
]

_session: ort.InferenceSession | None = None


def _sess() -> ort.InferenceSession:
    global _session
    if _session is None:
        _session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    return _session


def _letterbox(image: Image.Image, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize preserving aspect onto a size x size grey canvas; return CHW float input + the transform."""
    w0, h0 = image.size
    scale = min(size / w0, size / h0)
    nw, nh = round(w0 * scale), round(h0 * scale)
    resized = image.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (114, 114, 114))
    pad_x, pad_y = (size - nw) // 2, (size - nh) // 2
    canvas.paste(resized, (pad_x, pad_y))
    arr = np.asarray(canvas, dtype=np.float32) / 255.0          # HWC, 0..1
    return arr.transpose(2, 0, 1)[None], scale, pad_x, pad_y     # 1,C,H,W


def _iou(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], others[:, 0]); y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2]); y2 = np.minimum(box[3], others[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_others = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return inter / (area_box + area_others - inter + 1e-9)


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0]); keep.append(i)
        if order.size == 1:
            break
        order = order[1:][_iou(boxes[i], boxes[order[1:]]) < iou_thres]
    return keep


def main(inputs: dict[str, Any]) -> dict[str, Any]:
    image = Image.open(inputs["image"]).convert("RGB")
    width, height = image.size
    blob, scale, pad_x, pad_y = _letterbox(image, IMGSZ)

    session = _sess()
    out = session.run(None, {session.get_inputs()[0].name: blob})[0]   # [1, 56, 8400]
    pred = out[0].transpose(1, 0)                                       # [8400, 56]

    conf = pred[:, 4]
    pred, conf = pred[conf > CONF_THRES], conf[conf > CONF_THRES]
    persons: list[dict[str, Any]] = []
    if len(pred):
        cx, cy, bw, bh = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)
        keep = _nms(boxes, conf, IOU_THRES)
        boxes, conf, kpts = boxes[keep], conf[keep], pred[keep, 5:].reshape(-1, 17, 3)

        # undo letterbox → original-image pixel coords
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
        kpts[..., 0] = (kpts[..., 0] - pad_x) / scale
        kpts[..., 1] = (kpts[..., 1] - pad_y) / scale

        draw = ImageDraw.Draw(image)
        for b, c, kp in zip(boxes, conf, kpts):
            draw.rectangle([b[0], b[1], b[2], b[3]], outline=(0, 170, 255), width=2)
            draw.text((b[0], max(0, b[1] - 10)), f"person {c:.2f}", fill=(0, 170, 255))
            pts = {KEYPOINT_NAMES[j]: (float(kp[j, 0]), float(kp[j, 1])) for j in range(17) if kp[j, 2] > 0.5}
            for a, z in SKELETON:
                if a in pts and z in pts:
                    draw.line([pts[a], pts[z]], fill=(86, 200, 255), width=2)
            for (px, py) in pts.values():
                draw.ellipse([px - 3, py - 3, px + 3, py + 3], fill=(255, 86, 47), outline=(255, 255, 255))
            persons.append({"confidence": float(c), "keypoints": kp.astype(float).tolist()})

    overlay_path = str(Path(tempfile.gettempdir()) / "overlay.png")
    image.save(overlay_path)

    labels_path = str(Path(tempfile.gettempdir()) / "labels.txt")
    Path(labels_path).write_text(_yolo_pose_label(persons, width, height))

    return {
        "overlay": overlay_path,
        "labels": labels_path,
        "summary": {"persons": len(persons), "confidence": [round(p["confidence"], 3) for p in persons]},
    }


def _yolo_pose_label(persons: list[dict[str, Any]], width: int, height: int) -> str:
    """Predicted detections → YOLO-pose label text, coords normalized to the image."""
    lines: list[str] = []
    for p in persons:
        kp = np.asarray(p["keypoints"])                      # [17, 3] pixel x,y,conf
        xs, ys = kp[:, 0], kp[:, 1]
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        cx, cy = ((x1 + x2) / 2) / width, ((y1 + y2) / 2) / height
        bw, bh = (x2 - x1) / width, (y2 - y1) / height
        parts = ["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}"]
        for j in range(17):
            v = 2 if kp[j, 2] > 0.5 else (1 if kp[j, 2] > 0.0 else 0)
            parts += [f"{kp[j, 0] / width:.6f}", f"{kp[j, 1] / height:.6f}", str(v)]
        lines.append(" ".join(parts))
    return "\n".join(lines) + ("\n" if lines else "")
