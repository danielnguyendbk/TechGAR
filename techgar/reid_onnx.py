"""Phase C — ONNX Neural ReID with automatic fallback to color-spatial descriptor.

Per TechGAR.md Phase C:
- If an ONNX ReID model (MobileNetV3 / OSNet) is available and loads cleanly,
  neural embeddings (128-dim or 256-dim, L2-normalized) are computed.
- If the model is missing, corrupt, or onnxruntime is unavailable, the extractor
  automatically falls back to the deterministic 27-dimensional color-spatial descriptor,
  logging a warning and exposing reid_status: "fallback_color".
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .appearance import embed as color_embed, cosine_distance

logger = logging.getLogger("techgar.reid")


class ONNXReIDExtractor:
    """Lightweight ReID extractor supporting neural ONNX models with graceful fallback."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.session: Any = None
        self.status = "fallback_color"
        self.embedding_dim = 27
        self.input_name = ""
        self.input_shape: tuple[int, ...] = (1, 3, 128, 128)
        self._init_session()

    def _init_session(self) -> None:
        if self.model_path is None or not self.model_path.is_file():
            self.status = "fallback_color"
            return

        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            self.session = ort.InferenceSession(
                str(self.model_path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            inputs = self.session.get_inputs()
            self.input_name = inputs[0].name
            shape = inputs[0].shape
            # Default input size (NCHW)
            h = shape[2] if len(shape) > 2 and isinstance(shape[2], int) and shape[2] > 0 else 128
            w = shape[3] if len(shape) > 3 and isinstance(shape[3], int) and shape[3] > 0 else 128
            self.input_shape = (1, 3, h, w)

            outputs = self.session.get_outputs()
            out_shape = outputs[0].shape
            self.embedding_dim = out_shape[-1] if len(out_shape) > 1 and isinstance(out_shape[-1], int) else 128

            self.status = "neural"
            logger.info("Initialized Neural ONNX ReID from %s (dim=%d)", self.model_path.name, self.embedding_dim)
        except Exception as err:
            self.session = None
            self.status = "fallback_color"
            self.embedding_dim = 27
            logger.warning("Failed to initialize ONNX ReID model %s (%s); falling back to color-spatial", self.model_path, err)

    def extract(self, image: np.ndarray, bbox, mask: np.ndarray | None = None) -> np.ndarray:
        """Extract an L2-normalized feature embedding from the image bounding box."""
        if self.status != "neural" or self.session is None:
            return color_embed(image, bbox, mask)

        try:
            # Crop bbox
            h_img, w_img = image.shape[:2]
            x0, y0, x1, y1 = (float(v) for v in bbox)
            x0 = int(np.clip(np.floor(x0), 0, w_img - 1))
            x1 = int(np.clip(np.ceil(x1), x0 + 1, w_img))
            y0 = int(np.clip(np.floor(y0), 0, h_img - 1))
            y1 = int(np.clip(np.ceil(y1), y0 + 1, h_img))
            crop = image[y0:y1, x0:x1]

            if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                return color_embed(image, bbox, mask)

            # Resize to input shape
            target_h, target_w = self.input_shape[2], self.input_shape[3]
            try:
                import cv2
                resized = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            except ImportError:
                # Fallback nearest neighbor resize
                y_idx = (np.linspace(0, crop.shape[0] - 1, target_h)).astype(int)
                x_idx = (np.linspace(0, crop.shape[1] - 1, target_w)).astype(int)
                resized = crop[y_idx[:, None], x_idx]

            # Normalize and convert HWC -> NCHW
            rgb = resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            if rgb.ndim == 2:
                rgb = np.stack([rgb, rgb, rgb], axis=-1)
            elif rgb.shape[2] == 1:
                rgb = np.repeat(rgb, 3, axis=2)
            normalized = (rgb[:, :, :3] - mean) / std
            chw = np.transpose(normalized, (2, 0, 1))
            tensor = np.expand_dims(chw, axis=0).astype(np.float32)

            outputs = self.session.run(None, {self.input_name: tensor})
            feat = np.asarray(outputs[0], dtype=np.float32).reshape(-1)
            norm = float(np.linalg.norm(feat))
            return feat / norm if norm > 1e-6 else feat
        except Exception as err:
            logger.warning("Neural ReID inference error (%s); falling back to color-spatial", err)
            return color_embed(image, bbox, mask)


_DEFAULT_EXTRACTOR: ONNXReIDExtractor | None = None


def get_default_reid_extractor(model_path: str | Path | None = None) -> ONNXReIDExtractor:
    """Get or create singleton ReID extractor instance."""
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None or model_path is not None:
        _DEFAULT_EXTRACTOR = ONNXReIDExtractor(model_path)
    return _DEFAULT_EXTRACTOR
