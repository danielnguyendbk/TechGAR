"""
parking_cnn_utils.py – Wrapper cho CNN model phân loại ô đỗ xe
Dùng bởi: parking_dashboard.py, parking_slot_test.py, hybrid_detector.py
"""
import cv2
import numpy as np


class SlotCNNClassifier:
    """
    Load một model Keras (.h5 / .keras) đã train sẵn
    để phân loại ô đỗ xe: occupied (1) / empty (0).
    """

    def __init__(self, model_path="cnn_parking.h5", patch_size=64, decision_threshold=0.5):
        try:
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")  # tắt warning thừa
        except ImportError:
            raise RuntimeError(
                "Chưa cài TensorFlow. Cài bằng: pip install tensorflow"
            )

        self.model = tf.keras.models.load_model(model_path)
        self.patch_size = patch_size
        self.threshold = decision_threshold

        # Tự phát hiện input shape từ model
        input_shape = self.model.input_shape
        if input_shape and len(input_shape) >= 3:
            h, w = input_shape[1], input_shape[2]
            if h and w:
                self.patch_size = h  # dùng kích thước model yêu cầu
        print(f"[CNN] Model loaded: {model_path} | patch={self.patch_size}x{self.patch_size}")

    def predict_slot(self, roi_bgr):
        """
        Phân loại 1 ô đỗ xe từ ảnh BGR crop.

        Parameters:
            roi_bgr: numpy array (H, W, 3) – ảnh BGR crop của 1 ô xe

        Returns:
            occupied:   bool  – True nếu có xe
            confidence: float – độ tin cậy (0→1)
            p_occupied: float – xác suất có xe (output thô từ model)
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return False, 0.5, 0.5

        # Resize về kích thước model yêu cầu
        resized = cv2.resize(roi_bgr, (self.patch_size, self.patch_size))

        # Chuyển BGR → RGB (Keras convention)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Thêm batch dimension
        batch = np.expand_dims(normalized, axis=0)

        # Predict
        prediction = self.model.predict(batch, verbose=0)
        p_occ = float(prediction[0][0])

        # Phân loại
        occupied = p_occ > self.threshold
        confidence = p_occ if occupied else (1.0 - p_occ)

        return occupied, confidence, p_occ

    def predict_batch(self, roi_list):
        """
        Phân loại nhiều ô cùng lúc (nhanh hơn predict_slot từng cái).

        Parameters:
            roi_list: list of numpy arrays (BGR crops)

        Returns:
            list of (occupied, confidence, p_occupied)
        """
        if not roi_list:
            return []

        batch = []
        for roi in roi_list:
            if roi is None or roi.size == 0:
                # Padding cho ô lỗi
                batch.append(np.zeros((self.patch_size, self.patch_size, 3), dtype=np.float32))
            else:
                resized = cv2.resize(roi, (self.patch_size, self.patch_size))
                rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                batch.append(rgb.astype(np.float32) / 255.0)

        batch_array = np.array(batch)
        predictions = self.model.predict(batch_array, verbose=0)

        results = []
        for pred in predictions:
            p_occ = float(pred[0])
            occupied = p_occ > self.threshold
            confidence = p_occ if occupied else (1.0 - p_occ)
            results.append((occupied, confidence, p_occ))

        return results
