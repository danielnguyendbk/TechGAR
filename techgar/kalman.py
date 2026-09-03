"""Lag-aware Kalman filter — PLAN 2 §2, implemented literally.

State (§2.1):      x = [X, Y, VX, VY, W, H, R]^T in R^7
Transition (§2.2): x_{t|t-1} = F(dt) x_{t-1}, dt from *timestamps*
Process noise (§2.3): Q(dt) grows with real elapsed time, so the association gate
                   widens at exactly the rate the frame rate drops
Measurement (§2.4): z = H x + v,  R_t = R_0 (1 + lam(1-c) + mu*o + nu*s)
VR1 (§2.2):        for dt > dt_threshold the prediction is walked in ~100 ms
                   substeps with velocity damping lambda_d per step

One addition the plan implies but does not name: damping deliberately biases the
prediction towards "the object stopped", so the bias itself is added to the
predicted covariance.  Without that term a genuinely fast vehicle observed after
a 500 ms stall (PLAN 3 scenario F) would fall outside its own gate and the
tracker would mint a new identity — the exact failure the plan forbids.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config_vision import KalmanConfig
from .linalg import mahalanobis_sq, psd_inverse, symmetrize

STATE_DIM = 7
MEASUREMENT = np.zeros((2, STATE_DIM))
MEASUREMENT[0, 0] = 1.0
MEASUREMENT[1, 1] = 1.0


def transition_matrix(dt: float) -> np.ndarray:
    """F_CV(dt) exactly as PLAN 2 §2.2."""
    f = np.eye(STATE_DIM)
    f[0, 2] = dt
    f[1, 3] = dt
    return f


def process_noise(dt: float, q: float, q_size: float = 0.0) -> np.ndarray:
    """Q(dt) of PLAN 2 §2.3 on the position/velocity block, random walk on size."""
    dt = float(dt)
    dt2, dt3, dt4 = dt * dt, dt ** 3, dt ** 4
    q_matrix = np.zeros((STATE_DIM, STATE_DIM))
    for pos, vel in ((0, 2), (1, 3)):
        q_matrix[pos, pos] = q * dt4 / 4.0
        q_matrix[pos, vel] = q * dt3 / 2.0
        q_matrix[vel, pos] = q * dt3 / 2.0
        q_matrix[vel, vel] = q * dt2
    for size in (4, 5, 6):
        q_matrix[size, size] = q_size * dt
    return q_matrix


@dataclass
class Prediction:
    state: np.ndarray
    covariance: np.ndarray
    dt: float
    substeps: int = 1
    damping_bias: float = 0.0


@dataclass
class LagAwareKalman:
    config: KalmanConfig
    state: np.ndarray
    covariance: np.ndarray
    timestamp: float
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(2))
    model: str = "CV"

    @classmethod
    def create(cls, config: KalmanConfig, position, timestamp: float, size=(1.0, 1.0),
               position_sigma: float = 1.0, velocity_sigma: float = 4.0,
               model: str = "CV") -> "LagAwareKalman":
        state = np.zeros(STATE_DIM)
        state[0:2] = np.asarray(position, dtype=float).reshape(2)
        state[4], state[5] = float(size[0]), float(size[1])
        state[6] = float(size[0]) / max(float(size[1]), 1e-6)
        covariance = np.diag([position_sigma ** 2, position_sigma ** 2,
                              velocity_sigma ** 2, velocity_sigma ** 2,
                              (0.2 * size[0]) ** 2 + 1.0, (0.2 * size[1]) ** 2 + 1.0, 0.25])
        return cls(config, state, covariance, float(timestamp), model=model)

    # --- accessors ----------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.state[0:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.state[2:4].copy()

    @property
    def size(self) -> np.ndarray:
        return self.state[4:6].copy()

    @property
    def aspect(self) -> float:
        return float(self.state[6])

    @property
    def position_covariance(self) -> np.ndarray:
        return symmetrize(self.covariance[0:2, 0:2])

    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    # --- prediction ---------------------------------------------------------
    def predict(self, timestamp: float) -> Prediction:
        cfg = self.config
        dt = float(np.clip(timestamp - self.timestamp, 0.0, cfg.max_dt))
        if not cfg.enable_prediction:
            # Ablation C: no velocity extrapolation and no dt-dependent inflation.
            frozen = self.state.copy()
            frozen[2:4] = 0.0
            return Prediction(frozen, symmetrize(self.covariance + process_noise(
                min(dt, 0.1), cfg.q, cfg.q_size)), dt, 1, 0.0)
        if dt <= 0.0:
            return Prediction(self.state.copy(), symmetrize(self.covariance), 0.0, 1, 0.0)
        undamped = transition_matrix(dt) @ self.state
        if self.model == "CA":
            undamped[0:2] += 0.5 * self.acceleration * dt * dt
            undamped[2:4] += self.acceleration * dt
        if dt <= cfg.dt_threshold:
            state = undamped
            covariance = transition_matrix(dt) @ self.covariance @ transition_matrix(dt).T
            covariance = covariance + process_noise(dt, cfg.q, cfg.q_size)
            return Prediction(state, symmetrize(covariance), dt, 1, 0.0)
        substeps = int(np.ceil(dt / max(cfg.substep, 1e-3)))
        step = dt / substeps
        state = self.state.copy()
        covariance = self.covariance.copy()
        f_step = transition_matrix(step)
        q_step = process_noise(step, cfg.q, cfg.q_size)
        for _ in range(substeps):
            state = f_step @ state
            if self.model == "CA":
                state[0:2] += 0.5 * self.acceleration * step * step
                state[2:4] += self.acceleration * step
            state[2:4] *= cfg.velocity_damping        # lambda_d, PLAN 2 §2.2
            covariance = f_step @ covariance @ f_step.T + q_step
        bias = float(np.linalg.norm(undamped[0:2] - state[0:2]))
        if bias > 0.0:
            covariance[0:2, 0:2] += (bias ** 2) * np.eye(2)
            covariance[2:4, 2:4] += ((bias / max(dt, 1e-3)) ** 2) * np.eye(2)
        return Prediction(state, symmetrize(covariance), dt, substeps, bias)

    def apply(self, prediction: Prediction, timestamp: float) -> None:
        self.state = prediction.state
        self.covariance = prediction.covariance
        self.timestamp = float(timestamp)

    def advance(self, timestamp: float) -> Prediction:
        prediction = self.predict(timestamp)
        self.apply(prediction, timestamp)
        return prediction

    # --- measurement --------------------------------------------------------
    def measurement_covariance(self, confidence: float, occlusion: float = 0.0,
                               seam: float = 0.0) -> np.ndarray:
        """R_t of PLAN 2 §2.4."""
        cfg = self.config
        scale = (1.0 + cfg.lam * (1.0 - float(np.clip(confidence, 0.0, 1.0)))
                 + cfg.mu * float(occlusion) + cfg.nu * float(seam))
        return cfg.r0 * scale * np.eye(2)

    def innovation(self, measurement, measurement_covariance) -> tuple[np.ndarray, np.ndarray]:
        residual = np.asarray(measurement, dtype=float).reshape(2) - MEASUREMENT @ self.state
        s = MEASUREMENT @ self.covariance @ MEASUREMENT.T + symmetrize(measurement_covariance)
        return residual, symmetrize(s)

    def gate_distance(self, measurement, measurement_covariance) -> float:
        residual, s = self.innovation(measurement, measurement_covariance)
        return mahalanobis_sq(residual, s)

    def update(self, measurement, measurement_covariance, timestamp: float | None = None,
               size=None, size_gain: float = 0.3) -> np.ndarray:
        residual, s = self.innovation(measurement, measurement_covariance)
        gain = self.covariance @ MEASUREMENT.T @ psd_inverse(s)
        previous_velocity = self.velocity
        self.state = self.state + gain @ residual
        identity = np.eye(STATE_DIM)
        self.covariance = symmetrize((identity - gain @ MEASUREMENT) @ self.covariance)
        if size is not None:
            width, height = float(size[0]), float(size[1])
            self.state[4] += size_gain * (width - self.state[4])
            self.state[5] += size_gain * (height - self.state[5])
            self.state[6] = self.state[4] / max(self.state[5], 1e-6)
        if timestamp is not None:
            dt = float(timestamp) - self.timestamp
            if dt > 1e-6:
                self.acceleration = 0.5 * self.acceleration + 0.5 * (self.velocity
                                                                     - previous_velocity) / dt
            self.timestamp = float(timestamp)
        return residual
