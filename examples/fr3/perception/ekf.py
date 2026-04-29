import numpy as np
from typing import NamedTuple, Optional
from camera import WorldFrame, AprilDetector, multi_view_tag_pnp, get_rgbd_frames, TEE_TAG_ID, SERIAL_NUMS
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from geom_tee import THICKNESS


class PoseDetection(NamedTuple):
    name: str
    pose: np.ndarray   # (3,) [x, y, yaw]
    R: np.ndarray      # (3, 3)
    gated: bool = True

class EKF(NamedTuple):
    frame:   WorldFrame
    mean:    np.ndarray   # (3,)  [x, y, theta] in world frame
    cov:     np.ndarray   # (3, 3)
    Q:       np.ndarray = np.diag([1e-3, 1e-3, 1e-2])   # process noise
    gate:    float = 9.0                                # Mahalanobis^2 threshold

    @staticmethod
    def init(
        frame: WorldFrame,
        detections: list[PoseDetection] | None = None,
    ) -> "EKF":
        """Initialize EKF from existing pose detections. Falls back to zeros."""
        detections = [] if detections is None else detections

        if len(detections) > 0:
            # Prefer template if available, otherwise first detection.
            det = next(
                (d for d in detections if d.name == "template"),
                detections[0],
            )

            mean = det.pose.copy()
            mean[2] = wrap_angle(mean[2])

            print(
                f"[EKF] init from {det.name}: "
                f"xy=({mean[0]:.3f}, {mean[1]:.3f}), "
                f"yaw={np.degrees(mean[2]):.1f}°"
            )

            return EKF(
                frame=frame,
                mean=mean,
                cov=np.diag([0.01, 0.01, 0.1]),
            )

        print("[EKF] no init detection, defaulting to zeros")

        return EKF(
            frame=frame,
            mean=np.zeros(3),
            cov=np.diag([0.1, 0.1, 1.0]),
        )

    def predict(self) -> "EKF":
        return self._replace(cov=self.cov + self.Q)

    def update_pose(self, z: np.ndarray, R: np.ndarray, gated: bool = True) -> "EKF":
        """
        Fuse one direct pose measurement z = [x, y, yaw].

        Since the measurement is already in the same coordinates as the EKF state,
        the measurement model is simply:

            z = state + noise

        so H = I.
        """
        innov = z - self.mean
        innov[2] = (innov[2] + np.pi) % (2 * np.pi) - np.pi

        S = self.cov + R

        if gated:
            maha2 = float(innov @ np.linalg.solve(S, innov))
            if maha2 > self.gate:
                return self

        K = self.cov @ np.linalg.inv(S)

        mean_new = self.mean + K @ innov
        mean_new[2] = (mean_new[2] + np.pi) % (2 * np.pi) - np.pi
        print(mean_new)

        I = np.eye(3)
        cov_new = (I - K) @ self.cov @ (I - K).T + K @ R @ K.T
        cov_new = 0.5 * (cov_new + cov_new.T)

        return self._replace(mean=mean_new, cov=cov_new)

    def update(self, detections: list[PoseDetection]) -> "EKF":
        """Fuse already-computed pose detections."""
        ekf = self.predict()

        for det in detections:
            z = det.pose.copy()
            z[2] = align_angle_to_reference(z[2], ekf.mean[2])
            ekf = ekf.update_pose(z, det.R, gated=det.gated)

        return ekf

def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi

def align_angle_to_reference(theta: float, ref: float) -> float:
    candidates = np.array([theta, theta + np.pi, theta - np.pi])
    diffs = np.array([wrap_angle(c - ref) for c in candidates])
    return candidates[np.argmin(np.abs(diffs))]