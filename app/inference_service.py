from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


@dataclass
class DetectionResult:
    boxes: np.ndarray  # (N, 4) xyxy
    confidences: np.ndarray  # (N,)
    class_ids: np.ndarray  # (N,)
    class_names: List[str]
    risk_score: float
    risk_grid: np.ndarray  # (H_grid, W_grid) heatmap in [0,1]
    scenario_summary: str
    driving_instruction: str  # concise maneuver suggestion for the "self-driving" HUD


class SelfDrivingInferenceService:
    """
    Thin wrapper around a YOLO model that adds:
    - heuristic risk scoring
    - spatial risk heatmap
    - natural-language scenario summary
    """

    def __init__(self, weights_path: str, device: str | None = None) -> None:
        self.model = YOLO(weights_path)
        if device is not None:
            self.model.to(device)

        # YOLO stores class names in model.names (dict[int, str])
        self.class_names = self.model.model.names if hasattr(self.model, "model") else self.model.names

    def _yolo_predict(self, image_bgr: np.ndarray):
        results = self.model.predict(source=image_bgr, verbose=False)[0]
        return results

    def _compute_risk_features(
        self, boxes_xyxy: np.ndarray, class_ids: np.ndarray, confidences: np.ndarray, img_shape: Tuple[int, int, int]
    ) -> Tuple[float, np.ndarray, str, str]:
        """
        Very simple heuristic "risk" model:
        - Higher risk for objects near image center / bottom (assume ego-car is at bottom center).
        - Pedestrians and vehicles contribute more to risk than signs.
        - Red/unknown traffic lights add risk.
        """
        h, w, _ = img_shape
        grid_h, grid_w = 10, 16
        risk_grid = np.zeros((grid_h, grid_w), dtype=float)

        if len(boxes_xyxy) == 0:
            summary = "Clear road ahead. No significant objects detected."
            instruction = "Maintain speed and lane. Road is clear."
            return 0.05, risk_grid, summary, instruction

        # Simple class roles based on common COCO / driving datasets
        def class_role(cid: int) -> str:
            name = self.class_names.get(int(cid), "").lower()
            if any(k in name for k in ["person", "pedestrian"]):
                return "pedestrian"
            if any(k in name for k in ["car", "truck", "bus", "motor", "bicycle", "bike", "vehicle"]):
                return "vehicle"
            if "light" in name:
                return "traffic_light"
            if "sign" in name:
                return "traffic_sign"
            return "other"

        center_x, center_y = w / 2.0, h * 0.8  # ego viewpoint
        total_risk = 0.0
        lateral_score_left = 0.0
        lateral_score_right = 0.0

        for (x1, y1, x2, y2), cid, conf in zip(boxes_xyxy, class_ids, confidences):
            role = class_role(int(cid))
            bx_center = (x1 + x2) / 2.0
            by_center = (y1 + y2) / 2.0

            # Distance weight (closer -> higher)
            dist = np.hypot((bx_center - center_x) / w, (by_center - center_y) / h)
            dist_weight = np.exp(-4 * dist)  # decays quickly with distance

            # Role weight
            if role == "pedestrian":
                role_w = 1.5
            elif role == "vehicle":
                role_w = 1.2
            elif role == "traffic_light":
                role_w = 1.0
            elif role == "traffic_sign":
                role_w = 0.8
            else:
                role_w = 0.5

            base = float(conf) * role_w * dist_weight
            total_risk += base

            # Lateral distribution for simple steering hints
            if ((x1 + x2) / 2.0) < center_x:
                lateral_score_left += base
            else:
                lateral_score_right += base

            # Accumulate into grid
            gx1 = int(np.clip(x1 / w * grid_w, 0, grid_w - 1))
            gx2 = int(np.clip(x2 / w * grid_w, 0, grid_w - 1))
            gy1 = int(np.clip(y1 / h * grid_h, 0, grid_h - 1))
            gy2 = int(np.clip(y2 / h * grid_h, 0, grid_h - 1))
            risk_grid[gy1 : gy2 + 1, gx1 : gx2 + 1] += base

        # Normalize
        if risk_grid.max() > 0:
            risk_grid /= risk_grid.max()
        # Clip global risk to [0,1] with a soft cap
        risk_score = float(1 - np.exp(-total_risk))

        summary = self._summarize_situation(class_ids, risk_score)
        instruction = self._suggest_maneuver(risk_score, lateral_score_left, lateral_score_right, class_ids, boxes_xyxy, img_shape)
        return risk_score, risk_grid, summary, instruction

    def _summarize_situation(self, class_ids: np.ndarray, risk_score: float) -> str:
        names = [self.class_names.get(int(cid), f"class_{cid}") for cid in class_ids]

        has_ped = any("person" in n.lower() or "pedestrian" in n.lower() for n in names)
        has_vehicle = any(
            any(k in n.lower() for k in ["car", "truck", "bus", "motor", "bicycle", "bike"]) for n in names
        )
        has_light = any("light" in n.lower() for n in names)
        has_sign = any("sign" in n.lower() for n in names)

        if risk_score < 0.25:
            risk_text = "Low risk"
        elif risk_score < 0.6:
            risk_text = "Moderate risk"
        else:
            risk_text = "High risk"

        parts: List[str] = [f"{risk_text} scene."]
        if has_ped and has_vehicle:
            parts.append("Vehicles and pedestrians present; watch for crossings.")
        elif has_ped:
            parts.append("Pedestrians detected; maintain safe distance and be ready to stop.")
        elif has_vehicle:
            parts.append("Traffic ahead; keep safe following distance.")

        if has_light:
            parts.append("Traffic lights in view; obey signal state.")
        if has_sign:
            parts.append("Traffic signs present; adjust speed and behavior accordingly.")

        if not (has_ped or has_vehicle or has_light or has_sign):
            parts.append("Mostly clear road with few critical objects.")

        return " ".join(parts)

    def _suggest_maneuver(
        self,
        risk_score: float,
        lateral_left: float,
        lateral_right: float,
        class_ids: np.ndarray,
        boxes_xyxy: np.ndarray = None,
        img_shape: Tuple[int, int, int] = None,
    ) -> str:
        """
        Heuristic driving instruction based on global risk, lateral distribution, and gaps.
        """
        names = [self.class_names.get(int(cid), f"class_{cid}") for cid in class_ids]
        has_ped = any("person" in n.lower() or "pedestrian" in n.lower() for n in names)
        has_vehicle = any(
            any(k in n.lower() for k in ["car", "truck", "bus", "motor", "bicycle", "bike"]) for n in names
        )
        has_light = any("light" in n.lower() for n in names)

        # Base instruction by risk level
        if risk_score < 0.25:
            base = "Maintain current speed. "
        elif risk_score < 0.6:
            base = "Gently slow down and check surroundings. "
        else:
            base = "URGENT: Brake smoothly and prepare to stop! "

        # Distance and explicit explicit steering based on gaps
        steer = ""
        if boxes_xyxy is not None and len(boxes_xyxy) > 0 and img_shape is not None:
            h, w, _ = img_shape
            center_x = w / 2.0
            
            # Sort boxes by left edge
            boxes_sorted = sorted(boxes_xyxy, key=lambda b: b[0])
            
            largest_gap = 0
            gap_center = center_x
            
            # Left edge to first box
            prev_x = 0
            gap = boxes_sorted[0][0] - prev_x
            if gap > largest_gap:
                largest_gap = gap
                gap_center = gap / 2.0
                
            # Inter-object gaps
            for b in boxes_sorted:
                gap = b[0] - prev_x
                if gap > largest_gap:
                    largest_gap = gap
                    gap_center = prev_x + gap / 2.0
                prev_x = max(prev_x, b[2])
                
            # Last box to right edge
            gap = w - prev_x
            if gap > largest_gap:
                largest_gap = gap
                gap_center = prev_x + gap / 2.0
                
            # Distance approximation (max Y2 = lowest bounding box = closest object)
            closest_y = max((b[3] for b in boxes_xyxy), default=0)
            is_close = closest_y > h * 0.7
            
            # Decide direction
            if abs(gap_center - center_x) < w * 0.15:
                direction = "STRAIGHT AHEAD"
            elif gap_center < center_x:
                direction = "LEFT"
            else:
                direction = "RIGHT"
                
            steer = f"Best open path detected. Steer **{direction}**."
            if is_close and risk_score > 0.3:
                steer = f"Obstacle very close! Evasive action needed: Steer **{direction}** into the open space."
        else:
            steer = "Road is clear, steer STRAIGHT AHEAD."

        # Object-specific hint
        extra = ""
        if has_ped and risk_score >= 0.25:
            extra = " Watch for pedestrians."
        elif has_vehicle and risk_score >= 0.25:
            extra = " Keep safe gap to vehicle in front."
        elif has_light and risk_score >= 0.25:
            extra = " Check the traffic light state."

        return base + steer + extra

    def predict(self, image_bgr: np.ndarray) -> DetectionResult:
        results = self._yolo_predict(image_bgr)
        boxes = results.boxes
        if boxes is None or len(boxes) == 0:
            h, w, _ = image_bgr.shape
            risk_score, risk_grid, summary, instruction = self._compute_risk_features(
                np.zeros((0, 4)), np.zeros((0,), dtype=int), np.zeros((0,), dtype=float), image_bgr.shape
            )
            return DetectionResult(
                boxes=np.zeros((0, 4)),
                confidences=np.zeros((0,)),
                class_ids=np.zeros((0,), dtype=int),
                class_names=[],
                risk_score=risk_score,
                risk_grid=risk_grid,
                scenario_summary=summary,
                driving_instruction=instruction,
            )

        boxes_xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)

        risk_score, risk_grid, summary, instruction = self._compute_risk_features(
            boxes_xyxy, class_ids, confidences, image_bgr.shape
        )

        class_names = [self.class_names.get(int(cid), f"class_{cid}") for cid in class_ids]

        return DetectionResult(
            boxes=boxes_xyxy,
            confidences=confidences,
            class_ids=class_ids,
            class_names=class_names,
            risk_score=risk_score,
            risk_grid=risk_grid,
            scenario_summary=summary,
            driving_instruction=instruction,
        )


def draw_detections_on_image(image_bgr: np.ndarray, det: DetectionResult) -> np.ndarray:
    """Overlay YOLO detections and risk information on the image."""
    output = image_bgr.copy()
    h, w, _ = output.shape

    for (x1, y1, x2, y2), conf, cid, name in zip(
        det.boxes, det.confidences, det.class_ids, det.class_names
    ):
        color = (0, 255, 0)
        cv2.rectangle(output, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        label = f"{name} {conf:.2f}"
        cv2.putText(
            output,
            label,
            (int(x1), max(0, int(y1) - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    # Add global risk score bar
    bar_w = int(w * 0.4)
    bar_h = 18
    x0, y0 = 10, 10
    cv2.rectangle(output, (x0, y0), (x0 + bar_w, y0 + bar_h), (50, 50, 50), -1)
    filled_w = int(bar_w * det.risk_score)
    cv2.rectangle(output, (x0, y0), (x0 + filled_w, y0 + bar_h), (0, 0, 255), -1)
    cv2.putText(
        output,
        f"Risk: {det.risk_score:.2f}",
        (x0 + 5, y0 + bar_h - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return output

