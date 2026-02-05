"""
YOLO inference service for vehicle and damage detection
Uses pre-trained YOLO models until custom model is trained
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class YOLODamageDetector:
    """
    YOLO-based damage detection service

    Currently uses pre-trained YOLO models for vehicle detection.
    Will be upgraded to custom damage detection model once trained.
    """

    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.custom_model_path = Path("app/ml/models/yolo_damage/damage_detector_v1/weights/best.pt")

    def load_model(self):
        """Load YOLO model (pre-trained or custom if available)"""
        try:
            from ultralytics import YOLO

            # Try to load custom model first
            if self.custom_model_path.exists():
                logger.info(f"Loading custom damage detection model from {self.custom_model_path}")
                self.model = YOLO(str(self.custom_model_path))
                logger.info("Custom model loaded successfully")
            else:
                # Fall back to pre-trained YOLOv8
                logger.info("Custom model not found, using pre-trained YOLOv8n")
                self.model = YOLO('yolov8n.pt')  # Lightweight nano model
                logger.info("Pre-trained model loaded successfully")

            self.model_loaded = True
            return True

        except ImportError:
            logger.warning("Ultralytics not available. YOLO detection disabled.")
            self.model_loaded = False
            return False
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model_loaded = False
            return False

    def detect_damage(
        self,
        image_source: Union[str, Path, bytes, np.ndarray],
        confidence_threshold: float = 0.25
    ) -> Dict:
        """
        Detect damage in image using YOLO

        Args:
            image_source: Image path, bytes, or numpy array
            confidence_threshold: Minimum confidence for detections

        Returns:
            Dict with damage detections and metadata
        """
        if not self.model_loaded:
            if not self.load_model():
                return {
                    "success": False,
                    "error": "YOLO model not available",
                    "detections": [],
                    "using_custom_model": False
                }

        try:
            # Run inference
            results = self.model(image_source, conf=confidence_threshold)

            # Parse results
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    detection = {
                        "class_id": int(box.cls[0]),
                        "class_name": result.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                        "bbox_normalized": box.xywhn[0].tolist()  # [x_center, y_center, width, height] normalized
                    }
                    detections.append(detection)

            # Analyze damage types if using custom model
            damage_analysis = self._analyze_detections(detections)

            return {
                "success": True,
                "detections": detections,
                "damage_analysis": damage_analysis,
                "total_detections": len(detections),
                "using_custom_model": self.custom_model_path.exists(),
                "model_path": str(self.custom_model_path) if self.custom_model_path.exists() else "yolov8n.pt"
            }

        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "detections": [],
                "using_custom_model": False
            }

    def _analyze_detections(self, detections: List[Dict]) -> Dict:
        """Analyze detections to provide damage summary"""

        if not detections:
            return {
                "has_damage": False,
                "damage_types": [],
                "severity_estimate": "none",
                "confidence": 0.0
            }

        # If using custom model, damage classes are known
        damage_classes = {
            'Broken glass', 'Dent', 'Scratch',
            'front-end-damage', 'rear-end-damage', 'side-impact-damage'
        }

        detected_damage_types = [
            d["class_name"] for d in detections
            if d["class_name"] in damage_classes
        ]

        # Estimate severity based on number and confidence of detections
        avg_confidence = np.mean([d["confidence"] for d in detections])

        if len(detected_damage_types) >= 3:
            severity = "major"
        elif len(detected_damage_types) >= 2:
            severity = "moderate"
        elif len(detected_damage_types) >= 1:
            severity = "minor"
        else:
            severity = "none"

        return {
            "has_damage": len(detected_damage_types) > 0,
            "damage_types": list(set(detected_damage_types)),
            "damage_count": len(detected_damage_types),
            "severity_estimate": severity,
            "confidence": float(avg_confidence)
        }

    def get_model_info(self) -> Dict:
        """Get information about loaded model"""
        return {
            "model_loaded": self.model_loaded,
            "using_custom_model": self.custom_model_path.exists(),
            "custom_model_path": str(self.custom_model_path),
            "model_available": self.model is not None
        }


# Global instance
yolo_detector = YOLODamageDetector()


def detect_damage_yolo(image_source: Union[str, Path, bytes, np.ndarray]) -> Dict:
    """
    Convenience function for YOLO damage detection

    Usage:
        result = detect_damage_yolo("path/to/image.jpg")
        if result["success"]:
            print(f"Found {result['total_detections']} detections")
            print(f"Damage: {result['damage_analysis']}")
    """
    return yolo_detector.detect_damage(image_source)
