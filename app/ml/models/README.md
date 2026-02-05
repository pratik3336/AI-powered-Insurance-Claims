# ML Models

This directory contains trained machine learning models for ClaimGuard.

## Model Files

Due to their size, model files are not tracked in git. You need to train or download them separately:

### Fraud Detection Model
- **File**: `fraud_detector_v1.pkl` (~96 KB)
- **Type**: XGBoost classifier
- **Training**: Run `python -m app.ml.training.train_fraud_model`

### YOLO Damage Detection Model
- **File**: `yolo_damage/damage_detector_v1/weights/best.pt` (~6 MB)
- **Type**: YOLOv8 Nano model
- **Training**: Run `python scripts/train_ultralytics_damage.py`
- **Dataset**: Roboflow Car Damage Dataset (312 images)

## Directory Structure

```
models/
├── README.md
├── fraud_detector_v1.pkl          # XGBoost fraud model (not tracked)
├── best.pt                        # Quick access YOLO model (not tracked)
└── yolo_damage/
    └── damage_detector_v1/
        └── weights/
            └── best.pt            # Trained YOLO weights (not tracked)
```

## Note

Model files are excluded from git to keep the repository lightweight. After cloning, you'll need to either:
1. Train the models yourself using the provided scripts
2. Download pre-trained models separately
