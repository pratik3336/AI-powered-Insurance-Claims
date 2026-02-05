"""
Train YOLO damage detection model using Ultralytics library
"""

from ultralytics import YOLO
from pathlib import Path


def train_damage_model():
    """Train YOLO model on car damage dataset"""

    print("\n" + "="*60)
    print("🚀 Training YOLO Damage Detection Model")
    print("="*60 + "\n")

    # Dataset configuration
    data_yaml = Path("/Users/aakashbhatt/ClaimGuard/data/raw/roboflow_yolo/data.yaml")

    if not data_yaml.exists():
        print(f"❌ Dataset configuration not found: {data_yaml}")
        return

    # Initialize YOLO model (YOLOv8 nano for speed)
    print("📦 Loading YOLOv8 nano model...")
    model = YOLO('yolov8n.pt')

    # Training parameters
    print(f"\n📊 Training Configuration:")
    print(f"  - Model: YOLOv8 Nano")
    print(f"  - Epochs: 50")
    print(f"  - Batch size: 16")
    print(f"  - Image size: 640")
    print(f"  - Dataset: {data_yaml}")
    print(f"  - Device: CUDA if available, else CPU")

    # Train the model
    print("\n🏋️  Starting training...\n")

    try:
        results = model.train(
            data=str(data_yaml),
            epochs=50,
            imgsz=640,
            batch=16,
            name='damage_detector_v1',
            project='/Users/aakashbhatt/ClaimGuard/app/ml/models/yolo_damage',
            cache=True,
            patience=10,  # Early stopping
            save_period=10,  # Save checkpoint every 10 epochs
            verbose=True
        )

        print("\n" + "="*60)
        print("✅ Training Complete!")
        print("="*60)

        # Model location
        model_path = Path("/Users/aakashbhatt/ClaimGuard/app/ml/models/yolo_damage/damage_detector_v1/weights/best.pt")
        print(f"\n🎯 Best model: {model_path}")

        # Validation results
        print(f"\n📊 Final Results:")
        print(f"  - mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
        print(f"  - mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")

        print("\n🚀 Next steps:")
        print("1. Test model on validation images")
        print("2. Integrate into ClaimGuard dashboard")
        print("3. Compare with OpenAI Vision API")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("\nTry reducing batch size if running out of memory:")
        print("  Edit the script and change batch=16 to batch=8")


if __name__ == "__main__":
    train_damage_model()
