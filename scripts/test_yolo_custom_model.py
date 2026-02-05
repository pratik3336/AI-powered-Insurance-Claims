"""
Test custom YOLO damage detection model
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.damage.yolo_inference import yolo_detector


def test_custom_model():
    """Test if custom YOLO model loads and works"""

    print("\n" + "="*60)
    print("🧪 Testing Custom YOLO Damage Detection Model")
    print("="*60 + "\n")

    # Check model info
    info = yolo_detector.get_model_info()

    print("📊 Model Information:")
    print(f"  Model loaded: {info['model_loaded']}")
    print(f"  Using custom model: {info['using_custom_model']}")
    print(f"  Model path: {info['custom_model_path']}")

    if not info['using_custom_model']:
        print("\n⚠️  Custom model not detected!")
        print(f"Expected location: {info['custom_model_path']}")
        return False

    # Try to load the model
    print("\n📥 Loading model...")
    success = yolo_detector.load_model()

    if not success:
        print("❌ Failed to load model")
        return False

    print("✅ Model loaded successfully!")

    # Get a sample image to test
    sample_images_dir = Path("data/raw/roboflow_damage/test/images")

    if sample_images_dir.exists():
        sample_images = list(sample_images_dir.glob("*.jpg"))

        if sample_images:
            print(f"\n🔍 Testing on sample image: {sample_images[0].name}")

            # Run inference
            result = yolo_detector.detect_damage(str(sample_images[0]))

            if result['success']:
                print("\n✅ Inference successful!")
                print(f"\n📊 Results:")
                print(f"  Detections found: {result['total_detections']}")
                print(f"  Using custom model: {result['using_custom_model']}")

                if result['detections']:
                    print(f"\n🎯 Detected damages:")
                    for i, det in enumerate(result['detections'], 1):
                        print(f"    {i}. {det['class_name']} ({det['confidence']:.2%} confidence)")

                # Print damage analysis
                analysis = result.get('damage_analysis', {})
                print(f"\n📋 Damage Analysis:")
                print(f"  Has damage: {analysis.get('has_damage', False)}")
                print(f"  Damage types: {', '.join(analysis.get('damage_types', []))}")
                print(f"  Severity: {analysis.get('severity_estimate', 'unknown')}")
                print(f"  Confidence: {analysis.get('confidence', 0):.2%}")

                return True
            else:
                print(f"\n❌ Inference failed: {result.get('error', 'Unknown error')}")
                return False
        else:
            print("\n⚠️  No test images found")
            print("Model is loaded but can't test without images")
            return True
    else:
        print("\n⚠️  Test images directory not found")
        print("Model is loaded but can't test without images")
        return True


if __name__ == "__main__":
    success = test_custom_model()

    print("\n" + "="*60)
    if success:
        print("✅ Custom YOLO Model Test: PASSED")
        print("="*60)
        print("\n🚀 Your custom model is ready to use in ClaimGuard!")
        print("\nNext steps:")
        print("1. Restart the Gradio dashboard")
        print("2. Upload car damage images in the Damage Assessment tab")
        print("3. The custom model will automatically analyze them!")
    else:
        print("❌ Custom YOLO Model Test: FAILED")
        print("="*60)
        print("\nPlease check:")
        print("1. Model file exists at: app/ml/models/yolo_damage/damage_detector_v1/weights/best.pt")
        print("2. Ultralytics library is installed: pip install ultralytics")

    print("="*60 + "\n")
