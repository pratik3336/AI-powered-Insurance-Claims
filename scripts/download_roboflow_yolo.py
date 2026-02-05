"""
Download Roboflow car damage dataset in YOLOv5 format
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def download_roboflow_yolo():
    """Download car damage dataset in YOLO format"""

    print("\n" + "="*60)
    print("📥 Downloading Roboflow Car Damage Dataset (YOLO format)")
    print("="*60 + "\n")

    try:
        from roboflow import Roboflow

        # Initialize Roboflow
        print("🔗 Connecting to Roboflow...")
        rf = Roboflow(api_key="unauthorized")

        print("📦 Accessing car-damage-images project...")
        project = rf.workspace("car-damage-kadad").project("car-damage-images")

        print("🎯 Fetching version 3...")
        version = project.version(3)

        print("⬇️  Downloading dataset in YOLOv5 format...")
        dataset = version.download("yolov5", location="data/raw/roboflow_yolo")

        print("\n" + "="*60)
        print("✅ Dataset Downloaded Successfully!")
        print("="*60)
        print(f"\n📁 Location: {dataset.location}")

        # Check what we got
        dataset_path = Path(dataset.location)

        if dataset_path.exists():
            train_images = list((dataset_path / "train" / "images").glob("*.jpg"))
            valid_images = list((dataset_path / "valid" / "images").glob("*.jpg"))
            test_images = list((dataset_path / "test" / "images").glob("*.jpg"))

            print(f"\n📊 Dataset Statistics:")
            print(f"  - Training images: {len(train_images)}")
            print(f"  - Validation images: {len(valid_images)}")
            print(f"  - Test images: {len(test_images)}")
            print(f"  - Total: {len(train_images) + len(valid_images) + len(test_images)}")

            # Check for data.yaml
            yaml_file = dataset_path / "data.yaml"
            if yaml_file.exists():
                print(f"\n✅ Found data.yaml configuration")
                print(f"📄 Path: {yaml_file}")

        print("\n" + "="*60)
        print("🎯 Next Steps:")
        print("1. Set up YOLOv5 environment")
        print("2. Train custom damage detection model")
        print("3. Integrate into ClaimGuard")
        print("="*60 + "\n")

        return dataset.location

    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Verify Roboflow is accessible")
        print("3. Try manual download from:")
        print("   https://universe.roboflow.com/car-damage-kadad/car-damage-images/dataset/3")
        return None


if __name__ == "__main__":
    download_roboflow_yolo()
