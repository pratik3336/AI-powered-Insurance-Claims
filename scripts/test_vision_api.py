"""
Test OpenAI Vision API with sample vehicle damage image
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.ai.vision_analyzer import get_vision_analyzer
import json


def test_with_url():
    """Test with a sample vehicle damage image URL"""

    print("🔍 Testing OpenAI Vision API for Damage Assessment\n")
    print("="*60)

    analyzer = get_vision_analyzer()

    # Test with a sample context
    claim_context = {
        'incident_description': 'Vehicle collision with another car',
        'vehicle_make': 'Toyota',
        'vehicle_model': 'Camry',
        'claimed_damage': 5000.00
    }

    print("\n📝 Claim Context:")
    print(json.dumps(claim_context, indent=2))

    print("\n⏳ Analyzing sample image...")
    print("(Using a public sample car damage image)")

    # For testing, we'll need an actual image
    # You can download a sample image or use a URL
    print("\n⚠️  To test with an actual image:")
    print("1. Download a car damage image")
    print("2. Save it as 'test_damage.jpg' in this directory")
    print("3. Run this script again")

    # Estimate cost
    cost_per_image = analyzer.estimate_api_cost(1)
    print(f"\n💰 Estimated API cost per image: ${cost_per_image:.4f}")

    cost_1000 = analyzer.estimate_api_cost(1000)
    print(f"💰 Estimated cost for 1,000 images: ${cost_1000:.2f}")

    # Check if test image exists
    test_image = Path(__file__).parent / "test_damage.jpg"
    if test_image.exists():
        print(f"\n✅ Found test image: {test_image}")
        print("Analyzing...")

        result = analyzer.analyze_damage(test_image, claim_context)

        if result.get('success'):
            print("\n" + "="*60)
            print("✅ DAMAGE ASSESSMENT RESULTS")
            print("="*60)
            print(f"\nDamage Types: {', '.join(result.get('damage_types', []))}")
            print(f"Severity: {result.get('severity')}")
            print(f"Severity Score: {result.get('severity_score')}/100")
            print(f"Affected Areas: {', '.join(result.get('affected_areas', []))}")

            cost_est = result.get('cost_estimate', {})
            print(f"\nCost Estimate: ${cost_est.get('min', 0):,.2f} - ${cost_est.get('max', 0):,.2f}")

            if result.get('notes'):
                print(f"\nNotes: {result['notes']}")

            # Token usage
            usage = result.get('usage', {})
            print(f"\n📊 API Usage:")
            print(f"  Input tokens: {usage.get('prompt_tokens', 0)}")
            print(f"  Output tokens: {usage.get('completion_tokens', 0)}")
            print(f"  Total tokens: {usage.get('total_tokens', 0)}")

            print("\n" + "="*60)
        else:
            print(f"\n❌ Analysis failed: {result.get('error')}")
            if 'raw_response' in result:
                print(f"Raw response: {result['raw_response']}")
    else:
        print(f"\n❌ Test image not found at: {test_image}")
        print("\nTo run a full test:")
        print("1. Download a car damage image from the internet")
        print("2. Save it as 'test_damage.jpg' in the scripts/ directory")
        print("3. Run: python scripts/test_vision_api.py")


if __name__ == "__main__":
    try:
        test_with_url()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
