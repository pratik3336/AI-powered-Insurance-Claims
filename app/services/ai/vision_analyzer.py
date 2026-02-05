"""
OpenAI Vision API integration for vehicle damage assessment
Analyzes images and extracts damage information
"""

import base64
from pathlib import Path
from typing import Dict, List, Optional, Union
import json
from io import BytesIO

from openai import OpenAI
from PIL import Image

from app.core.config import settings
from app.services.ai.repair_rag_service import repair_rag_service


class VisionAnalyzer:
    """
    Analyzes vehicle damage images using OpenAI Vision API
    Extracts damage type, severity, affected areas, and cost estimates
    """

    def __init__(self):
        self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_VISION_MODEL

    def encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64 for API"""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def encode_image_bytes(self, image_bytes: bytes) -> str:
        """Encode image bytes to base64"""
        return base64.b64encode(image_bytes).decode('utf-8')

    def analyze_damage(
        self,
        image_source: Union[str, Path, bytes],
        claim_context: Optional[Dict] = None
    ) -> Dict:
        """
        Analyze vehicle damage from image

        Args:
            image_source: Path to image file or image bytes
            claim_context: Optional context about the claim

        Returns:
            Dict with damage analysis results
        """

        # Encode image
        if isinstance(image_source, bytes):
            base64_image = self.encode_image_bytes(image_source)
        else:
            base64_image = self.encode_image(image_source)

        # Build analysis prompt
        prompt = self._build_damage_analysis_prompt(claim_context)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert auto insurance claims adjuster specializing in vehicle damage assessment. Analyze images carefully and provide detailed, accurate assessments."
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1000,
                temperature=0.2
            )

            # Parse response
            analysis_text = response.choices[0].message.content
            analysis = self._parse_analysis_response(analysis_text)

            # Augment with RAG if damage types detected
            if analysis.get('success') and analysis.get('damage_types'):
                rag_augmentation = repair_rag_service.augment_damage_assessment(
                    detected_damages=analysis['damage_types'],
                    severity=analysis.get('severity', '').lower().replace('_', ' '),
                    additional_context=claim_context
                )

                # Override cost estimates with RAG data (more accurate)
                if rag_augmentation['cost_estimate']['confidence'] == 'high':
                    analysis['cost_estimate'] = {
                        'min': rag_augmentation['cost_estimate']['min'],
                        'max': rag_augmentation['cost_estimate']['max']
                    }
                    analysis['cost_estimate_source'] = 'RAG (Historical Data)'
                else:
                    analysis['cost_estimate_source'] = 'AI Estimate'

                # Add RAG context
                analysis['rag_context'] = {
                    'repair_timeline': rag_augmentation['repair_timeline'],
                    'recommendations': rag_augmentation['recommendations'],
                    'similar_scenarios': rag_augmentation['repair_context'].get('similar_scenarios', [])[:2]
                }

            # Add metadata
            analysis['model'] = self.model
            analysis['raw_response'] = analysis_text
            analysis['usage'] = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }

            return analysis

        except Exception as e:
            return {
                'error': str(e),
                'success': False
            }

    def analyze_multiple_images(
        self,
        image_sources: List[Union[str, Path, bytes]],
        claim_context: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Analyze multiple images for a single claim

        Args:
            image_sources: List of image paths or bytes
            claim_context: Optional context about the claim

        Returns:
            List of damage analysis results
        """
        results = []
        for image_source in image_sources:
            analysis = self.analyze_damage(image_source, claim_context)
            results.append(analysis)

        # Aggregate results
        aggregated = self._aggregate_analyses(results)
        return aggregated

    def _build_damage_analysis_prompt(self, claim_context: Optional[Dict] = None) -> str:
        """Build the analysis prompt based on context"""

        base_prompt = """Analyze this vehicle damage image and provide a detailed assessment in JSON format.

Please identify:

1. **Damage Types**: List all types of damage visible (scratch, dent, crack, broken, shattered, crushed, burned, water damage, etc.)

2. **Severity**: Rate the overall severity as:
   - MINOR: Cosmetic damage, vehicle fully functional
   - MODERATE: Functional damage, safe to drive with repairs needed
   - MAJOR: Significant damage, may not be safe to drive
   - TOTAL_LOSS: Vehicle likely totaled

3. **Affected Areas**: List all vehicle parts affected (bumper, hood, door, fender, windshield, headlight, etc.)

4. **Cost Estimate**: Provide a repair cost range (min and max in USD)

5. **Severity Score**: Rate damage severity from 0-100

6. **Additional Notes**: Any other relevant observations

Format your response as valid JSON with this structure:
{
  "damage_types": ["type1", "type2"],
  "severity": "MINOR|MODERATE|MAJOR|TOTAL_LOSS",
  "severity_score": 0-100,
  "affected_areas": ["area1", "area2"],
  "cost_estimate": {
    "min": 1000,
    "max": 3000
  },
  "notes": "Additional observations..."
}
"""

        if claim_context:
            context_info = f"\n\nClaim Context:\n"
            if 'incident_description' in claim_context:
                context_info += f"- Incident: {claim_context['incident_description']}\n"
            if 'vehicle_make' in claim_context:
                context_info += f"- Vehicle: {claim_context['vehicle_make']} {claim_context.get('vehicle_model', '')}\n"
            if 'claimed_damage' in claim_context:
                context_info += f"- Claimed damage: ${claim_context['claimed_damage']:,.2f}\n"

            base_prompt += context_info

        return base_prompt

    def _parse_analysis_response(self, response_text: str) -> Dict:
        """Parse the AI response into structured data"""

        try:
            # Try to extract JSON from response
            # Sometimes the model wraps JSON in markdown code blocks
            if '```json' in response_text:
                json_str = response_text.split('```json')[1].split('```')[0].strip()
            elif '```' in response_text:
                json_str = response_text.split('```')[1].split('```')[0].strip()
            else:
                json_str = response_text.strip()

            parsed = json.loads(json_str)
            parsed['success'] = True
            return parsed

        except json.JSONDecodeError:
            # Fallback: return raw response
            return {
                'success': False,
                'raw_response': response_text,
                'error': 'Failed to parse JSON response'
            }

    def _aggregate_analyses(self, analyses: List[Dict]) -> Dict:
        """Aggregate multiple image analyses into single assessment"""

        if not analyses:
            return {'error': 'No analyses provided'}

        # Filter successful analyses
        successful = [a for a in analyses if a.get('success')]

        if not successful:
            return {
                'success': False,
                'error': 'All analyses failed',
                'individual_analyses': analyses
            }

        # Aggregate damage types (union)
        all_damage_types = set()
        for analysis in successful:
            if 'damage_types' in analysis:
                all_damage_types.update(analysis['damage_types'])

        # Aggregate affected areas (union)
        all_affected_areas = set()
        for analysis in successful:
            if 'affected_areas' in analysis:
                all_affected_areas.update(analysis['affected_areas'])

        # Take maximum severity
        severity_order = {'MINOR': 1, 'MODERATE': 2, 'MAJOR': 3, 'TOTAL_LOSS': 4}
        max_severity = 'MINOR'
        max_severity_score = 0

        for analysis in successful:
            if 'severity' in analysis:
                current_severity = analysis['severity']
                if severity_order.get(current_severity, 0) > severity_order.get(max_severity, 0):
                    max_severity = current_severity

            if 'severity_score' in analysis:
                max_severity_score = max(max_severity_score, analysis['severity_score'])

        # Aggregate cost estimates (take max range)
        min_cost = max(a.get('cost_estimate', {}).get('min', 0) for a in successful)
        max_cost = max(a.get('cost_estimate', {}).get('max', 0) for a in successful)

        # Combine notes
        all_notes = []
        for i, analysis in enumerate(successful):
            if 'notes' in analysis and analysis['notes']:
                all_notes.append(f"Image {i+1}: {analysis['notes']}")

        return {
            'success': True,
            'damage_types': list(all_damage_types),
            'severity': max_severity,
            'severity_score': max_severity_score,
            'affected_areas': list(all_affected_areas),
            'cost_estimate': {
                'min': min_cost,
                'max': max_cost
            },
            'notes': '; '.join(all_notes),
            'image_count': len(analyses),
            'successful_analyses': len(successful),
            'individual_analyses': analyses
        }

    def estimate_api_cost(self, num_images: int, avg_tokens_per_image: int = 800) -> float:
        """
        Estimate OpenAI API cost for damage assessment

        Args:
            num_images: Number of images to analyze
            avg_tokens_per_image: Average tokens per image (default 800)

        Returns:
            Estimated cost in USD
        """
        # gpt-4o-mini pricing (as of Jan 2025)
        # $0.15 per 1M input tokens, $0.60 per 1M output tokens
        input_cost_per_token = 0.15 / 1_000_000
        output_cost_per_token = 0.60 / 1_000_000

        # High detail images cost ~765 tokens base + prompt
        image_tokens = 765
        prompt_tokens = 300
        output_tokens = avg_tokens_per_image

        total_input_tokens = num_images * (image_tokens + prompt_tokens)
        total_output_tokens = num_images * output_tokens

        total_cost = (
            total_input_tokens * input_cost_per_token +
            total_output_tokens * output_cost_per_token
        )

        return total_cost


# Singleton instance
_vision_analyzer = None

def get_vision_analyzer() -> VisionAnalyzer:
    """Get or create VisionAnalyzer instance"""
    global _vision_analyzer
    if _vision_analyzer is None:
        _vision_analyzer = VisionAnalyzer()
    return _vision_analyzer
