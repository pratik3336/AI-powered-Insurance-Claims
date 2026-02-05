"""
RAG (Retrieval-Augmented Generation) service for damage assessment
Uses historical repair data to provide accurate cost estimates
"""

import logging
from typing import Dict, List, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class RepairKnowledgeBase:
    """
    Knowledge base of vehicle repair costs and damage patterns
    Based on industry data and historical claims
    """

    # Historical repair cost data by damage type and vehicle part
    REPAIR_COSTS = {
        "front-end-damage": {
            "minor": {"min": 1500, "max": 3500, "avg_days": 3},
            "moderate": {"min": 3500, "max": 8000, "avg_days": 5},
            "major": {"min": 8000, "max": 15000, "avg_days": 10},
            "description": "Front bumper, grille, headlights, hood damage",
            "common_parts": ["bumper", "grille", "headlights", "hood", "radiator"]
        },
        "rear-end-damage": {
            "minor": {"min": 1200, "max": 3000, "avg_days": 3},
            "moderate": {"min": 3000, "max": 7000, "avg_days": 5},
            "major": {"min": 7000, "max": 14000, "avg_days": 8},
            "description": "Rear bumper, taillights, trunk damage",
            "common_parts": ["rear_bumper", "taillights", "trunk", "exhaust"]
        },
        "side-impact-damage": {
            "minor": {"min": 2000, "max": 4000, "avg_days": 4},
            "moderate": {"min": 4000, "max": 10000, "avg_days": 7},
            "major": {"min": 10000, "max": 25000, "avg_days": 14},
            "description": "Door, fender, side panel damage",
            "common_parts": ["door", "fender", "side_panel", "mirror", "window"]
        },
        "dent": {
            "minor": {"min": 150, "max": 500, "avg_days": 1},
            "moderate": {"min": 500, "max": 1500, "avg_days": 2},
            "major": {"min": 1500, "max": 3000, "avg_days": 3},
            "description": "Dent repair or panel replacement",
            "common_parts": ["panel", "door", "hood", "trunk"]
        },
        "scratch": {
            "minor": {"min": 100, "max": 400, "avg_days": 1},
            "moderate": {"min": 400, "max": 1200, "avg_days": 2},
            "major": {"min": 1200, "max": 2500, "avg_days": 3},
            "description": "Paint scratch repair and refinishing",
            "common_parts": ["paint", "clearcoat", "panel"]
        },
        "broken_glass": {
            "minor": {"min": 200, "max": 400, "avg_days": 1},
            "moderate": {"min": 400, "max": 800, "avg_days": 1},
            "major": {"min": 800, "max": 1500, "avg_days": 2},
            "description": "Window or windshield replacement",
            "common_parts": ["windshield", "window", "side_glass", "rear_glass"]
        },
        "crushed": {
            "minor": {"min": 3000, "max": 6000, "avg_days": 7},
            "moderate": {"min": 6000, "max": 15000, "avg_days": 14},
            "major": {"min": 15000, "max": 30000, "avg_days": 21},
            "description": "Severe structural damage",
            "common_parts": ["frame", "pillar", "roof", "floor_pan"]
        }
    }

    # Common repair scenarios
    REPAIR_SCENARIOS = [
        {
            "scenario": "Minor front bumper scuff",
            "damage_types": ["scratch", "dent"],
            "severity": "minor",
            "cost_range": (300, 800),
            "repair_time_days": 1,
            "notes": "Buffing and touch-up paint usually sufficient"
        },
        {
            "scenario": "Moderate front-end collision",
            "damage_types": ["front-end-damage", "broken_glass"],
            "severity": "moderate",
            "cost_range": (4000, 9000),
            "repair_time_days": 5,
            "notes": "Bumper replacement, headlight replacement, possible radiator damage"
        },
        {
            "scenario": "Severe side impact",
            "damage_types": ["side-impact-damage", "crushed", "broken_glass"],
            "severity": "major",
            "cost_range": (12000, 28000),
            "repair_time_days": 14,
            "notes": "Multiple panel replacement, possible frame damage, safety systems check required"
        },
        {
            "scenario": "Parking lot door ding",
            "damage_types": ["dent"],
            "severity": "minor",
            "cost_range": (150, 400),
            "repair_time_days": 1,
            "notes": "PDR (paintless dent repair) preferred method"
        },
        {
            "scenario": "Rear-end collision at low speed",
            "damage_types": ["rear-end-damage"],
            "severity": "minor",
            "cost_range": (1500, 3500),
            "repair_time_days": 3,
            "notes": "Bumper replacement, taillight check, trunk alignment"
        }
    ]

    @classmethod
    def get_repair_context(cls, damage_types: List[str], severity: Optional[str] = None) -> Dict:
        """
        Retrieve relevant repair cost context for given damage types

        Args:
            damage_types: List of detected damage types
            severity: Optional severity level (minor/moderate/major)

        Returns:
            Dict with repair cost ranges, scenarios, and recommendations
        """
        context = {
            "damage_breakdown": [],
            "total_cost_min": 0,
            "total_cost_max": 0,
            "estimated_repair_days": 0,
            "similar_scenarios": [],
            "recommendations": []
        }

        # Get cost data for each damage type
        for damage_type in damage_types:
            # Normalize damage type names
            normalized = damage_type.lower().replace(" ", "_").replace("-", "_")

            if normalized in cls.REPAIR_COSTS:
                damage_data = cls.REPAIR_COSTS[normalized]

                # Use provided severity or default to moderate
                sev = severity if severity else "moderate"
                if sev not in damage_data:
                    sev = "moderate"

                cost_info = damage_data[sev]

                context["damage_breakdown"].append({
                    "type": damage_type,
                    "severity": sev,
                    "cost_min": cost_info["min"],
                    "cost_max": cost_info["max"],
                    "repair_days": cost_info["avg_days"],
                    "description": damage_data["description"],
                    "common_parts": damage_data["common_parts"]
                })

                # Accumulate totals
                context["total_cost_min"] += cost_info["min"]
                context["total_cost_max"] += cost_info["max"]
                context["estimated_repair_days"] = max(
                    context["estimated_repair_days"],
                    cost_info["avg_days"]
                )

        # Find similar repair scenarios
        for scenario in cls.REPAIR_SCENARIOS:
            # Check if scenario damage types match detected damages
            scenario_types = set(dt.lower().replace(" ", "_").replace("-", "_")
                               for dt in scenario["damage_types"])
            detected_types = set(dt.lower().replace(" ", "_").replace("-", "_")
                               for dt in damage_types)

            # Calculate overlap
            overlap = len(scenario_types & detected_types)
            if overlap > 0:
                context["similar_scenarios"].append({
                    "scenario": scenario["scenario"],
                    "cost_range": scenario["cost_range"],
                    "repair_time": scenario["repair_time_days"],
                    "notes": scenario["notes"],
                    "relevance": overlap / len(scenario_types)
                })

        # Sort scenarios by relevance
        context["similar_scenarios"].sort(key=lambda x: x["relevance"], reverse=True)

        # Generate recommendations
        if context["total_cost_max"] > 15000:
            context["recommendations"].append(
                "High repair cost - recommend structural inspection"
            )
        if context["estimated_repair_days"] > 7:
            context["recommendations"].append(
                "Extended repair time - arrange rental vehicle"
            )
        if any(d["type"].lower() == "crushed" for d in context["damage_breakdown"]):
            context["recommendations"].append(
                "Severe damage - total loss evaluation recommended"
            )
        if len(damage_types) >= 3:
            context["recommendations"].append(
                "Multiple damage areas - comprehensive damage assessment required"
            )

        return context


class RepairRAGService:
    """
    RAG service for augmenting damage assessments with repair knowledge
    """

    def __init__(self):
        self.knowledge_base = RepairKnowledgeBase()

    def augment_damage_assessment(
        self,
        detected_damages: List[str],
        severity: Optional[str] = None,
        additional_context: Optional[Dict] = None
    ) -> Dict:
        """
        Augment damage assessment with RAG-retrieved repair knowledge

        Args:
            detected_damages: List of detected damage types
            severity: Optional overall severity
            additional_context: Optional additional context (vehicle info, etc.)

        Returns:
            Augmented assessment with cost estimates and recommendations
        """
        # Retrieve repair context
        repair_context = self.knowledge_base.get_repair_context(
            detected_damages,
            severity
        )

        # Build augmented prompt context
        prompt_context = self._build_prompt_context(repair_context)

        return {
            "repair_context": repair_context,
            "prompt_augmentation": prompt_context,
            "cost_estimate": {
                "min": repair_context["total_cost_min"],
                "max": repair_context["total_cost_max"],
                "confidence": "high" if repair_context["similar_scenarios"] else "medium"
            },
            "repair_timeline": {
                "estimated_days": repair_context["estimated_repair_days"],
                "confidence": "high"
            },
            "recommendations": repair_context["recommendations"]
        }

    def _build_prompt_context(self, repair_context: Dict) -> str:
        """Build context string to augment AI prompts"""

        context_parts = ["REPAIR COST KNOWLEDGE BASE:\n"]

        # Add damage breakdown
        if repair_context["damage_breakdown"]:
            context_parts.append("\nDetected Damage Cost Estimates:")
            for damage in repair_context["damage_breakdown"]:
                context_parts.append(
                    f"- {damage['type']} ({damage['severity']}): "
                    f"${damage['cost_min']:,} - ${damage['cost_max']:,} "
                    f"({damage['repair_days']} days)"
                )

        # Add similar scenarios
        if repair_context["similar_scenarios"]:
            context_parts.append("\n\nSimilar Past Claims:")
            for scenario in repair_context["similar_scenarios"][:3]:
                context_parts.append(
                    f"- {scenario['scenario']}: "
                    f"${scenario['cost_range'][0]:,} - ${scenario['cost_range'][1]:,} "
                    f"({scenario['repair_time']} days)"
                )

        # Add recommendations
        if repair_context["recommendations"]:
            context_parts.append("\n\nRecommendations:")
            for rec in repair_context["recommendations"]:
                context_parts.append(f"- {rec}")

        return "\n".join(context_parts)


# Global instance
repair_rag_service = RepairRAGService()
