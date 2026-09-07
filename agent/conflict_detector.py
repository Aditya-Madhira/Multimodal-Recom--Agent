"""
Multimodal Evidence Conflict Detector
Performs cross-modal reconciliation across visual features, manufacturer specifications,
marketing claims, and real-world customer reviews.
"""

from typing import List, Dict, Any
from .azure_llm import AzureOpenAIClient


class MultimodalConflictDetector:
    def __init__(self):
        self.reasoning_client = AzureOpenAIClient()

    def evaluate_candidates(
        self,
        retrieved_items: List[Dict[str, Any]],
        query: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Takes candidates retrieved by the multimodal vector store,
        runs evidence cross-examination on each item, and generates
        calibrated recommendation decisions.
        """
        evaluated_items = []

        for item in retrieved_items:
            product = item.get("product", {})
            similarity = item.get("similarity_score", 0.0)

            # Reconcile multi-source evidence
            evidence_result = self.reasoning_client.reconcile_evidence(product, query)

            confidence = evidence_result.get("calibrated_confidence", 0.5)
            
            # Grounded score balances visual/textual similarity with evidence reliability
            grounded_score = round((similarity * 0.5) + (confidence * 100 * 0.5), 1)

            evaluated_items.append({
                "product": product,
                "similarity_score": similarity,
                "confidence_score": round(confidence * 100, 1),
                "grounded_score": grounded_score,
                "has_conflict": evidence_result.get("has_conflict", False),
                "conflict_type": evidence_result.get("conflict_type", "None"),
                "severity": evidence_result.get("severity", "NONE"),
                "discrepancy_details": evidence_result.get("discrepancy_details", ""),
                "visual_summary": evidence_result.get("visual_evidence_summary", ""),
                "specs_summary": evidence_result.get("specs_evidence_summary", ""),
                "reviews_summary": evidence_result.get("reviews_evidence_summary", ""),
                "verdict": evidence_result.get("verdict", "RECOMMENDED"),
                "engine": evidence_result.get("engine", "Calibrated Evidence Engine")
            })

        # Sort by grounded score (prioritizes high similarity + high evidence consistency)
        evaluated_items.sort(key=lambda x: x["grounded_score"], reverse=True)
        return evaluated_items
