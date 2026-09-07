"""
Multimodal Evidence Conflict Detector
Performs cross-modal reconciliation across visual features, manufacturer specifications,
marketing claims, and real-world customer reviews using Azure Foundry GPT-5.
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

            # Reconcile multi-source evidence via GPT-5 or heuristic engine
            evidence_result = self.reasoning_client.reconcile_evidence(product, query)

            raw_conflict = evidence_result.get("has_conflict", False)
            verdict = evidence_result.get("verdict", "RECOMMENDED")
            severity = evidence_result.get("severity", "NONE").upper()
            audit_confidence = float(evidence_result.get("calibrated_confidence", 0.85))

            # Substantive conflict: discrepancy that compromises trust or safety
            is_substantive_conflict = raw_conflict and (
                severity in ["CRITICAL", "HIGH", "MEDIUM"] or verdict in ["CAUTION", "AVOID"]
            )

            # Calibrate product reliability score
            if is_substantive_conflict:
                if verdict == "AVOID" or severity in ["CRITICAL", "HIGH"]:
                    product_reliability = max(15.0, round((1.0 - (audit_confidence * 0.75)) * 100, 1))
                else:
                    product_reliability = max(35.0, round((1.0 - (audit_confidence * 0.5)) * 100, 1))
            else:
                product_reliability = round(audit_confidence * 100, 1)

            # Grounded score balances visual/textual similarity with evidence reliability
            grounded_score = round((similarity * 0.5) + (product_reliability * 0.5), 1)

            evaluated_items.append({
                "product": product,
                "similarity_score": similarity,
                "confidence_score": product_reliability,
                "audit_confidence": round(audit_confidence * 100, 1),
                "grounded_score": grounded_score,
                "has_conflict": is_substantive_conflict,
                "conflict_type": evidence_result.get("conflict_type", "None"),
                "severity": severity,
                "discrepancy_details": evidence_result.get("discrepancy_details", ""),
                "visual_summary": evidence_result.get("visual_evidence_summary", ""),
                "specs_summary": evidence_result.get("specs_evidence_summary", ""),
                "reviews_summary": evidence_result.get("reviews_evidence_summary", ""),
                "verdict": verdict,
                "engine": evidence_result.get("engine", "Azure Foundry (gpt-5)")
            })

        # Sort by grounded score (prioritizes high similarity + verified consistency)
        evaluated_items.sort(key=lambda x: x["grounded_score"], reverse=True)
        return evaluated_items
