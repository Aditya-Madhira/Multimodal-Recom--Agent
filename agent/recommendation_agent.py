"""
Multimodal Recommendation Agent
Integrates ChromaDB + CLIP Multimodal Retrieval with Evidence Conflict Detection.
"""

from typing import Optional, Dict, Any, List
from PIL import Image
import os

from retriever.multimodal_store import MultimodalRetriever
from .conflict_detector import MultimodalConflictDetector


class MultimodalRecommendationAgent:
    def __init__(self):
        self.retriever = MultimodalRetriever()
        self.conflict_detector = MultimodalConflictDetector()

    def process_query(
        self,
        query: str,
        image: Optional[Image.Image] = None,
        filter_conflicts: bool = False,
        top_k: int = 4
    ) -> Dict[str, Any]:
        """
        Full Evidence-Grounded Multimodal RAG Pipeline:
        1. Embed multimodal query (Text + Image) using Hugging Face CLIP
        2. Retrieve candidate items from ChromaDB
        3. Cross-examine multimodal evidence across Specs, Images, Marketing, and Reviews
        4. Detect contradictions and calculate calibrated confidence scores
        5. Synthesize evidence-grounded recommendation response
        """
        has_image = image is not None
        image_meta = {}

        if has_image:
            image_meta = {
                "format": image.format or "JPEG/PNG",
                "size": f"{image.width}x{image.height}px",
                "mode": image.mode
            }

        # 1 & 2. Cross-modal retrieval from ChromaDB
        candidates = self.retriever.search(
            query_text=query,
            query_image=image,
            top_k=top_k
        )

        if not candidates:
            return {
                "response_text": "I couldn't find products matching your query. Try broadening your description or uploading another reference image.",
                "has_image": has_image,
                "image_meta": image_meta,
                "evaluated_items": [],
                "conflict_summary": {}
            }

        # 3 & 4. Conflict detection & evidence reconciliation
        evaluated_items = self.conflict_detector.evaluate_candidates(
            retrieved_items=candidates,
            query=query
        )

        if filter_conflicts:
            evaluated_items = [item for item in evaluated_items if not item["has_conflict"]]

        # Count conflict statistics
        conflict_count = sum(1 for item in evaluated_items if item["has_conflict"])
        verified_count = len(evaluated_items) - conflict_count

        # 5. Build intelligent response narrative
        response_lines = []
        if has_image and query.strip():
            response_lines.append(
                f"### 🔍 Multimodal Search & Evidence Analysis\n"
                f"I cross-referenced visual features from your uploaded image (*{image_meta.get('size')}*) "
                f"with your text constraint *\"{query.strip()}\"*."
            )
        elif has_image:
            response_lines.append(
                f"### 📷 Visual Feature Retrieval & Evidence Check\n"
                f"I extracted visual style and material features from your image (*{image_meta.get('size')}*) "
                f"and queried our product index."
            )
        else:
            response_lines.append(
                f"### 📋 Evidence-Grounded Search Results\n"
                f"I searched for *\"{query.strip()}\"* and cross-examined the evidence across manufacturer specifications, "
                f"images, and customer reviews."
            )

        response_lines.append(
            f"\n**Evidence Audit Summary:** Identified **{len(evaluated_items)} candidate products** — "
            f"✅ **{verified_count} verified consistent**, ⚠️ **{conflict_count} with detected evidence contradictions**."
        )

        if conflict_count > 0:
            conflicted_titles = [
                f"*{item['product']['title']}* ({item['conflict_type']})"
                for item in evaluated_items if item["has_conflict"]
            ]
            response_lines.append(
                f"\n> ⚠️ **Discrepancy Warning**: Before purchasing, review the conflicting evidence flags below for: "
                f"{', '.join(conflicted_titles)}."
            )

        response_text = "\n".join(response_lines)

        return {
            "response_text": response_text,
            "has_image": has_image,
            "image_meta": image_meta,
            "evaluated_items": evaluated_items,
            "conflict_summary": {
                "total": len(evaluated_items),
                "verified": verified_count,
                "conflicts": conflict_count
            }
        }
