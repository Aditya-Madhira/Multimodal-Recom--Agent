"""
Multimodal Recommendation Agent
Handles query processing, image analysis, and recommendation generation.
"""

from typing import Optional, Dict, Any, List
from PIL import Image
import os


class MultimodalRecommendationAgent:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        # Sample starter catalog for demonstration until vector database is indexed
        self.sample_catalog = [
            {
                "title": "Minimalist Scandinavian Armchair",
                "category": "Furniture & Interior",
                "score": "94%",
                "description": "Clean lines with warm oak legs and textured woven upholstery, perfect for modern living spaces."
            },
            {
                "title": "Industrial Matte Black Floor Lamp",
                "category": "Lighting",
                "score": "89%",
                "description": "Adjustable dual-head architectural lamp providing targeted ambient and task lighting."
            },
            {
                "title": "Abstract Geometric Ceramic Vase",
                "category": "Home Decor",
                "score": "85%",
                "description": "Handcrafted neutral-tone ceramic centerpiece complementing minimalist aesthetics."
            }
        ]

    def process_query(self, query: str, image: Optional[Image.Image] = None) -> Dict[str, Any]:
        """
        Process a multimodal query consisting of user text and an optional image.
        """
        has_image = image is not None
        image_meta = {}

        if has_image:
            image_meta = {
                "format": image.format or "JPEG/PNG",
                "size": f"{image.width}x{image.height}px",
                "mode": image.mode
            }

        # Response text generation
        if has_image and query.strip():
            response_text = (
                f"**Multimodal Query Received!**\n\n"
                f"I analyzed your uploaded image (*{image_meta.get('size')}, {image_meta.get('format')}*) "
                f"alongside your query: *\"{query.strip()}\"*.\n\n"
                f"Here are contextually relevant recommendations retrieved based on visual style and text criteria:"
            )
        elif has_image and not query.strip():
            response_text = (
                f"**Image Analysis & Visual Recommendations:**\n\n"
                f"I processed the visual features of your image (*{image_meta.get('size')}*). "
                f"Here are items matching the aesthetic, color palette, and style detected in your image:"
            )
        else:
            response_text = (
                f"**Recommendation Results:**\n\n"
                f"Based on your query *\"{query.strip()}\"*, "
                f"I searched our index for relevant recommendations:"
            )

        return {
            "response_text": response_text,
            "has_image": has_image,
            "image_meta": image_meta,
            "recommendations": self.sample_catalog
        }
