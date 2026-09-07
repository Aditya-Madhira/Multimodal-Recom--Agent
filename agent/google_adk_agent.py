"""
Google ADK (Agent Development Kit) Agent Implementation
The agent is orchestrated using Google ADK (`google.adk.Agent`) with its brain
powered by Azure AI Foundry GPT-5 via a custom BaseLlm provider.
"""

import os
import json
import re
from typing import Dict, Any, List, Optional, AsyncGenerator
from dotenv import load_dotenv

import google.adk as adk
from google.adk.models.base_llm import BaseLlm
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.genai import types

from data.catalog import PRODUCTS_DATA, get_product_by_id

load_dotenv()


def extract_json(text: str) -> Dict[str, Any]:
    """Robustly extract and parse JSON object from model output."""
    if not text or not text.strip():
        raise ValueError("Empty response text from model")
    
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r'(\{[\s\S]*\})', cleaned)
        if match:
            return json.loads(match.group(1))
        raise


class GPT5FoundryLlm(BaseLlm):
    """
    Custom Google ADK BaseLlm model adapter connecting Google ADK to
    Azure AI Foundry GPT-5 with reasoning token accommodation.
    """
    model: str = "gpt-5"

    def _get_client(self):
        from openai import OpenAI
        raw_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "https://foundrytemp.services.ai.azure.com/openai/v1").strip()
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip()

        if "/responses" in raw_endpoint:
            raw_endpoint = raw_endpoint.replace("/responses", "")
        base_url = raw_endpoint
        if not base_url.endswith("/openai/v1") and not base_url.endswith("/v1"):
            base_url = base_url.rstrip("/") + "/openai/v1"

        return OpenAI(base_url=base_url, api_key=api_key)

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Google ADK standard async generation method for model turns."""
        client = self._get_client()

        # Extract prompt from request contents
        prompt_texts = []
        if llm_request.contents:
            for content in llm_request.contents:
                for part in content.parts:
                    if hasattr(part, "text") and part.text:
                        prompt_texts.append(part.text)

        full_prompt = "\n".join(prompt_texts) or "Hello from Google ADK"

        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": full_prompt}],
            max_completion_tokens=4000
        )

        content_text = response.choices[0].message.content or ""
        yield LlmResponse(
            content=types.Content(parts=[types.Part.from_text(text=content_text)]),
            turn_complete=True,
            partial=False
        )

    def audit_product_evidence(self, product: Dict[str, Any], query: str = "") -> Dict[str, Any]:
        """
        Direct evidence cross-examination method executed by the ADK agent's GPT-5 brain.
        """
        client = self._get_client()

        system_prompt = (
            "You are the reasoning brain of an Evidence-Grounded Multimodal Recommender Agent built on Google ADK.\n"
            "Your task is to critically cross-examine heterogeneous product evidence:\n"
            "- Visual features\n"
            "- Manufacturer description / marketing claims\n"
            "- Technical specifications table\n"
            "- Real customer reviews\n\n"
            "You must detect contradictions, discrepancies, or unverified marketing claims.\n"
            "Respond strictly in valid JSON with these keys:\n"
            "{\n"
            '  "has_conflict": bool,\n'
            '  "conflict_type": string,\n'
            '  "severity": "NONE" | "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",\n'
            '  "discrepancy_details": string,\n'
            '  "visual_evidence_summary": string,\n'
            '  "specs_evidence_summary": string,\n'
            '  "reviews_evidence_summary": string,\n'
            '  "calibrated_confidence": float (between 0.0 and 1.0),\n'
            '  "verdict": "RECOMMENDED" | "CAUTION" | "AVOID"\n'
            "}"
        )

        user_content = (
            f"Product: {product.get('title')}\n"
            f"User Query: {query}\n"
            f"Visual Features: {json.dumps(product.get('visual_features', {}))}\n"
            f"Specifications: {json.dumps(product.get('specs', {}))}\n"
            f"Manufacturer Description: {product.get('manufacturer_description', '')}\n"
            f"Customer Reviews: {json.dumps(product.get('reviews', []))}\n"
        )

        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=4000
        )

        raw_content = response.choices[0].message.content
        parsed = extract_json(raw_content)
        parsed["engine"] = f"Google ADK Agent (Brain: Azure Foundry {self.model})"
        return parsed


# Define Tools for Google ADK Agent
def search_multimodal_products(query: str, top_k: int = 4) -> List[Dict[str, Any]]:
    """
    Tool: Search product catalog using multimodal vector similarity.
    Args:
        query: User text description or requirements.
        top_k: Number of products to retrieve.
    """
    from retriever.multimodal_store import MultimodalRetriever
    retriever = MultimodalRetriever()
    return retriever.search(query_text=query, top_k=top_k)


def cross_examine_product_evidence(product_id: str, query: str = "") -> Dict[str, Any]:
    """
    Tool: Cross-examine multimodal evidence (specs, reviews, visual features, marketing)
    for a given product to detect contradictions.
    Args:
        product_id: The unique ID of the product.
        query: User query context.
    """
    prod = get_product_by_id(product_id)
    if not prod:
        return {"error": f"Product {product_id} not found."}
    
    gpt5_brain = GPT5FoundryLlm(model="gpt-5")
    return gpt5_brain.audit_product_evidence(prod, query)


def create_adk_agent() -> adk.Agent:
    """
    Factory creating the configured Google ADK Agent.
    """
    gpt5_brain = GPT5FoundryLlm(model="gpt-5")
    
    agent = adk.Agent(
        name="MultimodalEvidenceAgent",
        description="Google ADK Evidence-Grounded Multimodal Recommendation Agent",
        instruction=(
            "You are a multimodal recommendation agent constructed using the Google Agent Development Kit (ADK). "
            "Your reasoning brain is GPT-5. Your goal is to reconcile conflicting multimodal evidence across "
            "product images, specifications, marketing claims, and reviews to provide trustworthy recommendations."
        ),
        model=gpt5_brain,
        tools=[search_multimodal_products, cross_examine_product_evidence]
    )
    return agent
