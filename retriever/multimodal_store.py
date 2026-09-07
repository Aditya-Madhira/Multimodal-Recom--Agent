"""
Multimodal Vector Store using ChromaDB and Hugging Face CLIP / SigLIP embeddings.
Enables cross-modal semantic search across text descriptions and product images.
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from PIL import Image
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from data.catalog import PRODUCTS_DATA, get_product_by_id


class MultimodalRetriever:
    def __init__(self, persist_dir: str = "./chroma_db", model_name: str = "clip-ViT-B-32"):
        self.persist_dir = persist_dir
        self.model_name = model_name
        
        # Initialize Hugging Face CLIP embedding model
        self.model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB persistent client
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        self.collection_name = "multimodal_ecommerce_catalog"
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Index products if collection is empty
        if self.collection.count() == 0:
            self.index_catalog(PRODUCTS_DATA)

    def _prepare_product_text(self, product: Dict[str, Any]) -> str:
        """Create rich textual representation of multimodal product evidence."""
        specs_str = ", ".join([f"{k}: {v}" for k, v in product.get("specs", {}).items()])
        visual_str = ", ".join([f"{k}: {v}" for k, v in product.get("visual_features", {}).items()])
        
        return (
            f"Title: {product['title']}. "
            f"Category: {product['category']}. "
            f"Brand: {product['brand']}. "
            f"Visual Style: {visual_str}. "
            f"Specifications: {specs_str}. "
            f"Description: {product['manufacturer_description']}"
        )

    def index_catalog(self, products: List[Dict[str, Any]]):
        """Embed and index all products in ChromaDB using CLIP."""
        ids = []
        documents = []
        embeddings = []
        metadatas = []

        for p in products:
            doc_text = self._prepare_product_text(p)
            emb = self.model.encode(doc_text, normalize_embeddings=True).tolist()
            
            ids.append(p["id"])
            documents.append(doc_text)
            embeddings.append(emb)
            metadatas.append({
                "id": p["id"],
                "title": p["title"],
                "category": p["category"],
                "price": float(p["price"]),
                "brand": p["brand"],
                "has_conflict": p.get("ground_truth_conflict", {}).get("has_conflict", False),
                "severity": p.get("ground_truth_conflict", {}).get("severity", "NONE")
            })

        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def search(
        self,
        query_text: Optional[str] = None,
        query_image: Optional[Image.Image] = None,
        top_k: int = 4
    ) -> List[Dict[str, Any]]:
        """
        Retrieve products using cross-modal query embedding:
        - Text only -> CLIP text embedding
        - Image only -> CLIP visual embedding
        - Both -> Fused multimodal normalized embedding
        """
        if not query_text and query_image is None:
            return []

        text_emb = None
        img_emb = None

        if query_text and query_text.strip():
            text_emb = self.model.encode(query_text.strip(), normalize_embeddings=True)

        if query_image is not None:
            if query_image.mode != "RGB":
                query_image = query_image.convert("RGB")
            img_emb = self.model.encode(query_image, normalize_embeddings=True)

        if text_emb is not None and img_emb is not None:
            fused_emb = 0.5 * text_emb + 0.5 * img_emb
            norm = np.linalg.norm(fused_emb)
            query_embedding = (fused_emb / norm).tolist() if norm > 0 else text_emb.tolist()
        elif img_emb is not None:
            query_embedding = img_emb.tolist()
        else:
            query_embedding = text_emb.tolist()

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count())
        )

        matched_products = []
        if results and results["ids"] and len(results["ids"][0]) > 0:
            for idx, prod_id in enumerate(results["ids"][0]):
                full_product = get_product_by_id(prod_id)
                if full_product:
                    distance = results["distances"][0][idx] if "distances" in results and results["distances"] else 0.0
                    similarity = max(0.0, min(1.0, 1.0 - float(distance)))
                    matched_products.append({
                        "product": full_product,
                        "similarity_score": round(similarity * 100, 1)
                    })

        return matched_products
