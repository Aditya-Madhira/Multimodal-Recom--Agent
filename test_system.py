import os
from PIL import Image
from agent.recommendation_agent import MultimodalRecommendationAgent

def run_tests():
    print("=== STARTING MULTIMODAL RAG AGENT SYSTEM TESTS (GOOGLE ADK + GPT-5) ===")
    agent = MultimodalRecommendationAgent()

    # Test 1: Text Retrieval & Conflict Audit
    print("\n[Test 1] Query: 'leather armchair'...")
    res_text = agent.process_query("leather armchair", top_k=3)
    assert len(res_text["evaluated_items"]) > 0, "No items returned for text search"
    print(f"-> Retrieved & evaluated {len(res_text['evaluated_items'])} items.")
    
    # Check that conflicting items are detected
    conflicted = [item for item in res_text["evaluated_items"] if item["has_conflict"]]
    print(f"-> Detected {len(conflicted)} conflicting items in top results.")
    for item in conflicted:
        print(f"   * {item['product']['title']}: {item['conflict_type']} (Severity: {item['severity']}, Verdict: {item['verdict']})")

    # Test 2: Multimodal Image Query
    print("\n[Test 2] Query with Synthetic Reference Image + Text...")
    dummy_img = Image.new("RGB", (224, 224), color=(180, 120, 80))
    res_multi = agent.process_query("minimalist living room table", image=dummy_img, top_k=3)
    assert len(res_multi["evaluated_items"]) > 0, "No items returned for multimodal search"
    assert res_multi["has_image"] is True
    print(f"-> Multimodal search successful. Top item: {res_multi['evaluated_items'][0]['product']['title']}")

    # Test 3: Grounded Consistency Verification
    print("\n[Test 3] Verifying Grounded Consistent Product (Komorebi Teak Table)...")
    res_table = agent.process_query("solid teakwood table", top_k=3)
    teak_items = [i for i in res_table["evaluated_items"] if i["product"]["id"] == "prod_004"]
    assert len(teak_items) > 0, "Teak table not found in retrieved items"
    teak = teak_items[0]
    assert teak["verdict"] == "RECOMMENDED", f"Expected RECOMMENDED verdict, got: {teak['verdict']}"
    assert teak["confidence_score"] >= 75.0, f"Confidence score lower than expected: {teak['confidence_score']}"
    print(f"-> Teak table verified by GPT-5! Reliability: {teak['confidence_score']}% (Verdict: {teak['verdict']})")

    # Test 4: Strict Mode Filtering
    print("\n[Test 4] Testing Strict Verification Mode (Filter Conflicts)...")
    res_filtered = agent.process_query("leather armchair", filter_conflicts=True, top_k=4)
    for item in res_filtered["evaluated_items"]:
        assert not item["has_conflict"], f"Conflicting item found in filtered results: {item['product']['title']}"
    print(f"-> Strict filter active. {len(res_filtered['evaluated_items'])} verified items returned.")

    print("\n=== ALL GOOGLE ADK + GPT-5 SYSTEM TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_tests()
