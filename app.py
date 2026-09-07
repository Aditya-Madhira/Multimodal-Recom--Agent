import streamlit as st
from PIL import Image
import os
from dotenv import load_dotenv

from agent.recommendation_agent import MultimodalRecommendationAgent
from agent.google_adk_agent import create_adk_agent

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Multimodal Evidence Recommender • Google ADK + GPT-5",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .badge-adk {
        background-color: #1E3A8A;
        color: #93C5FD;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
        margin-right: 6px;
    }
    .badge-gpt5 {
        background-color: #14532D;
        color: #86EFAC;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-verified {
        background-color: #065F46;
        color: #D1FAE5;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-caution {
        background-color: #92400E;
        color: #FEF3C7;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }
    .badge-avoid {
        background-color: #991B1B;
        color: #FEE2E2;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.82rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing Google ADK Agent & Vector Index...")
def get_recommendation_system():
    recom_agent = MultimodalRecommendationAgent()
    adk_agent = create_adk_agent()
    return recom_agent, adk_agent


agent, adk_agent = get_recommendation_system()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome! I am your Multimodal Recommendation Agent built with Google ADK.**\n\n"
                "My reasoning brain is powered by **Azure AI Foundry GPT-5** and cross-modal retrieval is indexed with **ChromaDB + CLIP**.\n\n"
                "Rather than accepting marketing claims at face value, I **audit and cross-examine multimodal evidence** across:\n"
                "- 📸 **Visual features & product imagery**\n"
                "- 📋 **Technical specification tables**\n"
                "- 📣 **Manufacturer marketing claims**\n"
                "- 💬 **Real-world customer experience & reviews**\n\n"
                "Try searching for *\"leather armchair\"*, *\"ergonomic office chair\"*, or *\"brass pendant lamp\"*, "
                "or upload an image to see how contradictions (e.g. PU plastic vs full-grain leather, weight limits) are detected!"
            ),
            "image": None,
            "evaluated_items": []
        }
    ]

# Sidebar
with st.sidebar:
    st.title("🛍️ Agent Control Panel")
    st.markdown('<span class="badge-adk">Framework: Google ADK</span><span class="badge-gpt5">Brain: GPT-5</span>', unsafe_allow_html=True)
    st.caption("Multimodal RAG with Evidence Conflict Detection")
    
    st.divider()
    
    st.subheader("🤖 Google ADK Agent Status")
    st.write(f"**Agent Name:** `{adk_agent.name}`")
    st.write(f"**LLM Model Brain:** `{adk_agent.model.model}`")
    st.write(f"**ADK Tools Registered:** {len(adk_agent.tools)}")
    with st.expander("🛠️ View ADK Tools"):
        for t in adk_agent.tools:
            st.code(f"{t.__name__}(): {t.__doc__.strip().splitlines()[0] if t.__doc__ else ''}")

    st.divider()

    st.subheader("🛡️ Evidence Verification")
    strict_verification = st.toggle(
        "Strict Verification Mode",
        value=False,
        help="Exclude products with detected evidence contradictions from recommendations."
    )
    
    top_k_candidates = st.slider(
        "Candidate Retrieval Pool (Top K)",
        min_value=2,
        max_value=6,
        value=4,
        help="Number of candidates retrieved from ChromaDB for evidence cross-examination."
    )
    
    st.divider()
    
    # Visual Reference Upload (Sidebar)
    st.subheader("📷 Visual Query (Optional)")
    sidebar_file = st.file_uploader(
        "Upload reference image:",
        type=["jpg", "jpeg", "png", "webp"],
        key="sidebar_img_uploader",
        help="Upload an image to guide cross-modal style & material retrieval."
    )
    
    sidebar_image = None
    if sidebar_file:
        sidebar_image = Image.open(sidebar_file)
        st.image(sidebar_image, caption="Active Reference Image", use_container_width=True)
        if st.button("❌ Remove Reference Image", use_container_width=True):
            st.session_state.pop("sidebar_img_uploader", None)
            st.rerun()

    st.divider()
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Chat history cleared. What kind of products would you like me to inspect and recommend?",
                "image": None,
                "evaluated_items": []
            }
        ]
        st.rerun()

# Main Header
st.markdown('<div class="main-title">🛍️ Multimodal Evidence-Grounded Recommender</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Built on <b>Google ADK</b> with <b>Azure AI Foundry GPT-5</b> as the reasoning brain</div>',
    unsafe_allow_html=True
)

# Render Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg.get("image"):
            st.image(msg["image"], caption="User Visual Reference", width=320)

        st.markdown(msg["content"])

        # Render evaluated products with evidence breakdown
        if msg.get("evaluated_items"):
            st.markdown("---")
            st.markdown("#### 📦 Product Recommendations & Evidence Audits (Google ADK + GPT-5)")

            for item in msg["evaluated_items"]:
                prod = item["product"]
                has_conflict = item["has_conflict"]
                severity = item["severity"]
                verdict = item["verdict"]

                with st.container(border=True):
                    col1, col2 = st.columns([1, 2.5])

                    with col1:
                        if prod.get("image_url"):
                            st.image(prod["image_url"], caption=prod["title"], use_container_width=True)
                        else:
                            st.write("*(No product image)*")

                    with col2:
                        header_cols = st.columns([3, 1.5])
                        with header_cols[0]:
                            st.subheader(prod["title"])
                            st.caption(f"🏷️ **Brand:** {prod['brand']} | **Category:** {prod['category']} | **Price:** ₹{prod['price']:,}")
                        with header_cols[1]:
                            if verdict == "RECOMMENDED":
                                st.markdown('<span class="badge-verified">✅ VERIFIED CONSISTENT</span>', unsafe_allow_html=True)
                            elif verdict == "CAUTION":
                                st.markdown('<span class="badge-caution">⚠️ EVIDENCE CONFLICT</span>', unsafe_allow_html=True)
                            else:
                                st.markdown('<span class="badge-avoid">🔴 AVOID - CRITICAL MISMATCH</span>', unsafe_allow_html=True)

                        score_cols = st.columns(3)
                        with score_cols[0]:
                            st.metric("Visual/Text Match", f"{item['similarity_score']}%")
                        with score_cols[1]:
                            st.metric("Product Reliability", f"{item['confidence_score']}%")
                        with score_cols[2]:
                            st.metric("Grounded Score", f"{item['grounded_score']}%")

                        if has_conflict:
                            st.warning(f"**Discrepancy Flagged by GPT-5 ({item['conflict_type']}):** {item['discrepancy_details']}")
                        else:
                            st.success("**Evidence Grounded:** All modalities (specs, reviews, visual features) verified consistent by GPT-5.")

                        with st.expander("🔍 Cross-Examine Multimodal Evidence"):
                            ev_cols = st.columns(3)
                            with ev_cols[0]:
                                st.markdown("**📸 Visual Cues**")
                                st.write(item.get("visual_summary", "Standard appearance"))
                            with ev_cols[1]:
                                st.markdown("**📋 Spec Table**")
                                st.write(item.get("specs_summary", "Standard specs"))
                            with ev_cols[2]:
                                st.markdown("**💬 Customer Experience**")
                                st.write(item.get("reviews_summary", "Customer feedback"))

                            st.markdown(f"**Manufacturer Claim:** *\"{prod.get('manufacturer_description', '')}\"*")
                            st.caption(f"Reasoning Engine: {item.get('engine', 'Google ADK (Brain: GPT-5)')}")

# Chat Input Widget
chat_submission = st.chat_input(
    "Ask for a recommendation or attach a reference image...",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "webp"]
)

if chat_submission:
    user_text = chat_submission.text if hasattr(chat_submission, "text") else str(chat_submission)
    attached_files = getattr(chat_submission, "files", [])

    query_image = None
    if attached_files and len(attached_files) > 0:
        query_image = Image.open(attached_files[0])
    elif sidebar_image is not None:
        query_image = sidebar_image

    if not user_text.strip() and query_image is None:
        st.warning("Please type a question or attach an image.")
        st.stop()

    st.session_state.messages.append({
        "role": "user",
        "content": user_text if user_text.strip() else "(Attached visual reference image)",
        "image": query_image,
        "evaluated_items": []
    })

    with st.chat_message("user"):
        if query_image:
            st.image(query_image, caption="Query Reference Image", width=320)
        if user_text.strip():
            st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Google ADK Agent orchestrating retrieval & GPT-5 evidence cross-examination..."):
            agent_result = agent.process_query(
                query=user_text,
                image=query_image,
                filter_conflicts=strict_verification,
                top_k=top_k_candidates
            )

            response_text = agent_result.get("response_text", "")
            evaluated_items = agent_result.get("evaluated_items", [])

            st.markdown(response_text)

            if evaluated_items:
                st.markdown("---")
                st.markdown("#### 📦 Product Recommendations & Evidence Audits (Google ADK + GPT-5)")

                for item in evaluated_items:
                    prod = item["product"]
                    has_conflict = item["has_conflict"]
                    severity = item["severity"]
                    verdict = item["verdict"]

                    with st.container(border=True):
                        col1, col2 = st.columns([1, 2.5])

                        with col1:
                            if prod.get("image_url"):
                                st.image(prod["image_url"], caption=prod["title"], use_container_width=True)
                            else:
                                st.write("*(No product image)*")

                        with col2:
                            header_cols = st.columns([3, 1.5])
                            with header_cols[0]:
                                st.subheader(prod["title"])
                                st.caption(f"🏷️ **Brand:** {prod['brand']} | **Category:** {prod['category']} | **Price:** ₹{prod['price']:,}")
                            with header_cols[1]:
                                if verdict == "RECOMMENDED":
                                    st.markdown('<span class="badge-verified">✅ VERIFIED CONSISTENT</span>', unsafe_allow_html=True)
                                elif verdict == "CAUTION":
                                    st.markdown('<span class="badge-caution">⚠️ EVIDENCE CONFLICT</span>', unsafe_allow_html=True)
                                else:
                                    st.markdown('<span class="badge-avoid">🔴 AVOID - CRITICAL MISMATCH</span>', unsafe_allow_html=True)

                            score_cols = st.columns(3)
                            with score_cols[0]:
                                st.metric("Visual/Text Match", f"{item['similarity_score']}%")
                            with score_cols[1]:
                                st.metric("Product Reliability", f"{item['confidence_score']}%")
                            with score_cols[2]:
                                st.metric("Grounded Score", f"{item['grounded_score']}%")

                            if has_conflict:
                                st.warning(f"**Discrepancy Flagged by GPT-5 ({item['conflict_type']}):** {item['discrepancy_details']}")
                            else:
                                st.success("**Evidence Grounded:** All modalities (specs, reviews, visual features) verified consistent by GPT-5.")

                            with st.expander("🔍 Cross-Examine Multimodal Evidence"):
                                ev_cols = st.columns(3)
                                with ev_cols[0]:
                                    st.markdown("**📸 Visual Cues**")
                                    st.write(item.get("visual_summary", "Standard appearance"))
                                with ev_cols[1]:
                                    st.markdown("**📋 Spec Table**")
                                    st.write(item.get("specs_summary", "Standard specs"))
                                with ev_cols[2]:
                                    st.markdown("**💬 Customer Experience**")
                                    st.write(item.get("reviews_summary", "Customer feedback"))

                                st.markdown(f"**Manufacturer Claim:** *\"{prod.get('manufacturer_description', '')}\"*")
                                st.caption(f"Reasoning Engine: {item.get('engine', 'Google ADK (Brain: GPT-5)')}")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "image": None,
                "evaluated_items": evaluated_items
            })
