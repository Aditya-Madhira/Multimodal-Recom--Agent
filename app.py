import streamlit as st
from PIL import Image
import os
from dotenv import load_dotenv

from agent.recommendation_agent import MultimodalRecommendationAgent
from agent.azure_llm import AzureOpenAIClient

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Multimodal Evidence-Grounded Recommender",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for research-grade styling
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
    .evidence-box {
        background-color: #1E293B;
        border-left: 4px solid #3B82F6;
        padding: 12px 16px;
        border-radius: 4px;
        margin-top: 8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing Multimodal CLIP & ChromaDB Index...")
def get_agent():
    return MultimodalRecommendationAgent()


agent = get_agent()
azure_client = AzureOpenAIClient()

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 **Welcome to the Evidence-Grounded Multimodal Recommendation Agent!**\n\n"
                "Unlike conventional recommendation systems that blindly summarize marketing claims, "
                "this system **cross-examines multimodal evidence** across:\n"
                "- 📸 **Visual features & product imagery**\n"
                "- 📋 **Technical specification tables**\n"
                "- 📣 **Manufacturer marketing claims**\n"
                "- 💬 **Real-world customer experience & reviews**\n\n"
                "Try searching for *\"leather armchair\"*, *\"ergonomic office chair\"*, or *\"brass pendant light\"*, "
                "or upload a reference photo to observe how it flags contradictions like PU faux-leather, "
                "weight limit discrepancies, or plastic finishes!"
            ),
            "image": None,
            "evaluated_items": []
        }
    ]

# Sidebar
with st.sidebar:
    st.title("🛍️ Agent Controls")
    st.caption("Multimodal RAG with Evidence Conflict Detection")
    
    st.divider()
    
    st.subheader("🛡️ Verification Settings")
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
    
    # Azure OpenAI Connection Status
    st.subheader("🤖 Reasoning Engine")
    if azure_client.is_configured():
        st.success("🟢 Azure OpenAI: Active")
        st.caption(f"Deployment: `{azure_client.deployment}`")
    else:
        st.info("ℹ️ Engine: Calibrated Heuristic Engine")
        st.caption("Azure OpenAI credentials not yet detected. Using deterministic evidence cross-examination engine.")

    with st.expander("🔑 Configure Azure OpenAI"):
        st.write("Enter credentials or define them in `.env`:")
        az_endpoint = st.text_input("Endpoint", value=os.getenv("AZURE_OPENAI_ENDPOINT", ""), type="password")
        az_key = st.text_input("API Key", value=os.getenv("AZURE_OPENAI_API_KEY", ""), type="password")
        az_deploy = st.text_input("Deployment Name", value=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o"))
        if st.button("Save Credentials"):
            os.environ["AZURE_OPENAI_ENDPOINT"] = az_endpoint
            os.environ["AZURE_OPENAI_API_KEY"] = az_key
            os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"] = az_deploy
            st.success("Settings saved! Rerun query to use Azure OpenAI.")
            st.rerun()

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
    '<div class="sub-title">Reconciling conflicting evidence across images, specifications, marketing copy, and reviews</div>',
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
            st.markdown("#### 📦 Product Recommendations & Evidence Audits")

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
                        # Header with Title and Verification Badge
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

                        # Scores Row
                        score_cols = st.columns(3)
                        with score_cols[0]:
                            st.metric("Visual/Text Match", f"{item['similarity_score']}%")
                        with score_cols[1]:
                            st.metric("Evidence Confidence", f"{item['confidence_score']}%")
                        with score_cols[2]:
                            st.metric("Grounded Score", f"{item['grounded_score']}%")

                        # Conflict Callout
                        if has_conflict:
                            st.warning(f"**Discrepancy Flagged ({item['conflict_type']}):** {item['discrepancy_details']}")
                        else:
                            st.success("**Evidence Grounded:** All modalities (specs, reviews, visual features) are verified consistent.")

                        # Expandable Evidence Cross-Examination Drawer
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

# Chat Input Widget
chat_submission = st.chat_input(
    "Ask for a recommendation or attach a reference image...",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "webp"]
)

if chat_submission:
    user_text = chat_submission.text if hasattr(chat_submission, "text") else str(chat_submission)
    attached_files = getattr(chat_submission, "files", [])

    # Determine query image
    query_image = None
    if attached_files and len(attached_files) > 0:
        query_image = Image.open(attached_files[0])
    elif sidebar_image is not None:
        query_image = sidebar_image

    if not user_text.strip() and query_image is None:
        st.warning("Please type a question or attach an image.")
        st.stop()

    # Append User Message
    st.session_state.messages.append({
        "role": "user",
        "content": user_text if user_text.strip() else "(Attached visual reference image)",
        "image": query_image,
        "evaluated_items": []
    })

    # Render User Message immediately
    with st.chat_message("user"):
        if query_image:
            st.image(query_image, caption="Query Reference Image", width=320)
        if user_text.strip():
            st.markdown(user_text)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Embedding query, retrieving from ChromaDB, & cross-examining evidence..."):
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
                st.markdown("#### 📦 Product Recommendations & Evidence Audits")

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
                                st.metric("Evidence Confidence", f"{item['confidence_score']}%")
                            with score_cols[2]:
                                st.metric("Grounded Score", f"{item['grounded_score']}%")

                            if has_conflict:
                                st.warning(f"**Discrepancy Flagged ({item['conflict_type']}):** {item['discrepancy_details']}")
                            else:
                                st.success("**Evidence Grounded:** All modalities (specs, reviews, visual features) are verified consistent.")

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

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "image": None,
                "evaluated_items": evaluated_items
            })
