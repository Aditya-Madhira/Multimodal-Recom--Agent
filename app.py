import streamlit as st
from PIL import Image
import os
from dotenv import load_dotenv

from agent.recommendation_agent import MultimodalRecommendationAgent

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="Multimodal Recommendation Agent",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished UI
st.markdown("""
<style>
    .reportview-container {
        margin-top: -2em;
    }
    .recom-card {
        background-color: #1e2130;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
        border-left: 4px solid #4F46E5;
    }
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #312E81;
        color: #C7D2FE;
        margin-right: 8px;
    }
    .score-badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
        background-color: #065F46;
        color: #A7F3D0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 **Hello! I'm your Multimodal Recommendation Agent.**\n\nYou can ask for product recommendations or attach/upload an image to discover items matching a specific style, aesthetic, or visual reference.",
            "image": None,
            "recommendations": []
        }
    ]

if "agent" not in st.session_state:
    st.session_state.agent = MultimodalRecommendationAgent()

# Sidebar
with st.sidebar:
    st.title("🛍️ Multimodal Agent")
    st.caption("AI-powered visual & text recommendation engine")
    
    st.divider()
    
    st.subheader("⚙️ Agent Controls")
    
    # Recommendation Mode Selector
    mode = st.selectbox(
        "Recommendation Mode",
        options=["Product Recommendation", "Visual Aesthetic Match", "General Q&A"],
        index=0
    )
    
    # Sidebar Image Uploader (alternative to in-chat upload)
    st.subheader("📷 Visual Query (Optional)")
    sidebar_file = st.file_uploader(
        "Upload reference image:",
        type=["jpg", "jpeg", "png", "webp"],
        key="sidebar_uploader",
        help="Upload an image to guide recommendations."
    )
    
    sidebar_image = None
    if sidebar_file:
        sidebar_image = Image.open(sidebar_file)
        st.image(sidebar_image, caption="Current Active Reference", use_container_width=True)
        if st.button("❌ Remove Active Image", use_container_width=True):
            st.session_state.pop("sidebar_uploader", None)
            st.rerun()

    st.divider()
    
    # Chat Management
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "👋 Chat history cleared. How can I help you find recommendations today?",
                "image": None,
                "recommendations": []
            }
        ]
        st.rerun()
        
    st.divider()
    st.caption("Multimodal RAG Agent • Built with Streamlit")

# Main Header
st.title("🛍️ Multimodal Recommendation Chat")
st.write("Ask a question, describe what you're looking for, or attach an image to get personalized visual recommendations.")

# Render Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        # If user/assistant uploaded an image
        if msg.get("image"):
            st.image(msg["image"], caption="Reference Image", width=350)
            
        st.markdown(msg["content"])
        
        # If assistant returned structured recommendations
        if msg.get("recommendations"):
            st.markdown("### 🎯 Recommended Items")
            cols = st.columns(len(msg["recommendations"]))
            for i, item in enumerate(msg["recommendations"]):
                with cols[i % len(cols)]:
                    with st.container(border=True):
                        st.markdown(f"**{item.get('title')}**")
                        st.caption(f"🏷️ {item.get('category')} • ⭐ {item.get('score')} match")
                        st.write(item.get("description"))

# Chat Input with File Attachment Support
chat_submission = st.chat_input(
    "Ask for recommendations or attach an image...",
    accept_file=True,
    file_type=["jpg", "jpeg", "png", "webp"]
)

if chat_submission:
    user_text = chat_submission.text if hasattr(chat_submission, "text") else str(chat_submission)
    attached_files = getattr(chat_submission, "files", [])
    
    # Determine the query image (either from chat attachment or sidebar uploader)
    query_image = None
    if attached_files and len(attached_files) > 0:
        query_image = Image.open(attached_files[0])
    elif sidebar_image is not None:
        query_image = sidebar_image

    # Prevent empty submissions
    if not user_text.strip() and query_image is None:
        st.warning("Please type a message or provide an image.")
        st.stop()

    # Append User Message
    st.session_state.messages.append({
        "role": "user",
        "content": user_text if user_text.strip() else "(Attached visual reference image)",
        "image": query_image,
        "recommendations": []
    })

    # Display User Message immediately
    with st.chat_message("user"):
        if query_image:
            st.image(query_image, caption="Reference Image", width=350)
        if user_text.strip():
            st.markdown(user_text)

    # Generate Assistant Response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing query and visual features..."):
            agent_result = st.session_state.agent.process_query(
                query=user_text,
                image=query_image
            )
            
            response_text = agent_result.get("response_text", "")
            recommendations = agent_result.get("recommendations", [])
            
            st.markdown(response_text)
            
            if recommendations:
                st.markdown("### 🎯 Recommended Items")
                cols = st.columns(len(recommendations))
                for i, item in enumerate(recommendations):
                    with cols[i % len(cols)]:
                        with st.container(border=True):
                            st.markdown(f"**{item.get('title')}**")
                            st.caption(f"🏷️ {item.get('category')} • ⭐ {item.get('score')} match")
                            st.write(item.get("description"))

            # Append to session state
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "image": None,
                "recommendations": recommendations
            })
