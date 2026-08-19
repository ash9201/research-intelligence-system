"""
Streamlit web UI for Research Intelligence System
"""
import time
import requests
import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Research Intelligence System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stTabs [data-baseweb="tab-list"] button { font-size: 1.2em; }
    </style>
    """, unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"

def get_api_url():
    """Get API base URL from session state"""
    if "api_url" not in st.session_state:
        st.session_state.api_url = API_BASE_URL
    return st.session_state.api_url

def check_api_health():
    """Check if API is running"""
    try:
        response = requests.get(f"{get_api_url()}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_system_info():
    """Get system information from API"""
    try:
        response = requests.get(f"{get_api_url()}/info", timeout=5)
        return response.json()
    except Exception as e:
        st.error(f"Error getting system info: {e}")
        return None

# Header
st.title("🔬 Research Intelligence System")
st.markdown("**Production-oriented research intelligence and document reasoning platform**")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API URL configuration
    api_url = st.text_input(
        "API URL",
        value=get_api_url(),
        help="URL of the FastAPI backend"
    )
    st.session_state.api_url = api_url
    
    # Check API health
    if check_api_health():
        st.success("✅ API Connected")
    else:
        st.error("❌ API Not Connected")
        st.info("Make sure the FastAPI backend is running at the configured URL")
    
    # System info
    if st.button("🔄 Refresh Info"):
        st.session_state.system_info = None
    
    if "system_info" not in st.session_state or st.session_state.system_info is None:
        info = get_system_info()
        if info:
            st.session_state.system_info = info
    
    if "system_info" in st.session_state and st.session_state.system_info:
        info = st.session_state.system_info
        st.subheader("System Info")
        st.write(f"**LLM Provider:** {info.get('llm_provider', 'N/A')}")
        st.write(f"**LLM Model:** {info.get('llm_model', 'N/A')}")
        st.write(f"**Embedding Model:** {info.get('embedding_model', 'N/A')}")
        st.write(f"**Available Indexes:** {len(info.get('available_indexes', []))}")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 Ingest", "🔍 Retrieve", "💬 Ask", "📊 Info"])

# Tab 1: Document Ingestion
with tab1:
    st.header("📄 Document Ingestion")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose PDF or text files",
            type=["pdf", "txt", "md"],
            accept_multiple_files=True,
            help="Upload documents to index"
        )
        
        if st.button("📤 Upload & Ingest", key="ingest_btn"):
            if uploaded_files:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Ingesting: {file.name}")
                    
                    try:
                        files = {"file": file}
                        response = requests.post(
                            f"{get_api_url()}/ingest",
                            files=files,
                            timeout=30,
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success(f"✅ Ingested: {result['title']}")
                        else:
                            st.error(f"❌ Failed to ingest {file.name}: {response.text}")
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.text("Ingestion complete!")
            else:
                st.warning("Please select files to upload")
    
    with col2:
        st.subheader("Create Index")
        index_name = st.text_input(
            "Index Name",
            value="default_index",
            help="Name for the retrieval index"
        )
        
        if st.button("🔨 Create Index", key="create_index_btn"):
            with st.spinner("Creating index..."):
                try:
                    response = requests.post(
                        f"{get_api_url()}/index",
                        params={"index_name": index_name},
                        timeout=120,
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Index created successfully!")
                        st.info(f"Documents: {result['document_count']}, Chunks: {result['chunk_count']}")
                    else:
                        st.error(f"Failed to create index: {response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

# Tab 2: Retrieval
with tab2:
    st.header("🔍 Document Retrieval")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        query = st.text_input(
            "Search Query",
            placeholder="Enter your search query...",
            help="Search documents using hybrid retrieval"
        )
    
    with col2:
        top_k = st.number_input("Top K", value=5, min_value=1, max_value=20)
    
    use_reranking = st.checkbox("Use Reranking", value=True)
    
    if st.button("🔎 Search", key="retrieve_btn"):
        if query:
            with st.spinner("Retrieving documents..."):
                try:
                    response = requests.post(
                        f"{get_api_url()}/retrieve",
                        params={
                            "query": query,
                            "top_k": top_k,
                            "use_reranking": use_reranking,
                        },
                        timeout=30,
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        st.success(f"✅ Retrieved {result['total_count']} documents in {result['retrieval_time_ms']:.2f}ms")
                        
                        # Display results
                        for idx, doc in enumerate(result['results'], 1):
                            with st.expander(f"📋 Result {idx} (Score: {doc['score']:.3f})", expanded=(idx == 1)):
                                st.write(f"**Document:** {doc['doc_id']}")
                                st.write(f"**Method:** {doc['retrieval_method']}")
                                st.write(f"**Content:**")
                                st.text(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a query")

# Tab 3: Answer Generation
with tab3:
    st.header("💬 Grounded Answer Generation")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question = st.text_area(
            "Your Question",
            placeholder="Ask a question about the documents...",
            height=100,
            help="Your question will be answered using retrieved and reranked sources"
        )
    
    with col2:
        answer_top_k = st.number_input("Sources (Top K)", value=5, min_value=1, max_value=20)
    
    use_answer_reranking = st.checkbox("Use Reranking for Answer", value=True)
    
    if st.button("✍️ Generate Answer", key="answer_btn"):
        if question:
            with st.spinner("Generating answer..."):
                try:
                    response = requests.post(
                        f"{get_api_url()}/answer",
                        params={
                            "query": question,
                            "top_k": answer_top_k,
                            "use_reranking": use_answer_reranking,
                        },
                        timeout=60,
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        status = result.get("generation_status", {})
                        generation_mode = status.get("generation_mode", "unknown")
                        st.success(f"✅ Generated in {result['generation_time_ms']:.2f}ms")
                        if generation_mode == "provider":
                            st.caption(
                                f"Provider generation: {status.get('provider')} / "
                                f"configured={status.get('configured_model')} / used={status.get('used_model')} / "
                                f"provider_status={status.get('provider_status')} / "
                                f"grounding_status={status.get('grounding_status')}"
                            )
                        else:
                            st.warning(
                                "Retrieval-only evidence summary. No provider-generated synthesis was available "
                                f"({status.get('fallback_reason', 'unknown reason')})."
                            )
                            st.caption(
                                f"provider={status.get('provider')} / configured={status.get('configured_model')} / "
                                f"provider_status={status.get('provider_status')} / "
                                f"grounding_status={status.get('grounding_status')}"
                            )
                        
                        # Display answer
                        st.subheader("📝 Answer")
                        st.markdown(result['answer'])
                        
                        # Display confidence
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            reliability = result.get("reliability", {})
                            st.metric("Evidence quality", f"{reliability.get('evidence_quality', 0):.2f}")
                        with col2:
                            st.metric("Citation coverage", f"{reliability.get('citation_coverage', 0):.2f}")
                        with col3:
                            indicator = reliability.get("reliability_indicator")
                            st.metric("Reliability indicator", f"{indicator:.2f}" if indicator is not None else "N/A")
                        st.caption(reliability.get("score_type", ""))
                        
                        # Display one traceable mapping for every supplied source.
                        citations_by_chunk = {
                            citation["chunk_id"]: citation
                            for citation in result.get("citations", [])
                        }
                        st.subheader("📚 Source mapping")
                        evidence_sources = result.get("evidence_sources", [])
                        if evidence_sources:
                            for source in evidence_sources:
                                pages = source.get("pages", [])
                                citation = citations_by_chunk.get(source["chunk_id"])
                                citation_state = "cited" if citation else "not cited"
                                title = source.get("title") or source.get("doc_id") or "Untitled source"
                                with st.expander(f"[Source {source['source_index']}] {title} ({citation_state})"):
                                    st.write(f"**Section:** {source.get('section') or 'N/A'}")
                                    st.write(f"**Page(s):** {pages or source.get('page') or 'N/A'}")
                                    st.write(f"**Chunk ID:** `{source['chunk_id']}`")
                                    st.write(f"**Evidence excerpt:**")
                                    st.text(source["content"])
                        else:
                            st.info("No evidence sources were supplied")
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter a question")

# Tab 4: Information
with tab4:
    st.header("📊 System Information")
    
    if check_api_health():
        info = get_system_info()
        if info:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("LLM Configuration")
                st.write(f"Provider: **{info.get('llm_provider', 'N/A')}**")
                st.write(f"Model: **{info.get('llm_model', 'N/A')}**")
            
            with col2:
                st.subheader("Retrieval Models")
                st.write(f"Embedding: **{info.get('embedding_model', 'N/A')}**")
                st.write(f"Reranker: **{info.get('reranker_model', 'N/A')}**")
            
            st.subheader("Available Indexes")
            if info.get('available_indexes'):
                for idx in info['available_indexes']:
                    st.write(f"• {idx}")
            else:
                st.info("No indexes available. Create one using the Ingest tab.")
    else:
        st.error("API is not connected. Please ensure the backend is running.")
    
    # Footer
    st.divider()
    st.caption("Research Intelligence System v0.1.0 • For research and development purposes")
