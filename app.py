"""
Clinical RAG System - Streamlit Cloud Version
Optimized for Streamlit Community Cloud Deployment
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from typing import List
import os

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="Clinical RAG System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .answer-box {
        background-color: #e8f4f8;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .metadata-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# LOAD MODELS (with caching)
# =============================================================================

@st.cache_resource(show_spinner=True)
def load_embeddings():
    """Load embedding model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource(show_spinner=True)
def load_vectorstore(_embeddings):
    """Load FAISS vector store"""
    return FAISS.load_local(
        "vectorstore", 
        _embeddings,
        allow_dangerous_deserialization=True
    )

@st.cache_resource(show_spinner=True)
def load_llm():
    """Load language model"""
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    )
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
        do_sample=True
    )
    
    return HuggingFacePipeline(pipeline=pipe)

# =============================================================================
# CUSTOM RETRIEVER
# =============================================================================

class WorkingRetriever(BaseRetriever):
    """Custom retriever"""
    vectorstore: object
    k: int = 5

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=self.k)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def clean_llm_response(raw_answer):
    """Clean LLM response"""
    answer = raw_answer.strip()
    remove_phrases = ["You are a clinical assistant", "Use ONLY the context below to answer",
                      "CONTEXT:", "QUESTION:", "ANSWER:", "Direct Answer:", "NOTE_ID:"]
    for phrase in remove_phrases:
        if phrase in answer:
            parts = answer.split(phrase)
            if len(parts) > 1:
                answer = parts[-1].strip()
    
    lines = answer.split('\n')
    clean_lines = [line for line in lines if not line.strip().startswith('NOTE_ID:')]
    return '\n'.join(clean_lines).strip()

def format_source_document(doc, index):
    """Format source document"""
    note_id = doc.metadata.get('note_id', 'N/A')
    doc_type = doc.metadata.get('type', 'unknown')
    diagnosis = doc.metadata.get('diagnosis', 'N/A')
    content_preview = doc.page_content[:300].replace('\n', ' ')
    
    return f"""
**Source {index}**
- **Type:** {doc_type}
- **Note ID:** {note_id}
- **Diagnosis:** {diagnosis}

**Preview:** {content_preview}...
"""

# =============================================================================
# INITIALIZE SYSTEM
# =============================================================================

def initialize_system():
    """Initialize all components"""
    if 'initialized' not in st.session_state:
        with st.spinner("🔄 Loading RAG system..."):
            st.session_state.embeddings = load_embeddings()
            st.session_state.vectorstore = load_vectorstore(st.session_state.embeddings)
            st.session_state.llm = load_llm()
            st.session_state.retriever = WorkingRetriever(
                vectorstore=st.session_state.vectorstore, k=5
            )
            
            PROMPT = """You are a clinical assistant. Use ONLY the context below to answer.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
            
            prompt = PromptTemplate(template=PROMPT, input_variables=["context", "question"])
            
            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                chain_type="stuff",
                retriever=st.session_state.retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )
            
            st.session_state.initialized = True

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    st.markdown('<h1 class="main-header">🏥 Clinical RAG System</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Medical Note Analysis & Knowledge Retrieval</p>', unsafe_allow_html=True)
    
    initialize_system()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        num_sources = st.slider("Number of Source Documents", 1, 10, 5)
        st.session_state.retriever.k = num_sources
        
        st.divider()
        st.header("💡 Example Queries")
        st.markdown("""
        - What is the chief complaint for patient 18427803-DS-5?
        - What are the features of migraine with aura?
        - Key symptoms of stroke?
        """)
        st.divider()
        st.markdown(f"**Vectors:** {st.session_state.vectorstore.index.ntotal:,}")
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["💬 Query System", "👤 Patient Lookup", "📊 Info"])
    
    # TAB 1: General Query
    with tab1:
        st.header("Ask a Clinical Question")
        query = st.text_area("Your Query:", height=100, 
                            placeholder="E.g., What is the chief complaint for patient 18427803-DS-5?")
        
        col1, col2 = st.columns([1, 5])
        with col1:
            search_button = st.button("🔍 Search", type="primary", use_container_width=True)
        with col2:
            clear_button = st.button("🗑️ Clear", use_container_width=True)
        
        if clear_button:
            st.rerun()
        
        if search_button and query:
            with st.spinner("🔄 Processing..."):
                try:
                    result = st.session_state.qa_chain.invoke({"query": query})
                    clean_answer = clean_llm_response(result["result"])
                    
                    st.markdown("### 📝 Answer")
                    st.markdown(f'<div class="answer-box">{clean_answer}</div>', unsafe_allow_html=True)
                    
                    sources = result["source_documents"]
                    note_ids = list(set([doc.metadata.get('note_id', 'N/A') for doc in sources]))
                    
                    st.markdown("### 📊 Query Metadata")
                    metadata_html = f"""
                    <div class="metadata-box">
                    <strong>Retrieved:</strong> {len(sources)} documents<br>
                    <strong>Note IDs:</strong> {', '.join(note_ids[:3])}{'...' if len(note_ids) > 3 else ''}
                    </div>
                    """
                    st.markdown(metadata_html, unsafe_allow_html=True)
                    
                    st.markdown("### 📚 Source Documents")
                    for i, doc in enumerate(sources, 1):
                        with st.expander(f"Source {i}: {doc.metadata.get('note_id', 'N/A')}"):
                            st.markdown(format_source_document(doc, i))
                
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        elif search_button:
            st.warning("⚠️ Please enter a query!")
    
    # TAB 2: Patient Lookup
    with tab2:
        st.header("Patient-Specific Information")
        patient_id = st.text_input("Enter Patient Note ID:", placeholder="E.g., 18427803-DS-5")
        
        if st.button("📋 Get Patient Summary", type="primary"):
            if patient_id:
                with st.spinner(f"🔄 Retrieving info for {patient_id}..."):
                    try:
                        query = f"Provide a comprehensive clinical summary for patient {patient_id}"
                        result = st.session_state.qa_chain.invoke({"query": query})
                        clean_answer = clean_llm_response(result["result"])
                        
                        st.markdown("### 📝 Patient Summary")
                        st.markdown(f'<div class="answer-box">{clean_answer}</div>', unsafe_allow_html=True)
                        
                        st.markdown("### 📚 Source Documents")
                        for i, doc in enumerate(result["source_documents"], 1):
                            with st.expander(f"Source {i}"):
                                st.markdown(format_source_document(doc, i))
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
            else:
                st.warning("⚠️ Please enter a patient ID!")
    
    # TAB 3: System Info
    with tab3:
        st.header("System Information")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Vector Store")
            st.metric("Total Vectors", f"{st.session_state.vectorstore.index.ntotal:,}")
            st.metric("Embedding Model", "all-MiniLM-L6-v2")
        
        with col2:
            st.subheader("🤖 Language Model")
            st.metric("Model", "TinyLlama-1.1B-Chat")
            st.metric("Max Tokens", "256")
        
        st.divider()
        st.info("**Data:** MIMIC-IV Clinical Notes + Knowledge Graphs")

if __name__ == "__main__":
    main()