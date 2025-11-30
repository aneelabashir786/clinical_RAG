"""
Clinical RAG System - Streamlit Version
Run with: streamlit run app.py
"""

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import BaseRetriever, Document
from pydantic import ConfigDict
from typing import List
import re

# =============================================================================
# PAGE CONFIG
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
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
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
# LOAD MODELS
# =============================================================================

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

@st.cache_resource(show_spinner=False)
def load_vectorstore(_embeddings):
    return FAISS.load_local(
        "vectorstore",
        _embeddings,
        allow_dangerous_deserialization=True
    )

@st.cache_resource(show_spinner=False)
def load_llm():
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
    vectorstore: object
    k: int = 5

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self.vectorstore.similarity_search(query, k=self.k)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

# =============================================================================
# HELPERS
# =============================================================================

def clean_llm_response(raw_answer):
    answer = raw_answer.strip()

    remove_phrases = [
        "You are a clinical assistant", "Use ONLY the context below to answer",
        "CONTEXT:", "QUESTION:", "ANSWER:", "Direct Answer:", "NOTE_ID:"
    ]

    for phrase in remove_phrases:
        if phrase in answer:
            answer = answer.split(phrase)[-1].strip()

    if "CHIEF COMPLAINT:" in answer and len(answer) > 100:
        if answer.index("CHIEF COMPLAINT:") > 100:
            answer = answer.split("CHIEF COMPLAINT:")[0].strip()

    lines = answer.split("\n")
    lines = [line for line in lines if not line.strip().startswith("NOTE_ID:")]
    answer = "\n".join(lines).strip()

    sentences = answer.split(".")
    if len(answer) > 500 and len(sentences) > 1:
        answer = sentences[0].strip() + "."

    return answer

def format_source_document(doc, index):
    note_id = doc.metadata.get("note_id", "N/A")
    doc_type = doc.metadata.get("type", "unknown")
    diagnosis = doc.metadata.get("diagnosis", "N/A")
    chunk_idx = doc.metadata.get("chunk_index", "N/A")

    preview = doc.page_content[:300].replace("\n", " ")

    return f"""
**Source {index}**
- **Type:** {doc_type}
- **Note ID:** {note_id}
- **Diagnosis:** {diagnosis}
- **Chunk:** {chunk_idx}

**Preview:**  
{preview}...
"""

# =============================================================================
# INITIALIZE SYSTEM
# =============================================================================

def initialize_system():
    if "initialized" not in st.session_state:
        with st.spinner("🔄 Loading RAG system components..."):
            st.session_state.embeddings = load_embeddings()
            st.session_state.vectorstore = load_vectorstore(st.session_state.embeddings)
            st.session_state.llm = load_llm()

            st.session_state.retriever = WorkingRetriever(
                vectorstore=st.session_state.vectorstore,
                k=5
            )

            CLEAN_PROMPT = """You are a clinical assistant. Use ONLY the context below to answer.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""

            prompt = PromptTemplate(
                template=CLEAN_PROMPT,
                input_variables=["context", "question"]
            )

            st.session_state.qa_chain = RetrievalQA.from_chain_type(
                llm=st.session_state.llm,
                chain_type="stuff",
                retriever=st.session_state.retriever,
                chain_type_kwargs={"prompt": prompt},
                return_source_documents=True
            )

            st.session_state.initialized = True
            st.success("✅ System loaded successfully!")

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

        num_sources = st.slider(
            "Number of Source Documents",
            min_value=1,
            max_value=10,
            value=5,
            help="Number of documents to retrieve"
        )
        st.session_state.retriever.k = num_sources

        st.divider()
        st.header("📊 System Info")
        st.button("Show Statistics")

        st.divider()
        st.header("💡 Example Queries")
        st.markdown("""
        **Patient-Specific:**
        - What is the chief complaint for patient 18427803-DS-5?

        **Diagnosis:**
        - What are the features of migraine with aura?
        """)

        st.divider()
        st.markdown("**Data Source:** MIMIC-IV Clinical Notes")
        st.markdown(f"**Vectors:** {st.session_state.vectorstore.index.ntotal:,}")

    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Query System", "👤 Patient Lookup", "📊 System Stats"])

    # TAB 1: Query
    with tab1:
        st.header("Ask a Clinical Question")

        query = st.text_area("Your Query:", height=100)
        search_btn = st.button("🔍 Search", type="primary")
        clear_btn = st.button("🗑️ Clear")

        if clear_btn:
            st.rerun()

        if search_btn:
            if not query:
                st.warning("⚠️ Please enter a query first!")
            else:
                with st.spinner("🔄 Processing your query..."):
                    try:
                        result = st.session_state.qa_chain.invoke({"query": query})
                        clean_answer = clean_llm_response(result["result"])

                        st.markdown("### 📝 Answer")
                        st.markdown(f'<div class="answer-box">{clean_answer}</div>', unsafe_allow_html=True)

                        sources = result["source_documents"]

                        st.markdown("### 📚 Source Documents")
                        for i, doc in enumerate(sources, 1):
                            with st.expander(f"Source {i}: {doc.metadata.get('note_id', 'N/A')}"):
                                st.markdown(format_source_document(doc, i))

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    # TAB 2: Patient Lookup
    with tab2:
        st.header("Patient-Specific Information")

        patient_id = st.text_input("Enter Patient Note ID:")
        if st.button("📋 Get Patient Summary", type="primary"):
            if not patient_id:
                st.warning("⚠️ Enter patient ID!")
            else:
                with st.spinner(f"Retrieving info for {patient_id}..."):
                    try:
                        query = f"Provide a clinical summary for patient {patient_id}"
                        result = st.session_state.qa_chain.invoke({"query": query})
                        clean_answer = clean_llm_response(result["result"])

                        st.markdown("### Summary")
                        st.markdown(f'<div class="answer-box">{clean_answer}</div>', unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

    # TAB 3: Stats
    with tab3:
        st.header("System Statistics")
        st.metric("Total Vectors", st.session_state.vectorstore.index.ntotal)
        st.metric("Embedding Model", "all-MiniLM-L6-v2")

# =============================================================================
# RUN
# =============================================================================

if __name__ == "__main__":
    main()
