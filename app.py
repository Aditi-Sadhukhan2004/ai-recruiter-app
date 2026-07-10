import streamlit as st
import os
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="AI Recruiter Assistant", page_icon="💼", layout="wide")

st.title("💼 Public AI Resume Screener & Recruiter Assistant")
st.write("Upload both PDFs to analyze technical alignment instantly over the cloud.")

# Fetch the secret API key from the cloud environment variables
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📋 Step 1: Upload Job Criteria")
    job_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"], key="job_pdf")
    
    st.subheader("📄 Step 2: Upload Candidate Data")
    resume_file = st.file_uploader("Upload Candidate Resume (PDF)", type=["pdf"], key="resume_pdf")

def process_dual_pdf_rag(resume_path, job_path):
    job_loader = PyPDFLoader(job_path)
    job_docs = job_loader.load()
    job_description_text = "\n".join(doc.page_content for doc in job_docs)
    
    resume_loader = PyPDFLoader(resume_path)
    resume_docs = resume_loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    resume_chunks = text_splitter.split_documents(resume_docs)
    
    # Lightweight embedding model that runs perfectly on free servers
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vector_store = Chroma.from_documents(documents=resume_chunks, embedding=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # Cloud-hosted Llama3 engine
    llm = ChatGroq(model="llama3-8b-8192", groq_api_key=GROQ_API_KEY, temperature=0.1)
    
    system_prompt = (
        "You are an expert HR Recruiter.\n"
        "Evaluate the candidate's Resume Context against the provided Job Description criteria.\n\n"
        "Job Description Text:\n{job_description}\n\n"
        "Candidate Resume Context:\n{context}\n\n"
        "Provide a structured analysis covering exactly these points:\n"
        "OVERALL MATCH SCORE: [Provide a strict percentage match between 0% and 100%. Example format: 85%]\n"
        "1. KEY SKILLS MATCHED\n"
        "2. MISSING REQUISITES / GAPS\n"
        "3. 3 CUSTOM INTERVIEW QUESTIONS"
    )
    
    prompt = ChatPromptTemplate.from_template(system_prompt)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    chain = (
        {"context": retriever | format_docs, "job_description": lambda x: job_description_text}
        | prompt | llm | StrOutputParser()
    )
    
    return chain.invoke("Compare profiles.")

with col2:
    st.subheader("📊 Step 3: Screener Analysis Output")
    
    if st.button("Screen Documents 🚀", type="primary"):
        if not GROQ_API_KEY:
            st.error("Missing API Key! Please set GROQ_API_KEY in Render Environment Variables.")
        elif job_file is None or resume_file is None:
            st.warning("Please upload both PDF files.")
        else:
            with st.spinner("Analyzing alignment over cloud servers..."):
                temp_resume = "temp_resume.pdf"
                temp_job = "temp_job.pdf"
                
                with open(temp_resume, "wb") as f: f.write(resume_file.getbuffer())
                with open(temp_job, "wb") as f: f.write(job_file.getbuffer())
                
                try:
                    analysis_report = process_dual_pdf_rag(temp_resume, temp_job)
                    st.success("Analysis Complete!")
                    
                    match = re.search(r'(\d+)%', analysis_report)
                    if match:
                        st.metric(label="Calculated Alignment", value=f"🎯 {match.group(0)} Match")
                    
                    st.markdown("### 📋 Recruiter's Comparison Report")
                    st.info(analysis_report)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                finally:
                    if os.path.exists(temp_resume): os.remove(temp_resume)
                    if os.path.exists(temp_job): os.remove(temp_job)