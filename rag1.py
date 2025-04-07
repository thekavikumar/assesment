import os
import pandas as pd
import PyPDF2
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import GoogleGenerativeAIEmbeddings as GeminiEmbeddings, ChatGoogleGenerativeAI as GeminiChat
import streamlit as st
import requests
from bs4 import BeautifulSoup
import re

# Configuration
DATA_PATH = "data"
CSV_FILE = "shl_assessments_with_pdfs.csv"
PDFS_FOLDER = "pdfs"
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-2.0-flash"

# Initialize Gemini securely with st.secrets
@st.cache_resource
def init_gemini():
    try:
        google_api_key = st.secrets["google_api_key"]
    except Exception:
        st.error("❌ Google API key not found in secrets. Please add it to `.streamlit/secrets.toml`.")
        st.stop()

    genai.configure(api_key=google_api_key)
    return google_api_key

# Load and preprocess data
@st.cache_resource
def load_and_prepare_data():
    st.write("📥 Loading and preparing assessment data...")
    csv_path = os.path.join(DATA_PATH, CSV_FILE)

    try:
        df = pd.read_csv(csv_path)
        st.success(f"✅ Loaded {len(df)} assessments from CSV")
        
        df['remote'] = df['remote'].apply(lambda x: "Yes" if str(x).lower() in ['yes', 'true', '1'] else "No")
        df['adaptive'] = df['adaptive'].apply(lambda x: "Yes" if str(x).lower() in ['yes', 'true', '1'] else "No")
        df['assessment_length'] = df['assessment_length'].astype(str) + " min"
        
        df['full_description'] = df.apply(lambda row: (
            f"Assessment: {row['name']}\n"
            f"URL: {row['url']}\n"
            f"Remote Testing: {row['remote']}\n"
            f"Adaptive/IRT: {row['adaptive']}\n"
            f"Test Type: {row['test_type']}\n"
            f"Duration: {row['assessment_length']}\n"
            f"Job Levels: {row['job_levels']}\n"
            f"Languages: {row['languages']}\n"
            f"Description: {row['description']}"
        ), axis=1)
    except Exception as e:
        st.error(f"❌ Failed to load CSV data: {str(e)}")
        st.stop()

    # Process PDFs
    st.write("📄 Processing PDF brochures...")
    pdf_texts = []
    pdf_folder_path = os.path.join(DATA_PATH, PDFS_FOLDER)
    if os.path.exists(pdf_folder_path):
        for pdf_file in os.listdir(pdf_folder_path):
            if pdf_file.endswith('.pdf'):
                try:
                    with open(os.path.join(pdf_folder_path, pdf_file), 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        text = "\n".join([page.extract_text() for page in reader.pages])
                        pdf_texts.append(text)
                except Exception as e:
                    print(f"⚠️ Error processing {pdf_file}: {str(e)}")

    st.write("✂️ Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )

    documents = []
    for _, row in df.iterrows():
        documents.extend(text_splitter.split_text(row['full_description']))
    if pdf_texts:
        documents.extend(text_splitter.split_text("\n".join(pdf_texts)))

    st.write(f"🧠 Creating vector store with {len(documents)} chunks...")
    try:
        embeddings = GeminiEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=init_gemini()
        )
        vectorstore = FAISS.from_texts(documents, embeddings)
        st.success("✅ Vector store created successfully!")
    except Exception as e:
        st.error(f"❌ Failed to create vector store: {str(e)}")
        st.stop()

    return df, vectorstore

def fetch_jd_from_url(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')

        
        paragraphs = soup.find_all('span')
        text = "\n".join(p.get_text() for p in paragraphs if len(p.get_text()) > 50)

        return text.strip()[:3000]
    except Exception as e:
        st.error(f"⚠️ Failed to fetch JD from URL: {str(e)}")
        return ""

# Prompt Template
PROMPT_TEMPLATE = """
You are an expert SHL Assessment Recommender. Given the following query, recommend the top 10 most relevant SHL assessments from the provided context.

**Query:** {query}

**Context:** {context}

**Requirements:**
1. Return between 1-10 recommendations in a markdown table
2. Each recommendation MUST include these exact fields:
   - Assessment Name (as [Name](URL))
   - Remote Testing (Yes/No)
   - Adaptive/IRT (Yes/No)
   - Test Type
   - Duration (in minutes)
   - Languages (if specified in query)
   - Job Levels (if relevant)
3. Strictly filter by duration if mentioned (e.g., "under 30 minutes")
4. Filter by languages if mentioned (e.g., "Python tests")
5. Filter by job levels if mentioned (e.g., "for senior roles")
6. Prioritize exact matches from structured data

**Output Format:**

| # | Assessment Name | Remote | Adaptive | Test Type | Duration | Languages | Job Levels |
|---|-----------------|--------|----------|-----------|----------|-----------|------------|
| 1 | [Name](URL)     | Yes/No | Yes/No   | Type      | X min    | Langs     | Levels     |
"""

# Streamlit App
def main():
    st.set_page_config(page_title="SHL Assessment Recommender", page_icon="🔍")
    st.title("🔍 SHL Assessment Recommendation System")
    st.markdown("""
    Enter a job description or paste a job link to get tailored SHL assessment recommendations.

    **Examples:**
    - "Cognitive ability tests under 30 minutes"
    - "Python coding tests with remote proctoring"
    - [Paste LinkedIn/Indeed Job Link]
    """)

    df, vectorstore = load_and_prepare_data()

    input_mode = st.radio("📥 Choose input method:", ["Text Query", "Job Link (URL)"])

    if input_mode == "Text Query":
        query = st.text_area("💬 Enter your job description or query:", height=200)
    else:
        job_url = st.text_input("🔗 Paste a Job Posting URL:")
        query = fetch_jd_from_url(job_url) if job_url else ""

    if st.button("🚀 Get Recommendations") and query:
        with st.spinner("🔎 Analyzing your requirements..."):
            try:
                st.write("📚 Retrieving relevant documents...")
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": 15, "score_threshold": 0.7}
                )
                docs = retriever.get_relevant_documents(query)
                context = "\n\n".join([doc.page_content for doc in docs])

                st.write("🤖 Generating recommendations with Gemini...")
                llm = GeminiChat(
                    model=LLM_MODEL,
                    temperature=0.2,
                    google_api_key=init_gemini()
                )

                chain = (
                    {"context": lambda _: context, "query": RunnablePassthrough()}
                    | ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                    | llm
                    | StrOutputParser()
                )

                response = chain.invoke(query)
                st.markdown("### 📝 Recommended Assessments")
                st.markdown(response)

                # Optional: Show matching in database
                with st.expander("📊 View matching assessments from database"):
                    pattern = r"\[(.*?)\]\((.*?)\)"
                    matches = re.findall(pattern, response)
                    recommended_names = [match[0] for match in matches]
                    matching_df = df[df['name'].isin(recommended_names)][['name', 'test_type', 'assessment_length']]
                    
                    if not matching_df.empty:
                        st.dataframe(matching_df)
                    else:
                        st.write("No matching assessments found in the database.")
            except Exception as e:
                st.error(f"❌ Error generating recommendations: {str(e)}")

if __name__ == "__main__":
    main()
