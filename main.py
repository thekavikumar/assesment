from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
import requests
from bs4 import BeautifulSoup
import re

# -----------------------------
# Configuration & Initialization
# -----------------------------
DATA_PATH = "data"
CSV_FILE = "shl_assessments_with_pdfs.csv"
PDFS_FOLDER = "pdfs"
EMBEDDING_MODEL = "models/embedding-001"
LLM_MODEL = "gemini-2.0-flash"

# Get Google API key from environment variables
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise Exception("Google API key not found in environment variables.")

genai.configure(api_key=GOOGLE_API_KEY)

# -----------------------------
# Data Loading & Preparation
# -----------------------------
def load_and_prepare_data():
    csv_path = os.path.join(DATA_PATH, CSV_FILE)
    try:
        df = pd.read_csv(csv_path)
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
        raise HTTPException(status_code=500, detail=f"Failed to load CSV data: {str(e)}")
    
    # Process PDF brochures
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
                    print(f"Error processing {pdf_file}: {str(e)}")
    
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
    
    try:
        embeddings = GeminiEmbeddings(
            model=EMBEDDING_MODEL,
            google_api_key=GOOGLE_API_KEY
        )
        vectorstore = FAISS.from_texts(documents, embeddings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create vector store: {str(e)}")
    
    return df, vectorstore

# -----------------------------
# Optional: Fetch Job Description from URL
# -----------------------------
def fetch_jd_from_url(url: str) -> str:
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        res = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(res.text, 'html.parser')
        paragraphs = soup.find_all('span')
        text = "\n".join(p.get_text() for p in paragraphs if len(p.get_text()) > 50)
        return text.strip()[:3000]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch JD from URL: {str(e)}")

# -----------------------------
# Prompt Template for Recommendations
# -----------------------------
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

# -----------------------------
# FastAPI Request & Response Models
# -----------------------------
class RecommendationRequest(BaseModel):
    query: str
    job_url: str = None  # Optional field

class RecommendationResponse(BaseModel):
    recommendations: str  # Markdown table as a string
    matching_assessments: list = None

# -----------------------------
# FastAPI Application & Endpoint
# -----------------------------
app = FastAPI(title="SHL Assessment Recommendation API")

# Load data on startup (global variables)
df_global, vectorstore_global = load_and_prepare_data()

@app.post("/recommend", response_model=RecommendationResponse)
def recommend_assessments(request: RecommendationRequest):
    # Use query text or fetch from job_url if provided
    query_text = request.query
    if request.job_url:
        query_text = fetch_jd_from_url(request.job_url)
    if not query_text:
        raise HTTPException(status_code=400, detail="Query text is empty.")

    # Retrieve relevant document chunks from the vector store
    retriever = vectorstore_global.as_retriever(search_kwargs={"k": 15, "score_threshold": 0.7})
    docs = retriever.get_relevant_documents(query_text)
    context = "\n\n".join([doc.page_content for doc in docs])

    # Generate recommendations using GeminiChat
    try:
        llm = GeminiChat(
            model=LLM_MODEL,
            temperature=0.2,
            google_api_key=GOOGLE_API_KEY
        )
        chain = (
            {"context": lambda _: context, "query": RunnablePassthrough()}
            | ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            | llm
            | StrOutputParser()
        )
        response = chain.invoke(query_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

    # Optionally extract matching assessments from the database
    pattern = r"\[(.*?)\]\((.*?)\)"
    matches = re.findall(pattern, response)
    recommended_names = [match[0] for match in matches]
    matching_df = df_global[df_global['name'].isin(recommended_names)][['name', 'test_type', 'assessment_length']]
    matching_list = matching_df.to_dict(orient='records') if not matching_df.empty else []

    return RecommendationResponse(recommendations=response, matching_assessments=matching_list)
