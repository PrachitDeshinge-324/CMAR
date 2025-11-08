# build_vector_store.py
import os
import re
import yaml
import time
import requests
from dotenv import load_dotenv

from datasets import load_dataset
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Configuration ---
VECTOR_STORE_PATH = "vector_store/faiss_pubmed_combined"
# Let's process the first 200 questions from PubMedQA to find related PMC articles.
# This keeps the build time reasonable. You can increase this number for a more comprehensive DB.
MAX_PQA_QUESTIONS_TO_PROCESS = 200
MAX_PMC_ARTICLES_PER_QUESTION = 2 # Fetch top 2 most relevant articles per question

def load_config():
    """Loads the YAML configuration file."""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

# --- PubMedQA Data Loading ---
def create_documents_from_pubmedqa(dataset):
    """Converts the PubMedQA dataset to LangChain Documents."""
    documents = []
    print(f"-> Processing {len(dataset)} entries from PubMedQA...")
    for entry in dataset:
        content = entry['long_answer']
        metadata = { "source": "PubMedQA", "question": entry['question'], "pubid": entry['pubid'] }
        documents.append(Document(page_content=content, metadata=metadata))
    print(f"-> Successfully created {len(documents)} documents from PubMedQA.")
    return documents

# --- PubMed Central (PMC) Data Fetching ---
def fetch_pmc_articles_for_pqa(pqa_dataset):
    """
    Uses questions from PubMedQA to find and fetch full-text articles
    from PubMed Central.
    """
    print("\n--- Starting PubMed Central Fetch ---")
    headers = {"User-Agent": "CMAR_LangGraph_Agent/1.0 (test@example.com)"}
    pmc_documents = []
    
    # Take a slice of the dataset based on the configured limit
    for i, entry in enumerate(pqa_dataset):
        if i >= MAX_PQA_QUESTIONS_TO_PROCESS:
            print(f"\n-> Reached limit of {MAX_PQA_QUESTIONS_TO_PROCESS} questions for PMC search.")
            break
        
        query = entry['question']
        print(f"\n({i+1}/{MAX_PQA_QUESTIONS_TO_PROCESS}) Searching PMC for: '{query[:80]}...'")
        
        # 1. Search for PMC IDs
        search_params = {"db": "pmc", "term": query, "retmax": MAX_PMC_ARTICLES_PER_QUESTION, "retmode": "json"}
        try:
            search_response = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi", params=search_params, headers=headers)
            search_response.raise_for_status()
            pmc_ids = search_response.json().get("esearchresult", {}).get("idlist", [])
            time.sleep(0.4) # Respect NCBI API rate limits

            if not pmc_ids:
                print("  -> No relevant PMC articles found.")
                continue

            # 2. Fetch full text for found IDs
            print(f"  -> Found {len(pmc_ids)} articles. Fetching full text...")
            fetch_params = {"db": "pmc", "id": ",".join(pmc_ids), "retmode": "xml", "rettype": "full"}
            fetch_response = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi", params=fetch_params, headers=headers)
            fetch_response.raise_for_status()
            time.sleep(0.4)
            
            # 3. Basic XML parsing to extract text content
            full_text = fetch_response.text
            # Use regex to find content within body tags, then strip HTML tags
            body_content = re.search(r'<body>(.*?)</body>', full_text, re.DOTALL)
            if body_content:
                clean_text = re.sub('<[^<]+?>', '', body_content.group(1)).strip()
                # Create one document per fetched article
                doc = Document(
                    page_content=clean_text,
                    metadata={"source": "PubMed Central", "query": query, "pmcid": pmc_ids[0]}
                )
                pmc_documents.append(doc)
                print(f"  -> Successfully processed and added article {pmc_ids[0]}.")

        except requests.RequestException as e:
            print(f"  -> Error during API call: {e}")
            time.sleep(1) # Wait longer after an error

    print(f"\n-> Successfully fetched and created {len(pmc_documents)} documents from PubMed Central.")
    return pmc_documents

def main():
    """Main function to build and save the combined vector store."""
    print("--- Starting Combined Vector Store Build Process ---")
    
    load_dotenv()
    config = load_config()
    
    api_key = os.getenv("GOOGLE_API_KEY") or config['embeddings']['api_key']
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        raise ValueError("API key not found.")
        
    embeddings = GoogleGenerativeAIEmbeddings(model=config['embeddings']['model'], google_api_key=api_key)
    print(f"-> Embeddings model '{embeddings.model}' initialized.")

    # 1. Load base PubMedQA dataset
    print("-> Loading PubMedQA dataset from Hugging Face...")
    pqa_dataset = load_dataset("pubmed_qa", "pqa_labeled", split='train')
    
    # 2. Create documents from both sources
    documents_pqa = create_documents_from_pubmedqa(pqa_dataset)
    documents_pmc = fetch_pmc_articles_for_pqa(pqa_dataset)
    
    all_documents = documents_pqa + documents_pmc
    print(f"\n--- Total documents from all sources: {len(all_documents)} ---")
    
    # 3. Split Documents into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=250)
    docs = text_splitter.split_documents(all_documents)
    print(f"-> Split all documents into {len(docs)} chunks.")

    # 4. Create and Save FAISS Vector Store
    if not docs:
        print("No documents were processed. Aborting vector store creation.")
        return
        
    print(f"-> Creating FAISS vector store. This will take a significant amount of time...")
    vector_store = FAISS.from_documents(docs, embeddings)
    
    os.makedirs(os.path.dirname(VECTOR_STORE_PATH), exist_ok=True)
    vector_store.save_local(VECTOR_STORE_PATH)
    print(f"\n--- Combined Vector Store Build Complete! ---")
    print(f"-> FAISS index saved to: {VECTOR_STORE_PATH}")

if __name__ == "__main__":
    main()