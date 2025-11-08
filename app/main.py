# app/main.py
import yaml
import json
from dotenv import load_dotenv
import os
from app.graph import build_graph
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

def load_config():
    with open('config/config.yaml', 'r') as f: return yaml.safe_load(f)

def main():
    print("--- Starting CMAR Workflow ---")
    config = load_config()
    api_key = os.getenv("GOOGLE_API_KEY") or config['gemini']['api_key']
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        print("ERROR: Gemini API key not found."); return

    # 1. Initialize the LLM for generation
    llm = ChatGoogleGenerativeAI(model=config['gemini']['generation_model'], google_api_key=api_key, temperature=0)
    
    # 2. Initialize the local Embeddings model for RAG
    print("-> Initializing local embeddings model for RAG (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'mps'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("-> Embeddings model loaded.")
    
    # 3. Build the graph, passing in both clients
    app = build_graph(llm_client=llm, embeddings_client=embeddings)

    with open('data/patient_scenarios/scenario_1.json', 'r') as f:
        patient_scenario = json.load(f)

    print(f"\nAnalyzing patient: {patient_scenario['summary']}")
    initial_state = {"patient_scenario": patient_scenario}
    
    final_state = app.invoke(initial_state)

    print("\n\n--- CMAR Workflow Complete ---")
    print("\nFinal Graph State:")
    print(json.dumps(final_state, indent=2))

if __name__ == "__main__":
    main()