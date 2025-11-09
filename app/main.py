# app/main.py
import yaml
import json
from dotenv import load_dotenv
import os
from app.graph import build_graph
from utils.rate_limited_llm import RateLimitedChatGoogleGenerativeAI
from utils.rate_limiter import gemini_rate_limiter
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

    # Configure rate limiter from config
    rate_config = config['gemini'].get('rate_limit', {'max_calls': 8, 'time_window': 60})
    gemini_rate_limiter.configure(
        max_calls=rate_config['max_calls'],
        time_window=rate_config['time_window']
    )
    
    # Use rate-limited LLM to avoid hitting Gemini's quota limits
    llm = RateLimitedChatGoogleGenerativeAI(
        model=config['gemini']['generation_model'], 
        google_api_key=api_key, 
        temperature=0
    )
    print(f"-> Rate limiter enabled ({rate_config['max_calls']} requests per {rate_config['time_window']}s)")

    
    print("-> Initializing local embeddings model for RAG (this may take a moment)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'mps'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("-> Embeddings model loaded.")
    
    app = build_graph(llm_client=llm, embeddings_client=embeddings)

    with open('data/patient_scenarios/scenario_1.json', 'r') as f:
        patient_scenario = json.load(f)

    print(f"\nAnalyzing patient: {patient_scenario['summary']}")
    initial_state = {"patient_scenario": patient_scenario}
    
    final_state = app.invoke(initial_state)

    print("\n\n--- CMAR Workflow Complete ---")
    
    # --- PRETTY PRINT THE FINAL REPORT ---
    final_report = final_state.get('final_report', {})
    if final_report:
        print("\n\n================================================")
        print("          CMAR Final Diagnostic Report          ")
        print("================================================")
        print(f"\nPatient: {final_report.get('patient_summary')}\n")
        print("--- Ranked Differential Diagnoses (by Clinical Urgency) ---")
        for dx in final_report.get('differential_diagnoses', []):
            print(f"\n{dx['rank']}. {dx['diagnosis']}")
            print(f"   Severity: {dx['severity']}/10 | Likelihood: {dx['likelihood']}/10")
            print(f"   Justification: {dx['justification']}")
        
        print("\n\n--- Overall Assessment ---")
        print(final_report.get('overall_assessment'))
        print("\n================================================")
    else:
        print("\nNo final report was generated.")
        print("\nFinal Graph State:")
        print(json.dumps(final_state, indent=2))

if __name__ == "__main__":
    main()