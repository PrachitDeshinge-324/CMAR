# evaluate_medqa.py
import yaml
import json
import os
import time
from datetime import datetime
from tqdm import tqdm
from difflib import SequenceMatcher

# Import CMAR components
from dotenv import load_dotenv
from app.graph import build_graph
from utils.rate_limited_llm import RateLimitedChatGoogleGenerativeAI
from utils.rate_limiter import gemini_rate_limiter
from langchain_huggingface import HuggingFaceEmbeddings

# --- CONFIGURATION ---
BENCHMARK_FILE = "data/benchmark/medqa_usmle_100.json"
RESULTS_DIR = "evaluation_results"
TOP_K_CHECK = 3  # Diagnosis must be in top 3 to count as correct

# Rate Limit Buffer: Pause between cases to let the API "cool down"
# 15 RPM = 1 request every 4 seconds. 
# A case takes ~4 calls. That's 16 seconds of "quota".
# We add a buffer to be safe.
INTER_CASE_DELAY = 5 

def load_config():
    with open('config/config.yaml', 'r') as f: return yaml.safe_load(f)

def calculate_similarity(a, b):
    """Returns similarity ratio (0.0 - 1.0) between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def check_correctness(ground_truth, diagnoses):
    """
    Scientifically validates if ground truth is present in the differential.
    Returns: (is_correct, best_match_name, score, rank)
    """
    best_score = 0.0
    best_match = "None"
    best_rank = -1
    
    # Only check the top K ranked diagnoses
    for i, dx in enumerate(diagnoses[:TOP_K_CHECK]):
        prediction = dx['diagnosis']
        
        # 1. Direct Similarity Score
        score = calculate_similarity(ground_truth, prediction)
        
        # 2. Substring Bonus (e.g. "Acute Bronchitis" contains "Bronchitis")
        if ground_truth.lower() in prediction.lower() or prediction.lower() in ground_truth.lower():
            score = max(score, 0.85)
            
        if score > best_score:
            best_score = score
            best_match = prediction
            best_rank = i + 1

    # Threshold for correctness (0.6 is usually a good fuzzy match threshold)
    is_correct = best_score >= 0.6
    return is_correct, best_match, best_score, best_rank

def process_single_case(case_data, app_instance, case_id):
    """Runs CMAR on a single case."""
    patient_summary = case_data.get('text')
    ground_truth = case_data.get('diagnosis')
    
    try:
        # Run CMAR
        # Note: We do NOT pass ground_truth to the agent to prevent cheating
        result = app_instance.invoke({"patient_scenario": {"summary": patient_summary}})
        final_report = result.get('final_report', {})
        diagnoses = final_report.get('differential_diagnoses', [])
        
        # Validate Logic (Deterministic Python)
        is_correct, match_name, score, rank = check_correctness(ground_truth, diagnoses)
        
        return {
            "status": "success",
            "case_id": case_id,
            "summary": patient_summary[:100] + "...",
            "ground_truth": ground_truth,
            "cmar_top_diagnosis": diagnoses[0]['diagnosis'] if diagnoses else "None",
            "best_match": match_name,
            "match_score": score,
            "match_rank": rank,
            "is_correct": is_correct,
            "full_diagnoses": [d['diagnosis'] for d in diagnoses[:5]]
        }
        
    except Exception as e:
        return {
            "status": "error", 
            "case_id": case_id, 
            "error": str(e)
        }

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"medqa_results_{timestamp}.jsonl")
    
    print(f"--- Initializing CMAR Evaluation (MedQA - Sequential) ---")
    load_dotenv()
    config = load_config()
    api_key = os.getenv("GOOGLE_API_KEY") or config['gemini']['api_key']
    
    # Configure Rate Limiter: Conservative 15 RPM
    # This handles the internal calls within the graph
    gemini_rate_limiter.configure(max_calls=14, time_window=60)
    
    # Initialize shared models
    llm = RateLimitedChatGoogleGenerativeAI(
        model=config['gemini']['generation_model'], 
        google_api_key=api_key, 
        temperature=0
    )
    
    print("-> Loading Embeddings (NeuML/pubmedbert-base-embeddings)...")
    embeddings = HuggingFaceEmbeddings(
        model_name="NeuML/pubmedbert-base-embeddings", 
        model_kwargs={'device': 'mps'}, # 'mps' for Mac, 'cpu' or 'cuda' otherwise
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Build Graph
    app = build_graph(llm_client=llm, embeddings_client=embeddings)
    
    # Load Data
    try:
        with open(BENCHMARK_FILE, 'r') as f:
            cases = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: File '{BENCHMARK_FILE}' not found. Run get_medqa_data.py first!")
        return

    print(f"-> Starting SEQUENTIAL evaluation of {len(cases)} cases...")
    print(f"-> Rate Limit Protection: {INTER_CASE_DELAY}s delay between cases.")
    
    results = []
    correct_count = 0
    
    # Sequential Loop with TQDM progress bar
    for i, case in enumerate(tqdm(cases, desc="Evaluating")):
        
        # Process the case
        res = process_single_case(case, app, i)
        results.append(res)
        
        # Save result immediately (Recovery check)
        with open(output_file, 'a') as f:
            f.write(json.dumps(res) + "\n")
        
        # Console feedback for debugging
        if res['status'] == 'success':
            symbol = "✅" if res['is_correct'] else "❌"
            tqdm.write(f"  Case {i+1}: {symbol} GT: '{res['ground_truth']}' | Pred: '{res['best_match']}' (Score: {res['match_score']:.2f})")
            if res['is_correct']:
                correct_count += 1
        else:
            tqdm.write(f"  Case {i+1}: 🚨 Error: {res.get('error')}")
            
        # Force delay to respect strict RPM limits
        time.sleep(INTER_CASE_DELAY)

    # Final Stats
    accuracy = (correct_count / len(cases)) * 100
    print(f"\n\n========================================")
    print(f"FINAL RESULTS (MedQA USMLE)")
    print(f"========================================")
    print(f"Total Cases: {len(cases)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy (Top-{TOP_K_CHECK}): {accuracy:.1f}%")
    print(f"Detailed logs saved to: {output_file}")
    print(f"========================================")

if __name__ == "__main__":
    main()