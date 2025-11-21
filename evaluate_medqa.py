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
TOP_K_CHECK = 5  
INTER_CASE_DELAY = 5 

def load_config():
    with open('config/config.yaml', 'r') as f: return yaml.safe_load(f)

def calculate_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def check_correctness(ground_truth, diagnoses, options_list=None):
    """
    Validates if the prediction matches the Ground Truth.
    Also checks if the prediction is actually one of the valid Options.
    """
    if not diagnoses:
        return False, "None", 0.0, -1, "No Output"

    best_score = 0.0
    best_match = "None"
    best_rank = -1
    is_correct = False
    
    # 1. Check against Ground Truth
    for i, dx in enumerate(diagnoses[:TOP_K_CHECK]):
        prediction = dx['diagnosis']
        score = calculate_similarity(ground_truth, prediction)
        
        # Substring bonus (Critical for matching "Option A" to "Option A: Drug X")
        if ground_truth.lower() in prediction.lower() or prediction.lower() in ground_truth.lower():
            score = max(score, 1.0) 
            
        if score > best_score:
            best_score = score
            best_match = prediction
            best_rank = i + 1
            
        # Threshold for correctness
        if score >= 0.7 and i == 0: # Top-1 Accuracy Strictness
            is_correct = True

    # 2. Sanity Check: Did it pick a valid option?
    status_msg = "Valid"
    if options_list and best_match != "None":
        # Flatten options if dict
        valid_opts = options_list.values() if isinstance(options_list, dict) else options_list
        # Check if prediction resembles ANY valid option
        is_valid = False
        for opt in valid_opts:
            if calculate_similarity(opt, best_match) > 0.8 or opt in best_match or best_match in opt:
                is_valid = True
                break
        if not is_valid:
            status_msg = "⚠️ Hallucination (Not in options)"

    return is_correct, best_match, best_score, best_rank, status_msg

def process_single_case(case_data, app_instance, case_id):
    """Runs CMAR on a single case with Options Injection."""
    patient_summary = case_data.get('text')
    ground_truth = case_data.get('diagnosis')
    options = case_data.get('original_options', {}) 
    
    # --- CRITICAL FIX: INJECT OPTIONS INTO SUMMARY ---
    options_text = ""
    if isinstance(options, dict):
        for key, val in options.items():
            options_text += f"- {val}\n"
    elif isinstance(options, list):
        for val in options:
            options_text += f"- {val}\n"
            
    if options_text:
        full_prompt = f"{patient_summary}\n\n**CANDIDATE DIAGNOSES (OPTIONS):**\n{options_text}"
    else:
        full_prompt = patient_summary

    try:
        # Run CMAR
        result = app_instance.invoke({"patient_scenario": {"summary": full_prompt}})
        final_report = result.get('final_report', {})
        diagnoses = final_report.get('differential_diagnoses', [])
        
        # Validate
        is_correct, match_name, score, rank, status = check_correctness(ground_truth, diagnoses, options)
        
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
            "note": status
        }
        
    except Exception as e:
        return {"status": "error", "case_id": case_id, "error": str(e)}

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(RESULTS_DIR, f"medqa_results_fixed_{timestamp}.jsonl")
    
    print(f"--- Initializing CMAR Evaluation (MedQA with Options) ---")
    load_dotenv()
    config = load_config()
    api_key = os.getenv("GOOGLE_API_KEY") or config['gemini']['api_key']
    
    gemini_rate_limiter.configure(max_calls=14, time_window=60)
    
    llm = RateLimitedChatGoogleGenerativeAI(
        model=config['gemini']['generation_model'], 
        google_api_key=api_key, 
        temperature=0
    )
    
    embeddings = HuggingFaceEmbeddings(
        model_name="NeuML/pubmedbert-base-embeddings", 
        model_kwargs={'device': 'cuda'}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Build Graph in BENCHMARK MODE
    app = build_graph(llm_client=llm, embeddings_client=embeddings, benchmark_mode=True)
    
    try:
        with open(BENCHMARK_FILE, 'r') as f:
            cases = json.load(f)
    except FileNotFoundError:
        print(f"❌ Error: Run get_medqa_data.py first!")
        return

    # Run on 60 cases as per your previous run
    cases_to_run = cases[:60]
    print(f"-> Evaluating {len(cases_to_run)} cases...")
    
    correct_count = 0
    
    for i, case in enumerate(tqdm(cases_to_run, desc="Evaluating")):
        res = process_single_case(case, app, i)
        
        with open(output_file, 'a') as f:
            f.write(json.dumps(res) + "\n")
        
        if res['status'] == 'success':
            if res['is_correct']: correct_count += 1
            symbol = "✅" if res['is_correct'] else "❌"
            # Clean console output
            gt_short = res['ground_truth'][:30]
            pred_short = res['cmar_top_diagnosis'][:30]
            tqdm.write(f"  #{i+1}: {symbol} GT: '{gt_short}...' | Pred: '{pred_short}...' [{res.get('note')}]")
        else:
            tqdm.write(f"  #{i+1}: 🚨 Error: {res.get('error')}")
        
        time.sleep(INTER_CASE_DELAY)

    print(f"\n\n========================================")
    print(f"FINAL RESULTS (With Options Injection)")
    print(f"========================================")
    print(f"Total Cases: {len(cases_to_run)}")
    print(f"Correct: {correct_count}")
    print(f"Accuracy (Top-1): {(correct_count/len(cases_to_run))*100:.1f}%")
    print(f"========================================")

if __name__ == "__main__":
    main()