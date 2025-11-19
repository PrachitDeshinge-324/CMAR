# evaluate.py
import yaml
import json
from dotenv import load_dotenv
import os
from datetime import datetime
from tqdm import tqdm

# Import your CMAR application components
from app.graph import build_graph
from utils.rate_limited_llm import RateLimitedChatGoogleGenerativeAI
from utils.rate_limiter import gemini_rate_limiter
from langchain_huggingface import HuggingFaceEmbeddings

# --- Configuration ---
# --- UPDATED TO POINT TO YOUR NEW DATASET ---
BENCHMARK_FILE_PATH = "data/benchmark/disease_diagnosis_dataset_100.json"
TOP_K_ACCURACY = 10  # Changed from 5 to 10 for better coverage
RESULTS_DIR = "evaluation_results"

def load_config():
    """Loads configuration from YAML file."""
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def initialize_models_and_graph():
    """Initializes all necessary models and the CMAR graph."""
    print("--- Initializing CMAR System for Evaluation ---")
    load_dotenv()
    config = load_config()
    api_key = os.getenv("GOOGLE_API_KEY") or config['gemini']['api_key']
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        raise ValueError("Gemini API key not found.")

    rate_config = config['gemini'].get('rate_limit', {'max_calls': 8, 'time_window': 60})
    gemini_rate_limiter.configure(max_calls=rate_config['max_calls'], time_window=rate_config['time_window'])
    llm = RateLimitedChatGoogleGenerativeAI(
        model=config['gemini']['generation_model'], 
        google_api_key=api_key, 
        temperature=0,
        timeout=90,  # 90 second timeout for API calls
        max_retries=2  # Retry failed calls
    )
    print(f"-> Rate limiter enabled ({rate_config['max_calls']} reqs / {rate_config['time_window']}s)")
    
    # Get optimization config
    optimization_config = config['gemini'].get('optimization', {})
    print(f"\n-> API Call Optimization:")
    print(f"   - Critic loop: {'enabled' if optimization_config.get('enable_critic_loop', True) else 'DISABLED (saves ~3 calls/patient)'}")
    print(f"   - Batch risk assessment: {'enabled (saves ~9 calls/patient)' if optimization_config.get('batch_risk_assessment', False) else 'disabled'}")
    
    # Calculate expected API calls per patient
    base_calls = 2  # Hypothesis + Synthesizer
    if optimization_config.get('batch_risk_assessment', False):
        base_calls += 1  # Single batched risk call
    else:
        base_calls += 10  # Approximate specialty count
    
    if optimization_config.get('enable_critic_loop', True):
        max_loops = optimization_config.get('max_refinement_loops', 3)
        base_calls += max_loops  # Critic calls
    
    print(f"   - Expected API calls per patient: ~{base_calls}")
    print(f"   - Max patients per day (50 call limit): ~{50 // base_calls}")

    print("-> Initializing local embeddings model for RAG...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'mps'},
        encode_kwargs={'normalize_embeddings': True}
    )
    print("-> Embeddings model loaded.")
    app = build_graph(llm_client=llm, embeddings_client=embeddings, optimization_config=optimization_config)
    print("--- CMAR System Initialized Successfully ---")
    return app

def save_task_result(results_file, task_id, task_data, final_report, error=None):
    """Saves individual task result to a JSONL file with enhanced validation info."""
    
    # Extract validation results if available
    validation = final_report.get('ground_truth_validation') if final_report else None
    diagnoses = final_report.get('differential_diagnoses', []) if final_report else []
    
    # Get top K diagnoses for logging
    top_k_diagnoses = [
        {
            'rank': d.get('rank'),
            'diagnosis': d.get('diagnosis'),
            'general_name': d.get('general_name'),
            'matches_ground_truth': d.get('matches_ground_truth', False)
        }
        for d in diagnoses[:TOP_K_ACCURACY]
    ]
    
    result = {
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "patient_summary": task_data.get('text', ''),
        "ground_truth": task_data.get('diagnosis', ''),
        "cmar_top_k_diagnoses": top_k_diagnoses,
        "is_correct": validation.get('is_correct', False) if validation else False,
        "best_match_rank": validation.get('best_match_rank') if validation else None,
        "best_match_diagnosis": validation.get('best_match_diagnosis') if validation else None,
        "best_match_reasoning": validation.get('best_match_reasoning') if validation else None,
        "error": error
    }
    with open(results_file, 'a') as f:
        f.write(json.dumps(result) + '\n')

def save_final_summary(summary_file, stats):
    """Saves the final evaluation summary to a JSON file."""
    summary = {
        "evaluation_metadata": {
            "timestamp": datetime.now().isoformat(),
            "dataset": BENCHMARK_FILE_PATH,
            "top_k": TOP_K_ACCURACY
        },
        "results": stats
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ Final summary saved to: {summary_file}")

def main():
    """Main function to run the evaluation on the custom diagnosis dataset."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(RESULTS_DIR, f"evaluation_results_{timestamp}.jsonl")
    summary_file = os.path.join(RESULTS_DIR, f"evaluation_summary_{timestamp}.json")
    
    print(f"\n📊 Results will be saved to:\n   - Details: {results_file}\n   - Summary: {summary_file}")
    
    cmar_app = initialize_models_and_graph()

    print(f"\n--- Loading Local Dataset ({BENCHMARK_FILE_PATH}) ---")
    try:
        with open(BENCHMARK_FILE_PATH, 'r') as f:
            diagnosis_tasks = json.load(f)
    except FileNotFoundError:
        print(f"🚨 FATAL ERROR: The dataset was not found at '{BENCHMARK_FILE_PATH}'.")
        return
    except json.JSONDecodeError:
        print(f"🚨 FATAL ERROR: The file at '{BENCHMARK_FILE_PATH}' is not a valid JSON file.")
        return

    print(f"-> Found {len(diagnosis_tasks)} diagnosis tasks to evaluate.")
    
    correct_predictions = 0
    failed_tasks = 0
    
    for task_idx, task in enumerate(tqdm(diagnosis_tasks, desc="Evaluating Diagnosis Tasks"), start=1):
        # --- UPDATED KEY NAMES ---
        patient_summary = task.get('text', '')
        ground_truth = task.get('diagnosis', '').strip()
        
        if not patient_summary or not ground_truth:
            print(f"\nSkipping task {task_idx} due to missing data.")
            save_task_result(results_file, task_idx, task, {}, error="Missing 'text' or 'diagnosis' field")
            failed_tasks += 1
            continue
            
        print(f"\n--- Evaluating Task {task_idx}/{len(diagnosis_tasks)} ---")
        print(f"Patient: {patient_summary[:100]}...")
        print(f"Ground Truth Diagnosis: {ground_truth}")

        try:
            # Pass ground truth to the graph for validation in synthesizer
            initial_state = {
                "patient_scenario": {"summary": patient_summary},
                "ground_truth": ground_truth
            }
            final_state = cmar_app.invoke(initial_state)
            
            final_report = final_state.get('final_report', {})
            
            if not final_report:
                print("-> CMAR produced no final report.")
                save_task_result(results_file, task_idx, task, {}, error="No final report produced")
                failed_tasks += 1
                continue

            # Get validation results from the synthesizer (now integrated in the LLM response)
            validation = final_report.get('ground_truth_validation')
            
            if not validation:
                print("-> Warning: No validation results in final report.")
                save_task_result(results_file, task_idx, task, final_report, error="No validation performed")
                failed_tasks += 1
                continue
            
            is_correct = validation.get('is_correct', False)
            
            if is_correct:
                correct_predictions += 1
                print(f"-> ✅ CORRECT - Match at rank {validation.get('best_match_rank')}: {validation.get('best_match_diagnosis')}")
                print(f"   Reasoning: {validation.get('best_match_reasoning', 'N/A')[:100]}")
            else:
                print(f"-> ❌ INCORRECT - No match in top {TOP_K_ACCURACY}")
            
            save_task_result(results_file, task_idx, task, final_report)

        except Exception as e:
            error_msg = str(e)
            print(f"-> 🚨 ERROR processing task {task_idx}: {error_msg}")
            failed_tasks += 1
            save_task_result(results_file, task_idx, task, {}, error=error_msg)
            continue

    total_evaluated = len(diagnosis_tasks)
    if total_evaluated > 0:
        accuracy = (correct_predictions / total_evaluated) * 100
        stats = {
            "total_tasks": total_evaluated,
            "successful_evaluations": total_evaluated - failed_tasks,
            "failed_tasks": failed_tasks,
            "correct_predictions": correct_predictions,
            "top_k_accuracy": round(accuracy, 2),
            "top_k": TOP_K_ACCURACY
        }
        
        print("\n\n================================================")
        print("           Diagnosis Evaluation Results           ")
        print("================================================")
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title():<25}: {value}")
        print("================================================")
        
        save_final_summary(summary_file, stats)
        print(f"✅ Detailed results saved to: {results_file}")
    else:
        print("No tasks were evaluated.")

if __name__ == "__main__":
    main()