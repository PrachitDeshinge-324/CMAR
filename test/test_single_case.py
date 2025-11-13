# test_single_case.py
"""
Test a single case from the benchmark dataset with ground truth validation.
"""
import sys
import yaml
import json
from dotenv import load_dotenv
import os
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.graph import build_graph
from utils.rate_limited_llm import RateLimitedChatGoogleGenerativeAI
from utils.rate_limiter import gemini_rate_limiter
from langchain_huggingface import HuggingFaceEmbeddings

def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

def main():
    print("--- Testing Single Case with Ground Truth Validation ---\n")
    load_dotenv()
    config = load_config()
    api_key = os.getenv("GOOGLE_API_KEY") or config['gemini']['api_key']
    
    if not api_key or api_key == "YOUR_GEMINI_API_KEY":
        print("ERROR: Gemini API key not found.")
        return

    # Configure rate limiter
    rate_config = config['gemini'].get('rate_limit', {'max_calls': 8, 'time_window': 60})
    gemini_rate_limiter.configure(
        max_calls=rate_config['max_calls'],
        time_window=rate_config['time_window']
    )
    
    llm = RateLimitedChatGoogleGenerativeAI(
        model=config['gemini']['generation_model'], 
        google_api_key=api_key, 
        temperature=0
    )
    
    print("-> Initializing embeddings model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={'device': 'mps'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    app = build_graph(llm_client=llm, embeddings_client=embeddings)

    # Load first case from benchmark
    with open('data/benchmark/disease_diagnosis_dataset_100.json', 'r') as f:
        benchmark_cases = json.load(f)
    
    # Test the first case
    test_case = benchmark_cases[0]
    patient_summary = test_case['text']
    ground_truth = test_case['diagnosis']
    
    print(f"Patient Summary: {patient_summary}\n")
    print(f"Ground Truth: {ground_truth}\n")
    print("=" * 80)
    print("\nRunning CMAR Analysis...\n")
    
    initial_state = {
        "patient_scenario": {"summary": patient_summary},
        "ground_truth": ground_truth
    }
    
    final_state = app.invoke(initial_state)
    
    # Extract and display results
    final_report = final_state.get('final_report', {})
    
    if not final_report:
        print("ERROR: No final report generated!")
        return
    
    print("\n\n" + "=" * 80)
    print("                    FINAL DIAGNOSTIC REPORT")
    print("=" * 80)
    
    print(f"\nPatient: {final_report.get('patient_summary')}\n")
    
    print("--- Ranked Differential Diagnoses ---")
    for dx in final_report.get('differential_diagnoses', [])[:10]:  # Show top 10
        match_marker = "✓" if dx.get('matches_ground_truth') else " "
        print(f"\n[{match_marker}] {dx['rank']}. {dx['diagnosis']}")
        print(f"    General Name: {dx.get('general_name', 'N/A')}")
        print(f"    Severity: {dx['severity']}/10 | Likelihood: {dx['likelihood']}/10")
        print(f"    Justification: {dx['justification'][:150]}...")
    
    print("\n\n--- Ground Truth Validation ---")
    validation = final_report.get('ground_truth_validation')
    
    if validation:
        print(f"Ground Truth: {validation.get('ground_truth')}")
        print(f"Result: {'✅ CORRECT' if validation.get('is_correct') else '❌ INCORRECT'}")
        
        if validation.get('is_correct'):
            print(f"\nBest Match:")
            print(f"  Rank: {validation.get('best_match_rank')}")
            print(f"  Diagnosis: {validation.get('best_match_diagnosis')}")
            print(f"  Reasoning: {validation.get('best_match_reasoning')}")
    else:
        print("No validation results available.")
    
    print("\n" + "=" * 80)
    
    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"cmar_test_case_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump({
            "test_case": test_case,
            "final_state": final_state
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_file}")

if __name__ == "__main__":
    main()
