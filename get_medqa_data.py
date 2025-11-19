# get_medqa_data.py
import json
import random
import re
import os  # <--- Added this import
from datasets import load_dataset
from tqdm import tqdm

# Configuration
OUTPUT_FILE = "data/benchmark/medqa_usmle_100.json"
SAMPLE_SIZE = 100  # Number of cases to evaluate
SEED = 42

def clean_question_text(question):
    """
    USMLE questions often end with "Which of the following is the most likely diagnosis?".
    We want to strip that out to leave just the patient scenario for the agent.
    """
    # Remove common question suffixes to leave just the clinical vignette
    patterns = [
        r"Which of the following is the most likely diagnosis\?",
        r"Which of the following is the most appropriate next step in management\?",
        r"Which of the following is the most likely cause of.*?\?",
        r"What is the most likely diagnosis\?"
    ]
    
    cleaned_q = question
    for p in patterns:
        cleaned_q = re.sub(p, "", cleaned_q, flags=re.IGNORECASE)
    
    return cleaned_q.strip()

def main():
    print(f"--- Downloading MedQA (USMLE) Dataset ---")
    
    # Load the dataset from Hugging Face (using a reliable host for MedQA)
    # GBaker/MedQA-USMLE-4-options is a standard clean version
    try:
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative config...")
        return

    print(f"-> Loaded {len(dataset)} total test cases.")
    
    # Convert to list for sampling
    all_items = list(dataset)
    
    # Randomly sample
    random.seed(SEED)
    if len(all_items) > SAMPLE_SIZE:
        sample = random.sample(all_items, SAMPLE_SIZE)
    else:
        sample = all_items
        
    print(f"-> Selected {len(sample)} cases for evaluation.")
    
    formatted_data = []
    
    for item in tqdm(sample, desc="Processing cases"):
        # MedQA format: 'question', 'answer' (the text of the correct option), 'options'
        
        # 1. Get the Correct Diagnosis string
        ground_truth = item.get('answer')
        
        # 2. Get the Patient Vignette
        raw_question = item.get('question')
        patient_scenario = clean_question_text(raw_question)
        
        # Only keep valid entries
        if ground_truth and patient_scenario:
            formatted_data.append({
                "text": patient_scenario,
                "diagnosis": ground_truth,
                "original_options": item.get('options') # Keep for reference if needed
            })

    # --- THE FIX IS HERE ---
    # Ensure the directory exists before trying to write the file
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    # Save to file
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(formatted_data, f, indent=2)
        
    print(f"\n✅ Success! Saved {len(formatted_data)} USMLE scenarios to: {OUTPUT_FILE}")
    print("Example Entry:")
    if formatted_data:
        print(json.dumps(formatted_data[0], indent=2))

if __name__ == "__main__":
    main()