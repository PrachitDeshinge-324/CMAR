import os
import json
import argparse
from datasets import load_dataset, get_dataset_config_names

OUTPUT_DIR = "data_examples"
EXAMPLES_FILE = os.path.join(OUTPUT_DIR, "medagents_examples.jsonl")
DATASET_ID = "xk-huang/medagents-benchmark"
SPLIT = "test"
NUM_EXAMPLES = 200

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", help="Dataset config name (e.g. MedQA)")
    parser.add_argument("--n", "-n", type=int, default=NUM_EXAMPLES, help="Number of examples to print/save")
    args = parser.parse_args()

    chosen_config = args.config or os.environ.get("MEDAGENT_CONFIG")

    try:
        configs = get_dataset_config_names(DATASET_ID)
    except Exception as e:
        print(f"Failed to list configs for dataset {DATASET_ID}: {e}")
        return

    if not configs:
        print(f"No configs found for {DATASET_ID}")
        return

    print(f"Available configs: {configs}")

    if chosen_config:
        if chosen_config not in configs:
            print(f"Requested config '{chosen_config}' not found. Falling back to default: {configs[0]}")
            chosen_config = configs[0]
    else:
        chosen_config = configs[0]
        print(f"No config provided. Using default config: {chosen_config}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Loading dataset: {DATASET_ID} (config='{chosen_config}', split='{SPLIT}') ...")
    try:
        ds = load_dataset(DATASET_ID, chosen_config, split=SPLIT)
    except Exception as e:
        print(f"Failed to load dataset with config '{chosen_config}': {e}")
        return

    print(f"Dataset loaded. Total examples: {len(ds)}")
    fields = ds.column_names
    print(f"Fields: {fields}\n")

    n_to_save = min(args.n, len(ds))
    with open(EXAMPLES_FILE, "w") as out:
        for i, item in enumerate(ds.select(range(n_to_save))):
            print(f"--- Example {i+1} ---")
            for key in ["task_type", "question", "ground_truth", "answer", "choices"]:
                if key in item:
                    val = item.get(key)
                    if isinstance(val, str):
                        print(f"{key}: {val[:400]}{'...' if len(val)>400 else ''}")
                    else:
                        print(f"{key}: {val}")
            print()
            out.write(json.dumps(item, default=str) + "\n")

    print(f"Saved {n_to_save} examples to {EXAMPLES_FILE}")

if __name__ == "__main__":
    main()