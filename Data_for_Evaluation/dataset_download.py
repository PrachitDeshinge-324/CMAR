import json
import random
from datasets import load_dataset

# Load the disease-diagnosis dataset
dataset = load_dataset('sajjadhadi/disease-diagnosis-dataset')

# Take a random sample of 100 from the test split
all_items = list(dataset['test'])
sample_size = 150
if len(all_items) < sample_size:
    sample = all_items
else:
    sample = random.sample(all_items, sample_size)

# Save the sample to a local JSON file
output_file = 'disease_diagnosis_dataset_100.json'
with open(output_file, 'w') as f:
    f.write('[\n')
    total = len(sample)
    for i, item in enumerate(sample):
        json.dump(item, f)
        if i != total - 1:
            f.write(',\n')
        else:
            f.write('\n')
    f.write(']\n')

print(f"Saved dataset to {output_file} (random sample, entries: {len(sample)})")
