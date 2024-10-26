import json
import os
from sklearn.model_selection import train_test_split

# Load the dataset


def load_jsonl(filename):
    with open(filename, 'r') as f:
        return [json.loads(line) for line in f]


# Load train and test data
train_data = load_jsonl("data/CounterCoTQA/train.jsonl")
test_data = load_jsonl("data/CounterCoTQA/test.jsonl")

# Split the train data into train and dev sets
train, dev = train_test_split(train_data, test_size=0.2, random_state=42)

# Create the output directory
os.makedirs("data/DATASET", exist_ok=True)

# Function to save data to a JSON Lines file


def save_jsonl(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')


# Save train, dev, and test sets
save_jsonl(train, "data/DATASET/train.jsonl")
save_jsonl(dev, "data/DATASET/dev.jsonl")
save_jsonl(test_data, "data/DATASET/test.jsonl")

# Create and save counterfactual train set


def create_counterfactual(item):
    counterfactual = item.copy()
    for key in ['base', 'distractor', 'modified']:
        if 'answer' in counterfactual[key]:
            if counterfactual[key]['answer'] == 'true':
                counterfactual[key]['answer'] = 'false'
            elif counterfactual[key]['answer'] == 'false':
                counterfactual[key]['answer'] = 'true'
            # If the answer is 'N/A', we keep it as is
    return counterfactual


train_counterfactual = [create_counterfactual(item) for item in train]
save_jsonl(train_counterfactual, "data/DATASET/train.counterfactual.jsonl")

print("Dataset split and saved successfully.")
