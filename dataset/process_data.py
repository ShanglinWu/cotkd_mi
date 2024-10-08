import os
import json
import random
import re
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib as mpl

# Use system default font
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['DejaVu Sans',
                                   'Helvetica', 'Arial', 'sans-serif']

# Set input and output paths
input_dir = 'generate_result/CounterCoTQA'
output_file = 'all.jsonl'
train_file = 'train.jsonl'
test_file = 'test.jsonl'

# Initialize statistics dictionaries
hop_count = defaultdict(int)
rule_count = defaultdict(int)

# Read all JSON files and merge data
all_data = []
for filename in os.listdir(input_dir):
    if filename.endswith('.json'):
        file_path = os.path.join(input_dir, filename)

        # Check if file is empty
        if os.path.getsize(file_path) == 0:
            print(f"Skipping empty file: {filename}")
            continue

        # Extract hop count and deduction-rule
        match = re.match(r'(\d+)hop_(.*?)_generate-trio\.json', filename)
        if match:
            hop = int(match.group(1))
            rule = match.group(2)

            with open(file_path, 'r', encoding='utf-8') as f:
                file_data = [json.loads(line.strip()) for line in f]
                all_data.extend(file_data)

                # Update statistics
                hop_count[hop] += len(file_data)
                rule_count[rule] += len(file_data)

# Shuffle data
random.shuffle(all_data)

# Save merged data
with open(output_file, 'w', encoding='utf-8') as f:
    for item in all_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

# Calculate split point for train and test sets
split_point = int(len(all_data) * 0.2)  # 20% for testing, 80% for training

# Split data
test_data = all_data[:split_point]
train_data = all_data[split_point:]

# Save training set
with open(train_file, 'w', encoding='utf-8') as f:
    for item in train_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

# Save test set
with open(test_file, 'w', encoding='utf-8') as f:
    for item in test_data:
        json.dump(item, f, ensure_ascii=False)
        f.write('\n')

print(f"Total data entries: {len(all_data)}")
print(
    f"Training set entries: {len(train_data)} ({len(train_data)/len(all_data)*100:.2f}%)")
print(
    f"Test set entries: {len(test_data)} ({len(test_data)/len(all_data)*100:.2f}%)")

# Draw statistics charts
plt.figure(figsize=(15, 6))

# Hop count statistics chart
plt.subplot(1, 2, 1)
bars = plt.bar(hop_count.keys(), hop_count.values())
plt.title('Hop Count Statistics', fontsize=14)
plt.xlabel('Number of Hops', fontsize=12)
plt.ylabel('Number of Entries', fontsize=12)
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}',
             ha='center', va='bottom')

# Deduction-rule statistics chart
plt.subplot(1, 2, 2)
rules = list(rule_count.keys())
counts = list(rule_count.values())
bars = plt.bar(range(len(rules)), counts)
plt.title('Deduction-rule Statistics', fontsize=14)
plt.xlabel('Deduction-rule', fontsize=12)
plt.ylabel('Number of Entries', fontsize=12)
plt.xticks(range(len(rules)), rules, rotation=45, ha='right')
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height}',
             ha='center', va='bottom')

plt.tight_layout()
plt.savefig('statistics.png', dpi=300, bbox_inches='tight')
plt.show()

# Print detailed statistics
print("\nHop count statistics:")
for hop, count in sorted(hop_count.items()):
    print(f"{hop} hop: {count} entries")

print("\nDeduction-rule statistics:")
for rule, count in rule_count.items():
    print(f"{rule}: {count} entries")
