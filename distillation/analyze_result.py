import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from calculate_distance import compare_rationales
from tqdm import tqdm
import os


def load_model_and_tokenizer(checkpoint_path,):
    # 检查GPU是否可用
    device = torch.device('cuda:{}'.format(gpu))
    print(f"Using device: {device}")

    # 加载预训练的T5模型和tokenizer
    model = T5ForConditionalGeneration.from_pretrained("t5-3b")
    tokenizer = T5Tokenizer.from_pretrained("t5-3b")

    # 加载本地checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    model = model.to(device)
    model.eval()

    return model, tokenizer, device


def generate_cot_and_answer(model, tokenizer, question, query, device):
    prompt = question + query + "\nAnswer: "
    inputs = tokenizer(prompt, return_tensors="pt",
                       max_length=800, truncation=False).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_length=256, num_return_sequences=1)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 解析生成的文本来提取chain-of-thought和answer
    answer_index = generated_text.find("answer is ")
    if answer_index == -1:
        chain_of_thought = generated_text
        answer = "N/A"
    else:
        chain_of_thought = generated_text[:answer_index]
        answer = generated_text[answer_index+9:]

    chain_of_thought = chain_of_thought.replace("Prove: ", '')
    chain_of_thought = chain_of_thought.replace("Answer: ", '')
    chain_of_thought = chain_of_thought.replace("Question: ", '')
    chain_of_thought = chain_of_thought.replace("True or False: ", '')
    chain_of_thought = chain_of_thought.replace("True or false: ", '')
    chain_of_thought = chain_of_thought.replace("So the", '')

    # print(generated_text)
    # print('-'*20)
    # print(question+query)

    # print("Chain-of-thought is :")
    # print(chain_of_thought)
    # print("answer is:")
    # print(answer)
    print("--------------------------")

    return {"chain_of_thought": chain_of_thought, "answer": answer}


def evaluate_checkpoint(checkpoint_path, data_path, version, gold_data_path, num_samples=100):
    model, tokenizer, device = load_model_and_tokenizer(checkpoint_path)

    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    with open(gold_data_path, 'r') as f:
        gold_data = [json.loads(line) for line in f]

    data = data[:num_samples]
    gold_data = gold_data[:num_samples]

    distances = []

    for item, gold_item in tqdm(zip(data, gold_data)):
        generated = generate_cot_and_answer(
            model, tokenizer, item['question'], item['query'], device)

        cot1 = {"chain_of_thought": item['chain_of_thought'],
                "answer": item['answer'], "question": item['question']}
        cot2 = {"chain_of_thought": generated['chain_of_thought'],
                "answer": generated['answer'], "question": item['question']}

        expected_answer = ' '.join(
            gold_item[version]["chain_of_thought"])+' '+gold_item[version]["answer"]

        result = compare_rationales(cot1, cot2)
        result = 1-(1-result)*10
        print("Distance: " + str(result))
        distances.append(result)

    mean_distance = np.mean(distances)

    return distances, mean_distance


def evaluate_all_checkpoints(checkpoint_dir, data_path, version, gold_data_path, num_samples=100):
    all_distances = {}
    means = []

    for i in range(10):
        checkpoint_path = os.path.join(
            checkpoint_dir, f"checkpoint_{i}.ckpt")
        distances, mean_distance = evaluate_checkpoint(
            checkpoint_path, data_path, version, gold_data_path, num_samples)

        all_distances[i] = sorted(distances)
        means.append(mean_distance)

    return all_distances, means


def plot_sorted_distances(all_distances):
    plt.figure(figsize=(12, 6))

    checkpoints = [0, 2, 7]
    colors = ['red', 'green', 'blue']
    labels = ['Checkpoint 0', 'Checkpoint 2', 'Checkpoint 7']

    for checkpoint, color, label in zip(checkpoints, colors, labels):
        sorted_distances = all_distances[checkpoint]

        # 使用直方图而不是折线图
        plt.hist(sorted_distances, bins=20,
                 alpha=0.5, color=color, label=label)

    plt.xlabel('Distance')
    plt.ylabel('Frequency')
    plt.title('Distribution of Distances for Checkpoints 0, 2, and 7')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig("distances_histogram.png")
    plt.close()


def advanced_analysis(all_distances):
    plt.figure(figsize=(15, 10))

    # 箱型图
    plt.subplot(2, 2, 1)
    sns.boxplot(data=[all_distances[i] for i in range(10)])
    plt.title('Boxplot of Distances for Each Checkpoint')
    plt.xlabel('Checkpoint')
    plt.ylabel('Distance')

    # CDF
    plt.subplot(2, 2, 2)
    for i in range(0, 10, 2):  # 每隔一个checkpoint画一条CDF曲线
        sns.ecdfplot(all_distances[i], label=f'Checkpoint {i}')
    plt.title('Cumulative Distribution Function of Distances')
    plt.xlabel('Distance')
    plt.ylabel('Cumulative Probability')
    plt.legend()

    # 中位数和四分位数范围
    medians = [np.median(all_distances[i]) for i in range(10)]
    q1 = [np.percentile(all_distances[i], 25) for i in range(10)]
    q3 = [np.percentile(all_distances[i], 75) for i in range(10)]

    plt.subplot(2, 2, 3)
    plt.plot(range(10), medians, marker='o', label='Median')
    plt.fill_between(range(10), q1, q3, alpha=0.2, label='IQR')
    plt.title('Median and IQR of Distances')
    plt.xlabel('Checkpoint')
    plt.ylabel('Distance')
    plt.legend()

    # 距离变化率
    changes = [np.mean(all_distances[i+1]) - np.mean(all_distances[i])
               for i in range(9)]
    plt.subplot(2, 2, 4)
    plt.bar(range(9), changes)
    plt.title('Average Change in Distance Between Checkpoints')
    plt.xlabel('Checkpoint Transition')
    plt.ylabel('Change in Average Distance')

    plt.tight_layout()
    plt.savefig("advanced_analysis.png")
    plt.close()


def save_evaluation_results(all_distances, means, version):
    results = {
        "all_distances": all_distances,
        "means": means
    }

    with open("evaluation_results"+version+".json", "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation results saved to evaluation_results.json")


def plot_means(means):
    plt.figure(figsize=(10, 6))
    plt.plot(range(10), means, marker='o')
    plt.xlabel('Checkpoint')
    plt.ylabel('Mean Distance')
    plt.title('Mean Distance for Each Checkpoint')
    plt.xticks(range(10))
    plt.savefig("means_line_plot.png")
    plt.close()


# 运行评估
version = "modified"
checkpoint_dir = "./checkpoints/CounterCoTQA/"+version + \
    "/counterfactual0.5_t5-3b_bs8_gs1_lr1e-5_wd0_e3/checkpoints_seed42/"
data_path = "./outputs/CounterCoTQA/dev."+version+".explanation.jsonl"
gold_data_path = "./data/CounterCoTQA/dev.jsonl"
gpu = 3

all_distances, means = evaluate_all_checkpoints(
    checkpoint_dir, data_path, version, gold_data_path, num_samples=100)

save_evaluation_results(all_distances, means, version)
advanced_analysis(all_distances)

# 绘制直方图
plot_sorted_distances(all_distances)

# 绘制均值折线图
plot_means(means)

print("Evaluation complete. Plots have been saved.")
