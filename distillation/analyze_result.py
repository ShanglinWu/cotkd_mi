import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from calculate_distance import compare_rationales
from tqdm import tqdm
import argparse
import os
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, set_seed, AutoModelForSeq2SeqLM
from tqdm import tqdm
import re
from openai import OpenAI


def load_model_and_tokenizer(checkpoint_path, args):
    device = torch.device('cuda:{}'.format(args.gpu))
    print(f"Using device: {device}")

    if args.model == "trained-t5":
        model = T5ForConditionalGeneration.from_pretrained("t5-3b")
        tokenizer = T5Tokenizer.from_pretrained("t5-3b")
    elif args.model == "trained-gpt2":
        model = AutoModelForCausalLM.from_pretrained(
            "gpt2", torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained("gpt2", cache_dir='../cache')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    model.resize_token_embeddings(len(tokenizer))

    checkpoint = torch.load(checkpoint_path)['ckpt']
    model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    return model, tokenizer, device


def generate_cot_and_answer(model, tokenizer, prompt, device, args):
    if args.model == "gpt-4o":
        client = OpenAI()
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant providing chain-of-thought reasoning process"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=256,
            n=1
        )

        generated_text = response.choices[0].message.content.strip()

    else:
        inputs = tokenizer(prompt, return_tensors="pt",
                           max_length=500, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, pad_token_id=tokenizer.pad_token_id, max_length=800, num_return_sequences=1)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_text)

    # ---------------format the result-------------------
    answer_index = generated_text.find("answer is ")
    if answer_index == -1:
        chain_of_thought = generated_text
        answer = "N/A"
    else:
        chain_of_thought = generated_text[:answer_index]
        answer = generated_text[answer_index+9:]

    chain_of_thought_index = chain_of_thought.find("Chain of thought:")
    chain_of_thought = chain_of_thought[chain_of_thought_index+17:]

    prompt_index = chain_of_thought.find(
        "You need to only generate the answer part:")
    if prompt_index != -1:
        chain_of_thought = chain_of_thought[prompt_index +
                                            len("You need to only generate the answer part:"):]

    prompt_str = "Chain of thought: 1. Stella is an impus. 2. Stella is bright or an impus. 3. Everything that is bright or an impus is a vumpus. 4. Stella is a vumpus. 5. Stella is happy or a vumpus."
    prompt_index = chain_of_thought.find(prompt_str)
    if prompt_index != -1:
        chain_of_thought = chain_of_thought[prompt_index+len(prompt_str):]

    chain_of_thought = chain_of_thought.replace("Prove: ", '')
    chain_of_thought = chain_of_thought.replace("\n: ", '')
    chain_of_thought = chain_of_thought.replace("\n\n: ", '')
    chain_of_thought = chain_of_thought.replace("Answer: ", '')
    chain_of_thought = chain_of_thought.replace("Question: ", '')
    chain_of_thought = chain_of_thought.replace("True or False: ", '')
    chain_of_thought = chain_of_thought.replace("True or false: ", '')
    chain_of_thought = chain_of_thought.replace("So the", '')
    chain_of_thought = chain_of_thought.replace("Therefore, ", '')
    chain_of_thought = chain_of_thought.replace("By definition,", '')

    steps = re.findall(
        r'\d+\.\s*(.*?)(?=\s*\d+\.|$)', chain_of_thought, re.DOTALL)

    steps = [step.strip() for step in steps]

    steps = [step for step in steps if step]

    chain_of_thought = steps

    # print(generated_text)
    # print('-'*20)
    # print(question+query)

    print("Chain-of-thought is :")
    print(chain_of_thought)
    # print("answer is:")
    # print(answer)
    print("--------------------------")

    return {"chain_of_thought": chain_of_thought, "answer": answer}


def evaluate_checkpoint(checkpoint_path, data_path, version, gold_data_path, args, num_samples=100):
    model, tokenizer, device = load_model_and_tokenizer(checkpoint_path, args)

    with open(data_path, 'r') as f:
        data = [json.loads(line) for line in f]

    with open(gold_data_path, 'r') as f:
        gold_data = [json.loads(line) for line in f]

    data = data[:num_samples]
    gold_data = gold_data[:num_samples]

    distances = []

    for item, gold_item in tqdm(zip(data, gold_data), desc="Processing items", total=len(data)):

        with open('./prompts/CounterCoTQA.evaluate.txt', 'r') as fr:
            original_prompt = json.load(fr)["prompt"]
        prompt = original_prompt.format(item["question"], item["query"])

        generated = generate_cot_and_answer(
            model, tokenizer, prompt, device, args)

        cot1 = {"chain_of_thought": item['chain_of_thought'],
                "answer": item['answer'], "question": item['question']}
        cot2 = {"chain_of_thought": generated['chain_of_thought'],
                "answer": generated['answer'], "question": item['question']}

        gold_cot = {"chain_of_thought": gold_item[version]['chain_of_thought'],
                    "answer": gold_item[version]['answer'], "question": item['question']}

        try:
            result = compare_rationales(cot2, gold_cot)
            print("Distance: " + str(result))
            distances.append(result)
        except:
            continue

    mean_distance = np.mean(distances)
    generate_ratio = len(distances)/len(data)
    print("Mean Distance:" + str(mean_distance))
    print("Genetate Ratio:" + str(generate_ratio))

    return distances, mean_distance


def evaluate_all_checkpoints(checkpoint_dir, data_path, version, gold_data_path, args, num_samples=100):
    all_distances = {}
    means = []

    checkpoint_path = checkpoint_dir+"model_seed42.ckpt"
    distances, mean_distance = evaluate_checkpoint(
        checkpoint_path, data_path, version, gold_data_path, args, num_samples)
    all_distances[0] = sorted(distances)
    means.append(mean_distance)

    # for i in range(10):
    # # ! Modified for only test the final model
    #     checkpoint_path = os.path.join(
    #         checkpoint_dir, f"checkpoint_{i}.ckpt")
    #     distances, mean_distance = evaluate_checkpoint(
    #         checkpoint_path, data_path, version, gold_data_path, num_samples)

    #     all_distances[i] = sorted(distances)
    #     means.append(mean_distance)

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


def main(args):

    version = args.version
    data_path = "./outputs/CounterCoTQA/gpt-neox-20b/dev."+version+".explanation.jsonl"
    gold_data_path = "./data/CounterCoTQA/dev.jsonl"

    if args.model == "trained-t5":

        checkpoint_dir = "./checkpoints/CounterCoTQA/"+version + \
            "/counterfactual0.5_t5-3b_bs8_gs1_lr1e-5_wd0_e3/"

        gpu = 3

        all_distances, means = evaluate_all_checkpoints(
            checkpoint_dir, data_path, version, gold_data_path, args, num_samples=100)

        print(means)

        save_evaluation_results(all_distances, means, version)
        advanced_analysis(all_distances)

        plot_sorted_distances(all_distances)

        plot_means(means)

        print("Evaluation complete. Plots have been saved.")
    elif args.model == "trained-gpt2":

        checkpoint_dir = "./checkpoints/CounterCoTQA/"+version + \
            "/counterfactual0.5_gpt2_bs8_gs1_lr1e-5_wd0_e3/"

        gpu = 3

        all_distances, means = evaluate_all_checkpoints(
            checkpoint_dir, data_path, version, gold_data_path, args, num_samples=100)

        print(means)

        save_evaluation_results(all_distances, means, version)
        advanced_analysis(all_distances)

        plot_sorted_distances(all_distances)

        plot_means(means)

        print("Evaluation complete. Plots have been saved.")

    else:
        # for pre-trained model evaluation
        # -----------------Load Model-----------------------
        model_path = args.model
        if args.model != "gpt-4o":
            tokenizer = AutoTokenizer.from_pretrained(
                model_path, cache_dir='../cache')  # , use_fast=False)
            n_gpus = 1
            free_in_GB = 49
            max_memory = {i: "{}GB".format(free_in_GB)
                          for i in range(args.gpu, args.gpu + n_gpus)}

        if "t5" in args.model:
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_path, torch_dtype=torch.float16).to(args.device)

        else:
            if args.model != "gpt-4o":
                model = AutoModelForCausalLM.from_pretrained(
                    model_path, torch_dtype=torch.float16).to(args.device)

        if args.model != "gpt-4o":
            indicator_token_ids = {
                "stop": tokenizer.encode("\n\nQ")[-2],
            }

            model.eval()
        else:
            model = None
            tokenizer = None

        # -----------------data prepare------------------

        with open('./prompts/CounterCoTQA.evaluate.txt', 'r') as fr:
            original_prompt = json.load(fr)["prompt"]

        with open(data_path, 'r') as f:
            data = [json.loads(line) for line in f]

        with open(gold_data_path, 'r') as f:
            gold_data = [json.loads(line) for line in f]

        data = data[:100]
        gold_data = gold_data[:100]

        # ---------------------evaluate-----------------

        distances = []
        for item, gold_item in tqdm(zip(data, gold_data), desc="Processing items", total=len(data)):
            prompt = original_prompt.format(item["question"], item["query"])

            generated = generate_cot_and_answer(
                model, tokenizer, prompt, args.device, args)

            #! chain_of_thought here need to be a list
            cot1 = {"chain_of_thought": item['chain_of_thought'],
                    "answer": item['answer'], "question": item['question']}
            cot2 = {"chain_of_thought": generated['chain_of_thought'],
                    "answer": generated['answer'], "question": item['question']}

            gold_cot = {"chain_of_thought": gold_item[version]['chain_of_thought'],
                        "answer": gold_item[version]['answer'], "question": item['question']}

            # print(prompt)
            # print(cot2)
            # print(gold_cot)

            # where the first input is the expected answer
            try:
                result = compare_rationales(cot2, gold_cot)
                print("Distance: " + str(result))
                distances.append(result)
            except:
                continue

    mean_distance = np.mean(distances)
    generate_ratio = len(distances)/len(data)
    print("Mean Distance:" + str(mean_distance))
    print("Genetate Ratio:" + str(generate_ratio))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run main.')
    parser.add_argument('--model', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--version', type=str)
    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu))
    main(args)
