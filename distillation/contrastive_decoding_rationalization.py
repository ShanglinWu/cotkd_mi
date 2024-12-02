import json
import argparse
import random
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, set_seed
from tqdm import tqdm
import os
import re
from openai import OpenAI

torch.set_num_threads(4)


# for REPRODUCIBILITY
set_seed(42)

# ----------------------------------------------------- #
# hyper-parameters
num_return_sequences = 1
generation_length = 128

# ----------------------------------------------------- #


def contrastive_decoding(input_seq1, input_seq2, model, tokenizer, indicator_token_ids, args):
    inputs1 = tokenizer(input_seq1, truncation=True,
                        return_tensors='pt').to(args.device)
    input_length1 = len(inputs1.input_ids[0])
    generated1 = inputs1.input_ids
    past_key_values1 = None

    inputs2 = tokenizer(input_seq2, truncation=True,
                        return_tensors='pt').to(args.device)

    if args.debug:
        print("Input sequence 1:", input_seq1)
        print("Input sequence 2:", input_seq2)
        print("Tokenized input 1 length:", len(inputs1.input_ids[0]))
        print("Tokenized input 2 length:", len(inputs2.input_ids[0]))

    input_length2 = len(inputs2.input_ids[0])
    generated2 = inputs2.input_ids
    past_key_values2 = None

    if args.debug:
        print("Starting generation loop")

    with torch.no_grad():
        for step in range(generation_length):
            # get probs given by the original teacher
            if args.debug:
                print(f"Step {step} started")
            attention_mask1 = generated1.new_ones(generated1.shape)
            outputs1 = model(
                input_ids=generated1 if past_key_values1 is None else generated1[:, -1:],
                past_key_values=past_key_values1,
                attention_mask=attention_mask1,
            )
            logits1 = outputs1.logits[:, -1, :]
            past_key_values1 = outputs1.past_key_values
            prob1 = F.log_softmax(logits1 / args.temperature, dim=-1)

            candidate_next_token = prob1.argmax(dim=-1, keepdim=True)
            if args.debug:
                print(
                    f"Candidate next token: {tokenizer.decode(candidate_next_token[0])}")
            if candidate_next_token[0].item() == indicator_token_ids["stop"]:
                if args.debug:
                    print("Stop token encountered in candidate")
                break

            # get probs given by the hallucinating teacher
            attention_mask2 = generated2.new_ones(generated2.shape)
            outputs2 = model(
                input_ids=generated2 if past_key_values2 is None else generated2[:, -1:],
                past_key_values=past_key_values2,
                attention_mask=attention_mask2,
            )
            logits2 = outputs2.logits[:, -1, :]
            past_key_values2 = outputs2.past_key_values
            prob2 = F.log_softmax(logits2, dim=-1)

            # contrastive decoding
            debiased_prob = prob1 - args.interpolation * prob2
            next_token = debiased_prob.argmax(dim=-1, keepdim=True)

            if next_token[0] == indicator_token_ids["stop"]:
                break

            generated1 = torch.cat((generated1, next_token), dim=1)
            generated2 = torch.cat((generated2, next_token), dim=1)

            # if args.debug:
            #     print(
            #         f"Step {step}: Current generation: {tokenizer.decode(generated1[0], skip_special_tokens=True)}")

    generation = tokenizer.decode(
        generated1[0][input_length1:], skip_special_tokens=True)
    if args.debug:
        print('-'*20)
        print("Final generation:", generation)
        print('-'*20)
    return generation


def openai_completion(client, input_seq1, model_name, args):

    response = client.chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant providing chain-of-thought reasoning process"},
            {"role": "user", "content": input_seq1}
        ],
        max_tokens=256,
        n=1
    )

    generated_text = response.choices[0].message.content.strip()
    
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
    
    answer_index = chain_of_thought.find("Answer is")
    chain_of_thought = chain_of_thought[:answer_index]

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
    chain_of_thought = chain_of_thought.replace("\n", '')
    chain_of_thought = chain_of_thought.replace("Answer: ", '')
    chain_of_thought = chain_of_thought.replace("Question: ", '')
    chain_of_thought = chain_of_thought.replace("True or False: ", '')
    chain_of_thought = chain_of_thought.replace("True or false: ", '')
    chain_of_thought = chain_of_thought.replace("So the", '')
    chain_of_thought = chain_of_thought.replace("Therefore, ", '')
    chain_of_thought = chain_of_thought.replace("By definition,", '')

    return chain_of_thought


def main(args):
    # ----------------------------------------------------- #
    # load LM
    if args.model == 'gpt-4o':
        client = OpenAI()
    else:
        if args.model == 'google/gemma-2b':
            from huggingface_hub import login
            login("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")
        model_path = args.model
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, cache_dir='../cache')  # , use_fast=False)
        n_gpus = 1
        free_in_GB = 49
        max_memory = {i: "{}GB".format(free_in_GB)
                      for i in range(args.gpu, args.gpu + n_gpus)}

        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16).to(args.device)

        indicator_token_ids = {
            "stop": tokenizer.encode("\n\nQ")[-2],
        }

        model.eval()

    # ----------------------------------------------------- #
    # prepare data
    with open('./prompts/{}.{}.txt'.format(args.dataset, args.prompt), 'r') as fr:
        prompt = json.load(fr)["prompt"]

    prompt_without_question = '\n\n'.join(prompt.split('\n\n')[:-1])+'\n\n'

    # for split in args.eval_split.split(','):
    split = "train"
    with open('./data/{}/{}.jsonl'.format(args.dataset, split), 'r') as fr:
        examples = [json.loads(line) for line in fr.readlines()]

    output_path = os.path.join(args.output_prefix, '{}.{}.{}.jsonl'.format(
        split, args.version, args.prompt))

    fw = open(output_path, 'w', buffering=1)
    for example in tqdm(examples):
        if "distractor" in example:
            question = example[args.version]["question"]
            query = example[args.version]["query"]

            answer = example[args.version]["answer"]
            if answer == "true":
                wrong_answer = "false"
            elif answer == "false":
                wrong_answer = "true"
            else:  # N/A
                wrong_answer = "N/A"

            input_seq1 = prompt.format(question, query, answer)
            input_seq2 = prompt.format(question, query, wrong_answer)

            if args.model == "gpt-4o":
                generation = openai_completion(
                    client, input_seq1, args.model, args)
            else:
                generation = contrastive_decoding(
                    input_seq1, input_seq2, model, tokenizer, indicator_token_ids, args)

            # Split the generation into chain of thought and answer

            steps = re.findall(
                r'\d+\.\s*(.*?)(?=\s*\d+\.|$)', generation, re.DOTALL)

            steps = [step.strip() for step in steps]

            steps = [step for step in steps if step]

            fw.write(json.dumps({
                "id": example["qid"],
                "version": args.version,
                "question": question,
                "query": query,
                "answer": answer,
                "chain_of_thought": steps
            }) + "\n")

        # The code for using dataset other than CounterCoTQA, haven't add openai api feature
        else:
            if "context" in example:
                formatted_question = example["context"]
                choices = ["false", "true"]
                if "counterfactual" in split:
                    answer = "false" if example["answer"] == 1 else "true"
                    wrong_answer = "false" if example["answer"] == 0 else "true"
                else:
                    answer = "false" if example["answer"] == 0 else "true"
                    wrong_answer = "false" if example["answer"] == 1 else "true"
                question = example["context"]
            else:
                formatted_question = example["question"]
                choices = example["choices"] if "choices" in example else [
                    "no", "yes"]
                question = example["question"]
                if "choices" in example and len(example["choices"]) > 2:
                    if "counterfactual" in split:
                        answer = random.choice(
                            example["choices"][:example["answer"]] + example["choices"][example["answer"]+1:])
                        wrong_answer = example["choices"][example["answer"]]
                    else:
                        answer = example["choices"][example["answer"]]
                        wrong_answer = random.choice(
                            example["choices"][:example["answer"]] + example["choices"][example["answer"]+1:])
                else:
                    if "counterfactual" in split:
                        answer = "yes" if example["answer"] == 0 else "no"
                        wrong_answer = "yes" if example["answer"] == 1 else "no"
                    else:
                        answer = "yes" if example["answer"] == 1 else "no"
                        wrong_answer = "yes" if example["answer"] == 0 else "no"

            if "choices" in example and len(example["choices"]) > 2:
                choices_seq = ""
                formatted_question += "\nAnswer Choices:"
                for choice_id, choice in enumerate(example["choices"]):
                    formatted_question += "\n({}) {}".format(
                        chr(ord('a')+choice_id), choice)
                    choices_seq += " ({}) {}".format(chr(ord('A') +
                                                         choice_id), choice)

            input_seq1 = prompt.format(formatted_question, answer)
            # replace wrong_answer with "" if using empty string as the perturbed answer
            input_seq2 = prompt.format(formatted_question, wrong_answer)
            generation = contrastive_decoding(
                input_seq1, input_seq2, model, tokenizer, indicator_token_ids, args)

            if "context" in example:
                fw.write(json.dumps(
                    {"id": example["id"], "answer": answer, "statement": question, "explanation": generation_list}) + "\n")
            else:
                if "choices" in example and len(example["choices"]) > 2:
                    fw.write(json.dumps({"id": example["id"], "answer": answer, "question": question, "choices": choices_seq.strip(
                    ), "explanation": generation.strip()}) + "\n")
                else:
                    fw.write(json.dumps(
                        {"id": example["qid"], "answer": answer, "question": question, "explanation": generation.strip()}) + "\n")
    fw.close()
    # ----------------------------------------------------- #


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run main.')
    parser.add_argument('--dataset', '-d', type=str)
    parser.add_argument('--output_prefix', '-o', type=str)
    parser.add_argument('--prompt', '-p', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--num_process', type=int, default=1)
    parser.add_argument('--eval_split', type=str,
                        default='test,dev,train,train.counterfactual')
    parser.add_argument("--debug", action='store_true')
    parser.add_argument('--version', type=str, choices=['base', 'distractor', 'modified'],
                        required=True, help='Specify which version of the data to process')

    # debiased factor
    parser.add_argument('--interpolation', type=float, default=0.5)

    # decoding strategy
    parser.add_argument('--temperature', type=float, default=1.0)

    # gpu and workers option
    parser.add_argument('--gpu', type=int, default=0)

    args = parser.parse_args()
    args.device = torch.device('cuda:{}'.format(args.gpu))

    main(args)
