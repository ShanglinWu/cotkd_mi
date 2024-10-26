from run_experiment import parse_response, evaluate_response, parse_reasoning
from math import sqrt
import numpy as np


def wilson_conf_interval(p, n, z=1.96):
    if n == 0:
        return (0, 1)
    denominator = 1 + z**2/n
    centre_adjusted_probability = p + z*z / (2*n)
    adjusted_standard_deviation = sqrt((p*(1 - p) + z*z / (4*n)) / n)

    lower_bound = (centre_adjusted_probability - z *
                   adjusted_standard_deviation) / denominator
    upper_bound = (centre_adjusted_probability + z *
                   adjusted_standard_deviation) / denominator
    return (lower_bound, upper_bound)


def preprocess_answer(answer):
    return answer.rstrip('.').lower()


def compare_rationales(cot1, cot2):
    def analyze_chain(chain, question, expected_answer, proofs_only):
        parse_errors = []
        predicted_answer = chain['chain_of_thought']
        if isinstance(predicted_answer, list):
            predicted_answer = ' '.join(predicted_answer)

        (predicted_proof, predicted_label, errors) = parse_response(predicted_answer)
        parse_errors.extend(errors)

        predicted_label = preprocess_answer(predicted_label)
        processed_expected_answer = preprocess_answer(expected_answer)

        expected_proof = parse_reasoning(expected_answer, parse_errors)

        # print(chain['answer'])
        # print("prooooooooooooooooofs_onlt=")
        # print(proofs_only)

        result = evaluate_response(predicted_proof, predicted_label, expected_answer,
                                   parse_reasoning(question, parse_errors), proofs_only, parse_errors)

        (_, _, correct_steps, correct_and_useful_steps, redundant_steps, unparseable_steps,
         wrong_branch_steps, useful_skip_steps, wrong_skip_steps, useful_non_atomic_steps,
         wrong_non_atomic_steps, invalid_steps, incorrect_steps, found_conclusion, _, _) = result

        total_steps = len(chain['chain_of_thought'])
        accuracy = len(correct_and_useful_steps) / \
            total_steps if total_steps > 0 else 0

        return {
            'accuracy': accuracy,
            'correct_steps': correct_steps,
            'correct_and_useful_steps': correct_and_useful_steps,
            'redundant_steps': redundant_steps,
            'unparseable_steps': unparseable_steps,
            'wrong_branch_steps': wrong_branch_steps,
            'useful_skip_steps': useful_skip_steps,
            'wrong_skip_steps': wrong_skip_steps,
            'useful_non_atomic_steps': useful_non_atomic_steps,
            'wrong_non_atomic_steps': wrong_non_atomic_steps,
            'invalid_steps': invalid_steps,
            'incorrect_steps': incorrect_steps,
            'found_conclusion': found_conclusion,
            'proofs_only': proofs_only
        }

    question = cot1['question']

    # use cot1 as expected_answer
    expected_answer = " ".join(cot1['chain_of_thought']) + " " + cot1['answer']

    # check proofs_only
    proofs_only = cot1['answer'] == 'N/A' or cot2['answer'] == 'N/A'

    # analyze cot2 compare to cot1
    result2 = analyze_chain(cot2, question, expected_answer, proofs_only)

    accuracy2 = result2['accuracy']
    total_steps2 = len(cot2['chain_of_thought'])

    (lower2, upper2) = wilson_conf_interval(accuracy2, total_steps2)

    # calculate distance
    base_distance = 1 - accuracy2

    # consider answer
    if cot1['answer'].upper() == "N/A" and cot2['answer'].upper() == "N/A":
        answer_penalty = 0  # if both N/A, no penalty
    elif cot1['answer'].upper() == "N/A" or cot2['answer'].upper() == "N/A":
        answer_penalty = 0  # if only one is N/A, add penalty
    else:
        answer_penalty = 0 if cot1['answer'].lower(
        ) != cot2['answer'].lower() else 0

    distance = min(base_distance + answer_penalty, 1)

    differences = []
    for i in range(max(len(cot1['chain_of_thought']), len(cot2['chain_of_thought']))):
        if i >= len(cot1['chain_of_thought']) or i >= len(cot2['chain_of_thought']) or cot1['chain_of_thought'][i] != cot2['chain_of_thought'][i]:
            differences.append((i,
                                cot1['chain_of_thought'][i] if i < len(
                                    cot1['chain_of_thought']) else None,
                                cot2['chain_of_thought'][i] if i < len(cot2['chain_of_thought']) else None))

    return distance


def compare_rationales_1(cot1, cot2, expected_answer):
    def analyze_chain(chain, question, expected_answer, proofs_only):
        parse_errors = []
        predicted_answer = " ".join(chain['chain_of_thought'])
        if isinstance(predicted_answer, list):
            predicted_answer = ' '.join(predicted_answer)

        (predicted_proof, predicted_label, errors) = parse_response(predicted_answer)
        parse_errors.extend(errors)

        # 预处理预测的标签和期望的答案
        predicted_label = preprocess_answer(predicted_label)
        processed_expected_answer = preprocess_answer(expected_answer)

        # 解析期望的推理链
        expected_proof = parse_reasoning(expected_answer, parse_errors)

        result = evaluate_response(predicted_proof, predicted_label, expected_answer,
                                   parse_reasoning(question, parse_errors), proofs_only, parse_errors)

        (_, _, correct_steps, correct_and_useful_steps, redundant_steps, unparseable_steps,
         wrong_branch_steps, useful_skip_steps, wrong_skip_steps, useful_non_atomic_steps,
         wrong_non_atomic_steps, invalid_steps, incorrect_steps, found_conclusion, _, _) = result

        total_steps = len(chain['chain_of_thought'])
        accuracy = len(correct_and_useful_steps) / \
            total_steps if total_steps > 0 else 0

        return {
            'accuracy': accuracy,
            'correct_steps': correct_steps,
            'correct_and_useful_steps': correct_and_useful_steps,
            'redundant_steps': redundant_steps,
            'unparseable_steps': unparseable_steps,
            'wrong_branch_steps': wrong_branch_steps,
            'useful_skip_steps': useful_skip_steps,
            'wrong_skip_steps': wrong_skip_steps,
            'useful_non_atomic_steps': useful_non_atomic_steps,
            'wrong_non_atomic_steps': wrong_non_atomic_steps,
            'invalid_steps': invalid_steps,
            'incorrect_steps': incorrect_steps,
            'found_conclusion': found_conclusion
        }

    question = cot1['question']
    proofs_only = cot1['answer'] == 'N/A' or cot2['answer'] == 'N/A'

    result1 = analyze_chain(cot1, question, expected_answer, proofs_only)
    result2 = analyze_chain(cot2, question, expected_answer, proofs_only)

    accuracy1 = result1['accuracy']
    accuracy2 = result2['accuracy']

    total_steps1 = len(cot1['chain_of_thought'])
    total_steps2 = len(cot2['chain_of_thought'])

    (lower1, upper1) = wilson_conf_interval(accuracy1, total_steps1)
    (lower2, upper2) = wilson_conf_interval(accuracy2, total_steps2)

    distance = abs(accuracy1 - accuracy2)

    differences = []
    for i in range(max(total_steps1, total_steps2)):
        if i >= total_steps1 or i >= total_steps2 or cot1['chain_of_thought'][i] != cot2['chain_of_thought'][i]:
            differences.append((i,
                                cot1['chain_of_thought'][i] if i < total_steps1 else None,
                                cot2['chain_of_thought'][i] if i < total_steps2 else None))

    return distance
