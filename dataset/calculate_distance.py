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
    def analyze_chain(chain, question, expected_answer):
        parse_errors = []
        predicted_answer = " ".join(chain['chain_of_thought'])

        (predicted_proof, predicted_label, errors) = parse_response(predicted_answer)
        parse_errors.extend(errors)

        # 预处理预测的标签和期望的答案
        predicted_label = preprocess_answer(predicted_label)
        processed_expected_answer = preprocess_answer(expected_answer)

        # 解析期望的推理链
        expected_proof = parse_reasoning(expected_answer, parse_errors)

        result = evaluate_response(predicted_proof, predicted_label, expected_answer,
                                   parse_reasoning(question, parse_errors), False, parse_errors)

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

    # 使用 cot1 作为 expected_answer
    expected_answer = " ".join(cot1['chain_of_thought']) + " " + cot1['answer']

    # 只分析 cot2，使用 cot1 作为参考
    result2 = analyze_chain(cot2, question, expected_answer)

    accuracy2 = result2['accuracy']
    total_steps2 = len(cot2['chain_of_thought'])

    (lower2, upper2) = wilson_conf_interval(accuracy2, total_steps2)

    # 计算基础距离
    base_distance = 1 - accuracy2

    # 考虑答案的影响
    answer_penalty = 0.2 if cot1['answer'].lower(
    ) != cot2['answer'].lower() else 0

    # 计算最终距离，确保不超过1
    distance = min(base_distance + answer_penalty, 1)

    differences = []
    for i in range(max(len(cot1['chain_of_thought']), len(cot2['chain_of_thought']))):
        if i >= len(cot1['chain_of_thought']) or i >= len(cot2['chain_of_thought']) or cot1['chain_of_thought'][i] != cot2['chain_of_thought'][i]:
            differences.append((i,
                                cot1['chain_of_thought'][i] if i < len(
                                    cot1['chain_of_thought']) else None,
                                cot2['chain_of_thought'][i] if i < len(cot2['chain_of_thought']) else None))

    return {
        'distance': distance,
        'base_distance': base_distance,
        'answer_penalty': answer_penalty,
        'accuracy': accuracy2,
        'confidence_interval': (lower2, upper2),
        'result': result2,
        'differences': differences
    }


# 使用示例
cot1 = {}
cot2 = {}
cot1['chain_of_thought'] = ["Stella is a numpus.", "Each numpus is a rompus.", "Every rompus is a dumpus.",
                            "Stella is a dumpus.", "Each dumpus is a zumpus.", "Zumpuses are orange.", "Stella is orange."]
cot1['answer'] = "False"
cot2['chain_of_thought'] = ["Stella is a numpus.", "Every rompus is a dumpus.", "Each numpus is a rompus.",
                            "Stella is a dumpus.", "Zumpuses are orange.", "Each dumpus is a zumpus.", "Stella is orange."]
cot2['answer'] = "False"

cot1['question'] = cot2['question'] = "Each numpus is a rompus. Every rompus is a dumpus. Each dumpus is a zumpus. Zumpuses are orange. Stella is a numpus. True or false: Stella is not orange."

result = compare_rationales(cot1, cot2)

print(f"Distance between COT1 and COT2: {result['distance']}")
print(f"Base distance: {result['base_distance']}")
print(f"Answer penalty: {result['answer_penalty']}")
print(f"Accuracy of COT2 relative to COT1: {result['accuracy']}")
print(f"Confidence interval: {result['confidence_interval']}")
print("COT2 analysis:")
for key, value in result['result'].items():
    print(f"  {key}: {value}")
print("Differences:")
for diff in result['differences']:
    print(f"Step {diff[0]}:")
    print(f"  COT1: {diff[1]}")
    print(f"  COT2: {diff[2]}")
