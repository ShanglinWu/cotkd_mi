from run_experiment import parse_log, parse_response, evaluate_response, parse_reasoning
import json


def compare_rationales(example1, example2):
    # Extract relevant information from the examples
    rationale1 = example1['chain_of_thought']
    rationale2 = example2['chain_of_thought']
    label1 = example1['answer']
    label2 = example2['answer']

    # Compare the rationales
    max_steps = max(len(rationale1), len(rationale2))
    differences = []
    for i in range(max_steps):
        if i < len(rationale1) and i < len(rationale2):
            if rationale1[i] != rationale2[i]:
                differences.append(
                    f"Step {i+1}: '{rationale1[i]}' vs '{rationale2[i]}'")
        elif i < len(rationale1):
            differences.append(f"Step {i+1}: '{rationale1[i]}' vs (no step)")
        else:
            differences.append(f"Step {i+1}: (no step) vs '{rationale2[i]}'")

    # Compare labels
    if label1 != label2:
        differences.append(f"Label: '{label1}' vs '{label2}'")

    # Calculate a simple distance metric
    distance = len(differences)

    return distance, differences


if __name__ == "__main__":
    # Read the JSON file
    with open('1hop_generate-trio.json', 'r') as file:
        data = json.load(file)

    example1 = data['example1']

    for i in range(5):  # Assuming there are 5 in_context_examples
        distractor = example1[f'in_context_example{i}']['distractor']
        modified = example1[f'in_context_example{i}']['modified']

        distance, differences = compare_rationales(distractor, modified)

        print(f"In-context Example {i}:")
        print(f"Distance between distractor and modified: {distance}")
        if differences:
            print("Differences:")
            for diff in differences:
                print(diff)
        else:
            print("No differences found.")
        print("\n")

    # Compare across different in-context examples
    distractor0 = example1['in_context_example0']['distractor']
    distractor1 = example1['in_context_example1']['modified']

    distance, differences = compare_rationales(distractor0, distractor1)

    print("Comparison between distractor examples of in_context_example0 and in_context_example1:")
    print(f"Distance: {distance}")
    if differences:
        print("Differences:")
        for diff in differences:
            print(diff)
    else:
        print("No differences found.")
