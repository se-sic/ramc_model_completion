"""
Input to this script are json lists of prompts, ground_truth completions, and generated completions.
This script takes care of parsing and a strctural comparison of the ground truth completion and the generated completion.

The basic idea is that if we ensure that the "anchors" in the prompt are identical, then we only need to compare the completions and not the entire graphs.    
To this end, we add the id_s of the anchor node to the labels.

We only compare structure, change_type, and type of nodes and edges. We ommit further attributes.

We check the correct completion on the following "levels":
        # Attribute/Sementic Information -> HUman Manual Check
        # Type Information               -> Isomorphism Check
        # Change Information             -> Isomorphism Check
        # Structural Information         -> Iso Check w/o labels
"""
import json
import re
import sys, os
import pandas as pd
from networkx.algorithms.isomorphism import GraphMatcher, DiGraphMatcher
from scipy.stats import mannwhitneyu

from mining.isograph import IsoGraph
from edgel_utils import get_anchored_completion_graph, get_prompt_graphs, V_JSON

SYNTHETIC_DATASET_DEFAULT = False


def get_last_graph(example,  synthetic_dataset):
    prompt_graphs = get_prompt_graphs(example['prompt'],  synthetic_dataset=synthetic_dataset)
    prompt = prompt_graphs[-1] # select the last graph in the prompt

    #Todo 
    if not str(example['completion']).startswith('e'):
        completion =  'e' + example['completion']
    else:
        completion = example['completion']

    full_graph = prompt + "\n" + completion
    return prompt, completion, full_graph

def only_keep_change_types(graph):

    new_graph = graph.copy()
    for _, _, data in new_graph.edges(data=True):
        label_dict = json.loads(data['label'])
        label_dict = {"changeType": label_dict['changeType']}
        data['label'] = json.dumps(label_dict)

    for  _, data in new_graph.nodes(data=True):
        label_dict = json.loads(data['label'])
        if "changeType" in label_dict.keys():
            id_val = label_dict.get('serialized_node_id', '')
            label_dict = {"changeType": label_dict['changeType'], "serialized_node_id": id_val}
            data['label'] = json.dumps(label_dict)


    return new_graph

def remove_all_labels(graph):

    new_graph = graph.copy()
    for _, _, data in new_graph.edges(data=True):
        if 'label' in data:
            data['label'] = ""

    for  n, data in new_graph.nodes(data=True):
        if 'label' in data:
            id_val = json.loads(data['label']).get('serialized_node_id', '')
            data['label'] = {'serialized_node_id':id_val}

    return new_graph

def check_at_least_one_edge_presence(graph_gt, graph_generated):
    for edge in graph_gt.edges():
        if edge in graph_generated.edges():
            return True
    return False


def check_for_correctness(input_path: str, output_path: str, synthetic_dataset: bool = False):
    # Create folders for output (if necessary)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(input_path, 'r') as f:
        for i, line in enumerate(f):
            try:
                obj = json.loads(line)
            except ValueError as e:
                print(f"Error at line {i + 1}: {e}")

    # Load dataframe
    data = pd.read_json(input_path, lines=True)

    for idx, example in data.iterrows():
        # Skip empty completions (probably due to API error)
        if example['completion_string'] is None:
            print(f"WARN: For sample {example['sample_id']} no completion is available. Consider manual recalculation.")
            continue

        prompt, completion_gt, _ = get_last_graph(example, synthetic_dataset)
        # Note, we add here the space and time leading spaces in the generated completion because the Chat like LLM can sometimes mix this up

        if not str(example['completion_string']).startswith('e'):
            generated_completion = 'e ' + example['completion_string'].lstrip()
        else:
            generated_completion = example['completion_string'].lstrip()
        # Parse the completion graph for the ground truth
        graph_gt, _ = get_anchored_completion_graph(prompt, completion_gt, synthetic_dataset, version=V_JSON)

        # Parse the completion graph for the generated completion
        # TODO try except if parsing was not successful

        graph_generated, correct = get_anchored_completion_graph(prompt, generated_completion, synthetic_dataset,
                                                                 version=V_JSON)

        if graph_generated is None and not synthetic_dataset:

            data.loc[idx, "correct_format"] = False
        else:
            data.loc[idx, "correct_format"] = True

        # Check if they are isomorphic
        graph_gt = IsoGraph(graph_gt)
        graph_generated = IsoGraph(graph_generated)
        isomorphic_completion = graph_gt == graph_generated

        if (synthetic_dataset):
            graph_matcher_generated_in_gt = DiGraphMatcher(graph_gt, graph_generated)
            graph_matcher_gt_in_generated = DiGraphMatcher(graph_generated, graph_gt)
            # Returns True if a subgraph of G1 is isomorphic to G2.
            # G2 in G1

            # one node is also a subgraph, therefore we have to check
            # if we
            subgraph_isomorphic_completion_generated_in_gt = False if graph_generated.size() == 0 else graph_matcher_generated_in_gt.subgraph_is_isomorphic()
            subgraph_isomorphic_completion_gt_in_generated = graph_matcher_gt_in_generated.subgraph_is_isomorphic()
            at_least_one_edge = check_at_least_one_edge_presence(graph_gt, graph_generated)

            regex_edge = r"e (\d+) (\d+)"
            matches = re.findall(regex_edge, completion_gt)
            count_edges = len(matches)
            if (count_edges >= 0):
                data.loc[idx, "type_isomorphic_completion"] = isomorphic_completion
                data.loc[
                    idx, "type_subgraph_isomorphic_generated_in_gt"] = subgraph_isomorphic_completion_generated_in_gt
                data.loc[
                    idx, "type_subgraph_isomorphic_gt_in_generated"] = subgraph_isomorphic_completion_gt_in_generated
                data.loc[idx, "correct_format"] = correct or subgraph_isomorphic_completion_gt_in_generated
                data.loc[idx, "at_least_one_correct_edge"] = at_least_one_edge
                data.loc[
                    idx, "correct_values"] = isomorphic_completion or subgraph_isomorphic_completion_gt_in_generated or subgraph_isomorphic_completion_generated_in_gt or at_least_one_edge

            # subgraph_isomorphic_completion_gt_in_generated or correct
        else:
            # comparison only change types
            graph_generated_only_change_types = IsoGraph(only_keep_change_types(graph_generated))
            graph_gt_only_change_types = IsoGraph(only_keep_change_types(graph_gt))
            isomorphic_completion_only_change_types = graph_generated_only_change_types == graph_gt_only_change_types

            # comparison without labels
            graph_generated_no_labels = IsoGraph(remove_all_labels(graph_generated))
            graph_gt_no_labels = IsoGraph(remove_all_labels(graph_gt))

            isomorphic_completion_no_labels = graph_gt_no_labels == graph_generated_no_labels

            data.loc[idx, "type_isomorphic_completion"] = isomorphic_completion
            data.loc[idx, "change_type_isomorphic_completion"] = isomorphic_completion_only_change_types
            data.loc[idx, "structural_isomorphic_completion"] = isomorphic_completion_no_labels

    true_false_counts_isomorphic = data['type_isomorphic_completion'].value_counts()
    true_false_counts_subgraph_generated_in_gt = data['type_subgraph_isomorphic_generated_in_gt'].value_counts()
    true_false_counts_subgraph_gt_in_completion = data['type_subgraph_isomorphic_gt_in_generated'].value_counts()

    if synthetic_dataset:
        true_false_counts_subgraph = data['type_subgraph_isomorphic_generated_in_gt'].value_counts()
        true_false_counts_subgraph_gt_in_completion = data['type_subgraph_isomorphic_gt_in_generated'].value_counts()

        print("isomorphic" + str(true_false_counts_isomorphic) + "\n")
        print("subgraph_isomorphic_generated_in_gt" + str(true_false_counts_subgraph_generated_in_gt) + "\n")

        print("correct_format" + str(data['correct_format'].value_counts()))

        print("subgraph_isomorphic_gt_in_comple" + str(true_false_counts_subgraph_gt_in_completion) + "\n")
        print("at_least_one_edge" + str(data["at_least_one_correct_edge"].value_counts()) + "\n")

    data.to_csv(output_path + '.csv', header=True)
    return data


def main(input_path: str, output_path: str, synthetic_dataset: bool = False):
   # check_for_correctness(input_path, output_path, synthetic_dataset)
    path_to_finetuning_curie = "final_output_experiment_4/results_completion_curie_6D20E51p0_curie.csv"
    path_to_finetuning_ada ="final_output_experiment_4/completion_eval_ada.csv"
    path_to_few_shot_ada = "final_output_experiment_4/final_experiment1_fewshot_ada_syntetic.csv"
    path_to_few_shot_curie = "final_output_experiment_4/final_experiment1_fewshot_curie_syntetic.csv"

    # Reading the data from CSV files
    data_finetuning_ada = pd.read_csv(path_to_finetuning_ada, sep=';')
    data_finetuning_curie = pd.read_csv(path_to_finetuning_curie, sep=';')
    data_few_shot_ada = pd.read_csv(path_to_few_shot_ada)
    data_few_shot_curie = pd.read_csv(path_to_few_shot_curie)

    # Types to test
    types_to_test = [
        "type_isomorphic_completion",
        "type_subgraph_isomorphic_generated_in_gt",
        "type_subgraph_isomorphic_gt_in_generated",
        "at_least_one_correct_edge"
    ]


    # Datasets to compare
    datasets = [
        ( data_few_shot_ada,data_finetuning_ada, "Ada"),
        ( data_few_shot_curie,data_finetuning_curie, "Curie")
    ]

    # Performing Mann-Whitney U tests
    for data1, data2, name in datasets:

        # Check the conditions for each type in the fine-tuning dataset
        isomorphic_data2 = data2['completion_best_rank_result'] == 'ISOMORPHIC'
        generated_in_gt_data2 = (data2['completion_best_rank_result'] == 'TOO_SMALL') | (
                    data2['completion_best_rank_result'] == 'ISOMORPHIC')
        gt_in_generated_data2 = (data2['completion_best_rank_result'] == 'TOO_LARGE') | (
                    data2['completion_best_rank_result'] == 'ISOMORPHIC')
        correct_edge_data2 = data2['completion_best_rank_correct_edges'] >= 1

        # Collect data based on types for few-shot dataset
        isomorphic_data1 = data1['type_isomorphic_completion']
        generated_in_gt_data1 = data1['type_subgraph_isomorphic_generated_in_gt']
        gt_in_generated_data1 = data1['type_subgraph_isomorphic_gt_in_generated']
        correct_edge_data1 = data1['at_least_one_correct_edge']

        #what to run:
        what_to_compare= [
            (isomorphic_data1, isomorphic_data2, 'Isomorphic'),
            (generated_in_gt_data1, generated_in_gt_data2, 'Generated_in_gt'),
            (gt_in_generated_data1, gt_in_generated_data2, 'Gt_in_generated'),
            (correct_edge_data1, correct_edge_data2, 'At Least One Correct Edge')
        ]

        for data1_subset, data2_subset, test_name in what_to_compare:
            # Count the number of True and False in each subset to check if ever<thing correct
            #count_true_data1 = sum(data1_subset == True)
            #count_false_data1 = sum(data1_subset == False)

            #count_true_data2 = sum(data2_subset == True)
            #count_false_data2 = sum(data2_subset == False)

            # Print the counts
            #print(f"For {test_name}:")
            #print(f"data1 - True: {count_true_data1}, False: {count_false_data1}")
            #print(f"data2 - True: {count_true_data2}, False: {count_false_data2}")

            # Two-sided
            u_statistic, p_value_two_sided = mannwhitneyu(data1_subset, data2_subset, alternative='two-sided')
            conclusion_two_sided = "likely not from the same distribution" if p_value_two_sided < 0.05 else "not enough evidence to conclude they are from different distributions"

            # One-sided (greater)
            u_statistic, p_value_greater = mannwhitneyu(data1_subset, data2_subset, alternative='greater')
            conclusion_greater = "likely greater" if p_value_greater < 0.05 else "not enough evidence to conclude greater"

            # One-sided (less)
            u_statistic, p_value_less = mannwhitneyu(data1_subset, data2_subset, alternative='less')
            conclusion_less = "likely less" if p_value_less < 0.05 else "not enough evidence to conclude less"

            print(
                f"{test_name}, {name}:\n Two-sided: U={u_statistic}, p-value={p_value_two_sided}. Conclusion: {conclusion_two_sided}\n Greater: U={u_statistic}, p-value={p_value_greater}. Conclusion: {conclusion_greater}\n Less: U={u_statistic}, p-value={p_value_less}. Conclusion: {conclusion_less}")



if __name__ == "__main__":
    """
    Executes when called as python module.
    """

    if len(sys.argv) == 3:
        synthetic_dataset = SYNTHETIC_DATASET_DEFAULT
        main(sys.argv[1], sys.argv[2], synthetic_dataset)
    elif len(sys.argv) == 4:
        synthetic_dataset = sys.argv[3] == 'True'
        main(sys.argv[1], sys.argv[2], synthetic_dataset)   
    else:
        print("Unexpected number of arguments. Call like python eval_completions.py [input_path] [output_path] [synthetic_dataset_switch (optional: defaults to False)].")


