#!/usr/bin/python
import json

from transformers import GPT2TokenizerFast

import os, sys
import shutil

# random for choosing random subsets and other random stuff ...
import random

# graph library
import networkx as nx

# regular expressions (for parsing)
import re

# for performance measurements
import time

# Pandas library for preparing the training data
import pandas as pd

# Parse utils
from mining.parse_utils import import_tlv_folder

# For parsing the information in the dataset names
from common import parse_dataset_params
from edgel_utils import serialize_graph, serialize_edgeL_database

from math import ceil, floor

# To split dataset in train and test dataset
from sklearn.model_selection import train_test_split

import logging

LOGGER = logging.getLogger()

TEST_SET_RATIO = 0.25

#TODO support model specific tokenization and use tiktokens instead of huggingface (much higher throughput)

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


####################### GRAPH MODIFICATIONS ########################

def reconstruct_graph(name, nodes=[], edges=[]):
    G = nx.DiGraph()
    G.name = name
    for n in nodes:
        G.add_node(n[0], **n[1])
    for e in edges:
        if ((e[2]) != None):
            G.add_edge(e[0], e[1], **e[2])
        else:
            G.add_edge(e[0], e[1])

    return G


def remove_random_edge(graph, number_of_edges_to_remove):
    G = graph.copy()
    edges = G.edges(data=True)
    # Filter edges based on label condition
    # filtered_edges = [(u, v) for u, v, data in edges if data.get('label', '').startswith(('Add_', 'Delete_'))]

    filtered_edges = []

    for u, v, data in edges:
        stri = data.get('label', '').strip('"').replace("'", '"')
        data_dict = json.loads(stri)
        if (data_dict.get('changeType').startswith(('Add', 'Delete', 'Remove', 'Change'))):

            stri_u = G.nodes[u].get('label', '')
            stri_v = G.nodes[v].get('label', '')
            _, temp_u = re.split(r"'changeType': |\"changeType\": ", stri_u)
            _, temp_v = re.split(r"'changeType': |\"changeType\": ", stri_v)
            temp_v = temp_v.replace('"', "'")
            temp_u = temp_u.replace('"', "'")
            if (data_dict.get('changeType').startswith(('Change'))):
              filtered_edges.append((u, v, data_dict.get('changeType') + "_attribute"))

            elif temp_u.startswith(("'Add'", "'Delete'", "'Remove'", "'Change'")) or temp_v.startswith(
                    ("'Add'", "'Delete'", "'Remove'", "'Change'")):


              filtered_edges.append((u, v, data_dict.get('changeType') + "_node"))
            else:  # both nodes connected to the edge are preserved


              filtered_edges.append((u, v, data_dict.get('changeType') + "_edge"))


    number_of_edges_to_remove = min(number_of_edges_to_remove, len(filtered_edges))
    number_of_edges_modified_graph = G.size() - number_of_edges_to_remove

    random_edges = random.sample(filtered_edges, number_of_edges_to_remove)

    nodes = []
    for e in random_edges:
        nodes.append((e[0], G.nodes[e[0]]))
        nodes.append((e[1], G.nodes[e[1]]))

    G.remove_edges_from(random_edges)

    completion = reconstruct_graph(G.name, nodes=nodes,
                                   edges=((e[0], e[1], graph.get_edge_data(e[0], e[1])) for e in random_edges))
    change_types = [r[2]  for r in random_edges]
    return G, completion, number_of_edges_modified_graph, number_of_edges_to_remove, change_types


def remove_random_nodes(graph, number_nodes_to_remove):
    G = graph.copy()
    nodes = G.nodes(data=True)
    # Filter nodes based on label condition
    filtered_nodes = []

    for n, data in nodes:
        stri = data.get('label', '')
        _, _, temp = stri.partition("'changeType': ")
        if (temp.startswith(("'Add'", "'Delete'", "'Remove'", "'Change'"))):
            changeType = temp.split("'")[1]
            filtered_nodes.append((n, data, changeType))

    number_nodes_to_remove = min(number_nodes_to_remove, len(filtered_nodes))

    # Select x random nodes
    random_nodes = random.sample(filtered_nodes, number_nodes_to_remove)
    random_nodes_all = random_nodes.copy()

    edges_data_list = []
    for node in random_nodes:
        # Get the edges connected to the node
        neighbors = set(G.predecessors(node[0])).union(set(G.successors(node[0])))
        for neighbor in neighbors:
            if G.has_edge(node[0], neighbor):
                edge_data = G.get_edge_data(node[0], neighbor)
                edge = (node[0], neighbor, edge_data)
                edges_data_list.append(edge)

            if G.has_edge(neighbor, node[0]):
                edge_data = G.get_edge_data(neighbor, node[0])
                edge = (neighbor, node[0], edge_data)
                edges_data_list.append(edge)

        random_nodes_all.append((neighbor, dict(G.nodes[neighbor])))

    number_nodes_prompt = G.size() - len(edges_data_list)
    for n in random_nodes:
        G.remove_node(n[0])

    change_type = [r[2] + "_node" for r in random_nodes]

    completion = reconstruct_graph(G.name, nodes=random_nodes_all, edges=edges_data_list)

    return G, completion, number_nodes_prompt, number_nodes_to_remove, change_type


def modify_graphs(graph_components):
    modified_graphs = []
    completions = []
    count_removals = {}
    change_types = {}
    count_modified_graph_edges = {}

    for graph in graph_components:
        modified_graph, completion, modified_graph_edges, removals, change_type = remove_random_edge(graph,
                                                                                                     1)  # number of esges to remove =1
        # modified_graph, completion,  removals, change_type = remove_random_nodes(graph,2)
        modified_graphs.append(modified_graph)
        completions.append(completion)
        count_removals[modified_graph.name] = removals
        change_types[modified_graph.name] = change_type
        count_modified_graph_edges[modified_graph.name] = modified_graph_edges

    return modified_graphs, completions, count_modified_graph_edges, count_removals, change_types


####################### END GRAPH MODIFICATIONS ########################


####################### SERIALIZATION STRATEGIES ########################
# Random Serialization strategy
def random_edges(graph: nx.Graph):
    nnodes = graph.number_of_nodes()
    mapping = dict(zip(range(nnodes), random.sample(range(nnodes), nnodes)))
    edges = list(graph.edges(data=True))
    random.shuffle(edges)
    return graph, edges


# as-is
def as_is(graph: nx.Graph):
    return graph, list(graph.edges(data=True))


# dfs-serialization
def dfs_edges(graph: nx.Graph):
    # Get all nodes with in-degree zero (we need to start the dfs from there)
    # roots = [node for (node, val) in graph.in_degree() if val == 0]
    return graph, [(x, y, graph.get_edge_data(x, y)) for (x, y, dir) in
                   nx.algorithms.traversal.edgedfs.edge_dfs(graph, orientation='ignore')]  # , source=roots))


###################### END SERIALIZATION STRATEGIES #######################


################# Helpers for the Training Set Generation #################################################
def cut_graph(serialized_graph: str, cut_point: int):
    '''
  Assumes the graph is given by a list of edges, separated by the new line symbol.
  Returns the first cut_point lines and all lines after cut_point lines until the last line.
  '''
    # separate by new line symbol
    lines = list(filter(None, serialized_graph.split('\n')))
    assert len(lines) >= cut_point

    # return prompt and completion
    return '\n'.join(lines[:cut_point]), '\n'.join(lines[cut_point:])


def save_prompt_completion(prompt, completion, file_path):
    with open(file_path, 'w') as f:
        f.write(prompt + '\n$$\n' + completion)

# deprecated method to create training samples by random cuts in the edgel serialization
# See modify graphs for the new approach
def read_cut_write(input_folder, output_folder, lower_cut_percentage, upper_cut_percentage):
    # Create the output folder first (if it doesn't exist yet)
    shutil.rmtree(output_folder, ignore_errors=True)
    os.makedirs(output_folder, exist_ok=False)

    for file in os.listdir(input_folder):
        with open(input_folder + '/' + file, 'r') as f:
            graph = f.read()
            # We can throw away the header here
            graph_lines = list(filter(None, graph.split('\n')))[1:]

            nb_edges = len(graph_lines)
            graph = '\n'.join(graph_lines)

            # We want at least one edge in the prompt and one in the completion. This is not possible with one edge, so we skip.
            if nb_edges == 1:
                LOGGER.info("There is a graph with only one edge. Ommiting.")
                continue

            # LOGGER.info(f'{file}, {nb_edges}, {ceil(lower_cut_percentage * nb_edges)}, {floor(upper_cut_percentage * nb_edges)} ')

            lower_cut_point = random.randint(0, ceil(lower_cut_percentage * nb_edges))
            middle_cut_point = random.randint(floor(lower_cut_percentage * nb_edges),
                                              ceil(upper_cut_percentage * nb_edges))
            upper_cut_point = random.randint(floor(upper_cut_percentage * nb_edges), nb_edges)

            # At least one edge in completion and prompt
            lower_cut_point = min(nb_edges - 1, max(1, lower_cut_point))
            middle_cut_point = min(nb_edges - 1, max(1, middle_cut_point))
            upper_cut_point = min(nb_edges - 1, max(1, upper_cut_point))

            # Cut and save
            prompt, completion = cut_graph(graph, lower_cut_point)
            output_file_name = f'{file.split(".")[0]}_{lower_cut_point}.edgel'
            save_prompt_completion(prompt, completion, f'{output_folder}/{output_file_name}')

            prompt, completion = cut_graph(graph, middle_cut_point)
            output_file_name = f'{file.split(".")[0]}_{middle_cut_point}.edgel'
            save_prompt_completion(prompt, completion, f'{output_folder}/{output_file_name}')

            prompt, completion = cut_graph(graph, upper_cut_point)
            output_file_name = f'{file.split(".")[0]}_{upper_cut_point}.edgel'
            save_prompt_completion(prompt, completion, f'{output_folder}/{output_file_name}')


def create_pandas_dataframe_from_edgel(input_folder: str):
    prompts = []
    completions = []
    for file in os.listdir(input_folder):
        with open(input_folder + '/' + file, 'r') as f:
            sample = f.read()
            prompt, completion = sample.split('$$\n')
            if not prompt or not completion:
                LOGGER.info("Empty prompt or empty completion. Skipping this sample.")
                continue

            prompts.append(prompt + 'e')
            completions.append(completion[1:] + '\n$$')

    return pd.DataFrame(zip(prompts, completions), columns=['prompt', 'completion'])


################# END Helpers for the Training Set Generation #################################################

def create_training_data(graph_components, output_dir, serialization_strategy):
    # The ground truth (change) graphs are used to generate our training and test sets. For details, look at modify_graphs

    #count_modified: how many edges are in the prompt graph, count_removals: how many are removed, so in completion
    modified_graphs, completion, count_modified, count_removals, change_types = modify_graphs(graph_components)

    uncompleted_graph_set = serialize_edgeL_database(modified_graphs, output_dir + '/edgel/',
                                                     serialization_strategy=serialization_strategy, single_file=False,
                                                     is_completion=False)

    serialize_edgeL_database(completion, output_dir + '/edgel/', serialization_strategy=serialization_strategy,
                             single_file=False, is_completion=True, uncompleted_graph_set=uncompleted_graph_set)

    # TODO here we actually need to do something else
    # read_cut_write(output_dir +'/edgel/', output_dir +'/edgel_cut/', 0.1, 0.9)
    # Create JSONS with pandas
    training_df = create_pandas_dataframe_from_edgel(output_dir + '/edgel/')

    # Count duplicates and tokens
    tokens = 0
    unique_samples = set()
    duplicates = set()
    
    complete_graphs = dict()
    for idx, row in training_df.iterrows():
        # we store all complete graphs in a dictionary to later perform operations on them.
        complete_graphs[idx] = row['prompt'] + row['completion']

        training_df['graph_id'] = training_df['prompt'].str.extract(r'# (\d+)')
        training_df['number_of_edges_graph'] = training_df['graph_id'].map(count_modified)
        training_df['number_of_removed_items'] = training_df['graph_id'].map(count_removals)
        training_df['change_type'] = training_df['graph_id'].map(change_types)

        # training_df['number_of_removed_items'] = count_removals[training_df['graph_id']]

        # TODO add number of completion edges

        # We use $$ just as a seperator symbol
        sample_txt = row['prompt'] + "$$" + row['completion']
        if sample_txt in unique_samples:
            duplicates.add(sample_txt)
        else:
            unique_samples.add(sample_txt)

    # Now, we compute number of tokens
    tokenizer_output = tokenizer(list(complete_graphs.values()))
    token_count_list = [len(sample_token_ids) for sample_token_ids in tokenizer_output['input_ids']]
    # add tokens to the datataframe
    for i, idx  in enumerate(complete_graphs.keys()):
        training_df.at[idx, 'token_count'] = token_count_list[i]
    # Total number of tokens
    tokens = sum(token_count_list)


    nb_samples = len(training_df.index)
    nb_duplicates = nb_samples - len(unique_samples)
    # write down duplicates

    with open(output_dir + '/' + serialization_strategy.__name__ + '_training_duplicates.txt', 'w') as f:
        for duplicate in duplicates:
            f.write(duplicate)

    os.makedirs(output_dir + '/finetune_training/', exist_ok=True)

    # We do not want to shuffle, because test should be "in the future" and train should be "on the past"
    train, test = train_test_split(training_df, test_size=TEST_SET_RATIO, shuffle=False)

    train.to_json(output_dir + '/finetune_training/dataset_' + serialization_strategy.__name__ + '.jsonl',
                  orient='records', lines=True)
    test.to_json(output_dir + '/finetune_training/dataset_' + serialization_strategy.__name__ + '_test.jsonl',
                 orient='records', lines=True)
    
    # Get change type statistics
    train_changed_attributes, train_added_nodes, train_removed_nodes, train_added_edges, train_removed_edges = change_type_stats_for_df(train)
    test_changed_attributes, test_added_nodes, test_removed_nodes, test_added_edges, test_removed_edges = change_type_stats_for_df(test)
   
    
    return nb_samples, tokens, nb_duplicates, len(train.index), len(test.index), train_changed_attributes, train_added_nodes, train_removed_nodes, train_added_edges, train_removed_edges, test_changed_attributes, test_added_nodes, test_removed_nodes, test_added_edges, test_removed_edges

def change_type_stats_for_df(df: pd.DataFrame):
    # Change_attribute, Add_node, Remove_node, Add_edge, Remove_edge
    changed_attributes = len(df[df['change_type'].apply(lambda change_types: "Change_attribute" in change_types)])
    added_nodes = len(df[df['change_type'].apply(lambda change_types: "Add_node" in change_types)])
    removed_nodes = len(df[df['change_type'].apply(lambda change_types: "Remove_node" in change_types)])
    added_edges = len(df[df['change_type'].apply(lambda change_types: "Add_edge" in change_types)])
    removed_edges = len(df[df['change_type'].apply(lambda change_types: "Remove_edge" in change_types)])
    return changed_attributes, added_nodes, removed_nodes, added_edges, removed_edges

def main(input_path, output_path):
    # Logger settings
    logging.basicConfig(level=logging.INFO)
    
    # Create output folder
    os.makedirs(output_path, exist_ok=True)

    # path for the output of the csv file
    results_path = output_path + '/results.csv'
    if os.path.exists(results_path):
        LOGGER.info(f"WARN: There was already a results file in {output_path}.")
        os.remove(results_path)
    # Write the header of the results file
    with open(results_path, 'w') as f:
        f.write(
            "Id;Strategy;Number_Samples;Number_Training_Samples;Number_Test_Samples;Number_Tokens;Duplicates;Training_Set_Creation_Time;train_changed_attributes;train_added_nodes;train_removed_nodes;train_added_edges;train_removed_edges;test_changed_attributes;test_added_nodes;test_removed_nodes;test_added_edges;test_removed_edges\n")

    # Loop over all datasets
    for folder_name in os.listdir(input_path):
        # Skip files in the input_path
        if not os.path.isdir(input_path + '/' + folder_name):
            continue
        # Extract dataset parameters from folder name
        nb_diffs, nb_eos, pertubation = parse_dataset_params(folder_name)
        # Generate name for the output folder
        input_dir = input_path + '/' + folder_name
        output_dir = output_path + '/' + folder_name
        LOGGER.info(f"Processing dataset: {input_dir}")
        graph_components = import_tlv_folder(input_dir, parse_support=False)
        assert len(graph_components) > 0
        assert isinstance(graph_components[0], nx.classes.digraph.DiGraph)
        assert len(graph_components[0].nodes()) > 0

        ########### Create training sets #####################
        # as-is strategy
        #start_time = time.time()
        #nb_samples, nb_tokens, nb_duplicates, nb_train, nb_test = create_training_data(graph_components, output_dir, as_is)
        #end_time = time.time()
        #with open(results_path, 'a') as f:
        #    f.write(f"{folder_name};as_is;{nb_samples};{nb_train};{nb_test};{nb_tokens};{nb_duplicates};{end_time-start_time}\n")
        
        # random strategy
        # start_time = time.time()
        # nb_samples, nb_tokens, nb_duplicates, nb_train, nb_test = create_training_data(graph_components, output_dir, random_edges)
        #end_time = time.time()
        #with open(results_path, 'a') as f:
        #    f.write(f"{folder_name};random_edges;{nb_samples};{nb_train};{nb_test};{nb_tokens};{nb_duplicates};{end_time-start_time}\n")
        
        # dfs (undirected) strategy
        start_time = time.time()
        nb_samples, nb_tokens, nb_duplicates, nb_train, nb_test, train_changed_attributes, train_added_nodes, train_removed_nodes, train_added_edges, train_removed_edges, test_changed_attributes, test_added_nodes, test_removed_nodes, test_added_edges, test_removed_edges = create_training_data(graph_components, output_dir, dfs_edges)
        end_time = time.time()
        with open(results_path, 'a') as f:
            f.write(f"{folder_name};dfs_edges;{nb_samples};{nb_train};{nb_test};{nb_tokens};{nb_duplicates};{end_time-start_time};{train_changed_attributes};{train_added_nodes};{train_removed_nodes};{train_added_edges};{train_removed_edges};{test_changed_attributes};{test_added_nodes};{test_removed_nodes};{test_added_edges};{test_removed_edges}\n")


if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2])
    else:
        LOGGER.info("Unexpected number of arguments. At least input path, output path are expected")
