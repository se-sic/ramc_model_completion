#!/usr/bin/python  

import mining.compute_components
from mining.parse_utils import export_TLV
from mining.parse_utils import import_tlv
import time
import os, sys
from common import parse_dataset_params
import pickle
from mining.isograph import IsoGraph


def main(input_path, output_path, correct_graphs_path, filter_correct="False"):
    # is_eval flag (for controlled experiment)
    is_eval = correct_graphs_path is not None
    if is_eval:
        filter_correct = filter_correct == "True"
    else:
        filter_correct = False

    # Create output folder
    os.makedirs(output_path, exist_ok=True)
    
    # path for the output of the csv file
    results_path = output_path + '/results.csv'
    if os.path.exists(results_path):
        print(f"WARN: There was already a results file in {output_path}.")
        os.remove(results_path)

    # Write the header of the results file
    with open(results_path, 'w') as f:
        f.write("Id;Diffs;EOs;Pertubation;Components;Nodes;Edges;Filtered;Component_Computation_Time;Filter_Correct;Correct_1_Matches;Correct_2_Matches;Correct_3_Matches\n")

    if is_eval:
        # Load the correct graphs (so that we can count them
        correct_1 = import_tlv(correct_graphs_path + '/correct_1.lg', parse_support=False)[0]
        correct_2 = import_tlv(correct_graphs_path + '/correct_2.lg', parse_support=False)[0] 
        correct_3 = import_tlv(correct_graphs_path + '/correct_3.lg', parse_support=False)[0]
        
    # Loop over all datasets
    for folder_name in os.listdir(input_path):
        # Skip files in the input_path
        if not os.path.isdir(input_path + '/' + folder_name):
            continue
        if is_eval:
            # Extract dataset parameters from folder name
            nb_diffs, nb_eos, pertubation = parse_dataset_params(folder_name)
        else: 
            nb_diffs, nb_eos, pertubation = ("None", "None", "None")
        
        # Generate name for the output folder
        input_dir = input_path + '/' + folder_name + '/diffgraphs/'
        output_dir = output_path + '/' + folder_name
        
        # Compute connected components
        start_time = time.time()
        components, nb_of_components_per_diff, filtered = mining.compute_components.get_components(input_dir, formatting=mining.compute_components.INPUT_FORMAT_NX, filtered=False)

        end_time = time.time()
        computation_time = str(end_time-start_time)
            
        if is_eval:
            # Count the number of correct graphs in the datasets
            non_matches = [component for component in components if IsoGraph(component) != IsoGraph(correct_1) and IsoGraph(component) != IsoGraph(correct_2) and IsoGraph(component) != IsoGraph(correct_3)]
            matches_1 = [component for component in components if IsoGraph(component) == IsoGraph(correct_1)]
            matches_2 = [component for component in components if IsoGraph(component) == IsoGraph(correct_2)]
            matches_3 = [component for component in components if IsoGraph(component) == IsoGraph(correct_3)]
            correct_1_matches = len(matches_1)
            correct_2_matches = len(matches_2)
            correct_3_matches = len(matches_3)
            if filter_correct:
                components = non_matches
            else:
                components = non_matches + matches_1 + matches_2 + matches_3
        else:
            correct_1_matches = ""
            correct_2_matches = ""
            correct_3_matches = ""

        # Count number of nodes
        nb_nodes = sum([len(component.nodes()) for component in components])
        nb_edges = sum([len(component.edges()) for component in components])  

        # Create output folder if it doesn't exist yet
        os.makedirs(output_dir, exist_ok=True)

        # Exports
        export_TLV(components, output_dir + '/connected_components.lg')
        
        # Write csv
        with open(results_path, 'a') as f:
            f.write(f"{folder_name};{nb_diffs};{nb_eos};{pertubation};{len(components)};{nb_nodes};{nb_edges};{filtered};{computation_time};{filter_correct};{correct_1_matches};{correct_2_matches};{correct_3_matches}\n")
        

if __name__ == "__main__":
    if len(sys.argv) == 3:
        main(sys.argv[1], sys.argv[2], None)
    elif len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Unexpected number of arguments. At least input path, output path, and optionally path to the correct graphs are expected. When pointing to correct graphs, also another argument if they should be filtered has to be given [True/False].")
