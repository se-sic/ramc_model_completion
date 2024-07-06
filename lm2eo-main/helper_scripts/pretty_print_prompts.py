import os
import re
import pandas as pd
from edgel_utils import parse_edge, V_JSON, is_header
import json
import ast


PATH='../model_completion_dataset/revision/few_shot_samples/few_shot_dataset.jsonl'
#PATH='../projects/model_completion_dataset/SMO/results/few_shot_samples/downsampled_few_shot_dataset_results.jsonl'
#PATH='../../../AAAPaper/downsampled_few_shot_dataset_batch_1_results.jsonl'
OUTPUT = '../model_completion_dataset/revision/few_shot_samples/few_shot_dataset/'
#OUTPUT='../model_completion_dataset/SMO/results/few_shot_samples/completion_results_pretty_print/'
#OUTPUT = '../../../AAAPaper'


# TOGGLE IF GENERATED COMPLETION IS AVAILABLE TO PRINT
IS_EVAL = False

def clean_up_string(input: str) -> str:
    input = input.replace('\\\'', '"') # used for quotes in strings
    input = re.sub('\'\'([^\']+)\'\'', '"\\1"', input) # double single quotes also used for quotes in strings
    input = re.sub('\'(\\w+)\'\\\\\\\\nVersion .', '\\1', input) # one-off thing, don't know how this string actuall is created but it's there, so we have to handle it.
    input = re.sub('\'(\\w+)\'\\\\nVersion .', '\\1', input) # one-off thing, don't know how this string actuall is created but it's there, so we have to handle it.
    input = re.sub('\'\\\\\\\\nVersion .', '', input) # one-off thing, don't know how this string actuall is created but it's there, so we have to handle it.
    input = re.sub('\'s ', 's ', input) # genitive s is a problem
    input = re.sub('\'open\'', 'open', input)# a one-of fix.... we should try to clean up this stuff earlier
    input = re.sub('_\'in\'', '_in', input)# a one-of fix.... we should try to clean up this stuff earlier

    return input

def json_pretty_print(input: str) -> str:
    input = input.strip('"')
    input = clean_up_string(input)
    input_dict = ast.literal_eval(input)
    return json.dumps(input_dict, sort_keys=True, indent=2)

# First Step: Read JSONL and Create Output Folder

data_frame: pd.DataFrame = pd.read_json(PATH, lines=True)
os.makedirs(OUTPUT, exist_ok=True)
data_frame.to_csv(OUTPUT + '/result.csv', header=True)


# Second Step: Extract Prompts
prompts = list(data_frame['prompt'])

# Third Step: Pretty Format
for idx, row in data_frame.iterrows():
    file_name = OUTPUT + '/' + str(row['id'])
    with open(file_name, 'w') as file:
        # Print the prompt
        for line in row['prompt'].split("\n"):
            is_edge, src_id, tgt_id, edge_label, src_label, tgt_label = parse_edge(line, version=V_JSON)
            if is_header(line):
                file.write(100*"-" + "\n")
            if is_edge:
                file.write(f"{src_id} {tgt_id} \n Edge: {json_pretty_print(edge_label)} \n Source Node: {json_pretty_print(src_label)} \n Target Node:{json_pretty_print(tgt_label)}\n")
        # Now append the completion
        file.write(50*"-" + "COMPLETION" + 50*"-" + "\n")
        for line in ("e" + row['completion']).split("\n"): # we have to add the e of the first completion edge, since we use the "e" token as a kind of start token in the edgl completions
            is_edge, src_id, tgt_id, edge_label, src_label, tgt_label = parse_edge(line, version=V_JSON)
            if is_edge:
                file.write(f"{src_id} {tgt_id} \n Edge: {json_pretty_print(edge_label)} \n Source Node: {json_pretty_print(src_label)} \n Target Node:{json_pretty_print(tgt_label)}\n")

        if IS_EVAL:
            # Now append the result of CHatCPT
            file.write(50 * "-" + "COMPLETION_STRING" + 50 * "-" + "\n")


        if row["completion_string"] is None:
            print(f"WARN: For sample {row['sample_id']} no completion is available. Consider manual recalculation.")
            continue
        if row['completion_string'] is None:
            file.write("PROBLEM\n")
            print(row['id'])
        else:

            for line in ("e " + row['completion_string']).split("\n"):  # we have to add the e of the first completion edge, since we use the "e" token as a kind of start token in the edgl completions
                is_edge, src_id, tgt_id, edge_label, src_label, tgt_label = parse_edge(line, version=V_JSON)
                if is_edge:
                    file.write(
                        f"{src_id} {tgt_id} \n Edge: {json_pretty_print(edge_label)} \n Source Node: {json_pretty_print(src_label)} \n Target Node:{json_pretty_print(tgt_label)}\n")




