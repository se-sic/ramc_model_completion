import math
import pickle
from lm_miner import algo
from mining.parse_utils import export_TLV
from mining.parse_utils import import_tlv_folder
from eval_utils import count_pattern_occurrences, get_rank
import os
import sys

import pandas as pd
from model_specific.simple_component_model.simple_component_model import correct_mm_edges

from metrics import *


def generate_patterns(model_id, output_folder):
  initial_pattern = "e"
  patterns, token_count, faulty_graphs, faulty_mm, duplicates = algo(model_id=model_id, sample_edges=True, initial_pattern=initial_pattern, mm_edges=correct_mm_edges)     

  # Create output folder
  os.makedirs(output_folder, exist_ok=True)
  os.makedirs(output_folder + "linegraphs/", exist_ok=True)

  with open(output_folder + "pattern_candidates.json", "w") as f:
    f.write(str([pattern.pretty_print() for pattern in patterns]))
  with open(output_folder + "pattern_candidates.pickle", "wb") as f:
    pickle.dump(patterns, f)

  # Output patterns as LG
  export_TLV([pattern.parsed for pattern in patterns], output_folder + "linegraphs/pattern_candidates_parsed.lg")
  return patterns, token_count, faulty_graphs, faulty_mm, duplicates

def eval(patterns, correct_graphs, graph_db):
  # different ranking metrics
  ranks_factorial = get_rank(patterns, pattern_score_factorial, correct_graphs, "factorial")
  ranks_probability = get_rank(patterns, pattern_score_probability, correct_graphs, "probability")
  ranks_edges = get_rank(patterns, pattern_score_edges_scaled, correct_graphs, "edges_scaled")

  # Pattern candidates with frequency (transaction-based) counted
  pattern_candidates = count_pattern_occurrences(patterns, graph_db)
  
  # Compute ranks based on the compression key
  ranks_compression = get_rank(pattern_candidates, patter_score_compresion, correct_graphs, "compression")

  return ranks_factorial, ranks_probability, ranks_edges, ranks_compression


def main(results_file, correct_graphs_path, output_folder):
  # Results folder
  path = output_folder + "/results/pattern_candidates/"
  # Load the correct graphs (so that we can count them)
  correct_graphs = import_tlv_folder(correct_graphs_path, parse_support=False)

  # Read in results file
  training_results = pd.read_csv(results_file, sep=";", header=0, keep_default_na=False)
  # iterate over all models
  for idx, row in training_results.iterrows():
    # We sleep here for a while to relax the rate-limiting of the API (TODO this should be handled by a better API wrapper in the future)
    if idx != 0:
          print("Going to sleep for a minute to relax API rate limitter...")
          # TODO sleep for a minute
          #time.sleep(60)

    # Extract unique identifier
    data_set_id = row["Id"]
    friendly_model_identifier = data_set_id + "_" + row["Strategy"] + "_" + row["Base_Model"] + "_" + str(row["Epochs"])
    
    # generate pattern candidates
    model_output_folder = path + friendly_model_identifier + "/"
    # TODO uncomment again and remove manual pattern loading
    #patterns, token_count, faulty_graphs, faulty_mm, duplicates = generate_patterns(row["model_name"], model_output_folder)
    patterns = pickle.load(open(model_output_folder + "pattern_candidates.pickle", "rb"))

    # TODO uncomment again
    # Attach generating information to the results
    #training_results.loc[idx, 'token_count'] = token_count
    #training_results.loc[idx, 'faulty_graphs'] = faulty_graphs
    #training_results.loc[idx, 'faulty_mm'] = faulty_mm
    #training_results.loc[idx, 'duplicates'] = duplicates

    # Evaluate the patterns
    components_path = output_folder + "/results/components/" + data_set_id + "/"
    graph_db = import_tlv_folder(components_path, parse_support=False)
    ranks_factorial, ranks_probability, ranks_edges, ranks_compression = eval(patterns, correct_graphs, graph_db)

    # TODO uncomment again
    # Attach ranks to the results
    #for key in ranks_factorial.keys():
    #  training_results.loc[idx, key] = ranks_factorial[key]
    #for key in ranks_probability.keys():
    #  training_results.loc[idx, key] = ranks_probability[key]
    #for key in ranks_edges.keys():
    #  training_results.loc[idx, key] = ranks_edges[key]
    for key in ranks_compression.keys():
      training_results.loc[idx, key] = ranks_compression[key]

    # write results to file, i.e., ranks of correct pattern, incorrect graph, incorrect mm, duplicates
    training_results.to_csv(output_folder + "/results/all_results.csv", sep=";", index = False)


if __name__ == "__main__":
  if len(sys.argv) == 4:
    main(sys.argv[1], sys.argv[2], sys.argv[3])
  else:
    print(f"Unexpected number of arguments: {len(sys.argv)}. At least path to training results file, graph to the correct graphs, and data set folder is needed.")