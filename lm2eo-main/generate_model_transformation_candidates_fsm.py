from mining.parse_utils import import_tlv_folder
from eval_utils import get_rank

import sys

import pandas as pd

from pattern_candidate import PatternBase


def main(results_file, correct_graphs_path, output_folder):
  # Thresholds
  t = [6,8,10]
  for t_ in t:
    eval_pattern_candidates(results_file, correct_graphs_path, output_folder, t_)

def eval_pattern_candidates(results_file, correct_graphs_path, output_folder, t):
  # Results folder
  pattern_candidate_path = f"{output_folder}/results/pattern_candidates/{t}/"
  # Load the correct graphs (so that we can count them)
  # TODO we use subgraph isomorphism algos which only work with undirected. Therefore, we can only compare for undirected here.
  correct_graphs = import_tlv_folder(correct_graphs_path, is_directed=False, parse_support=False)

  # Read in results file
  ds_list = pd.read_csv(results_file, sep=";", header=0, keep_default_na=False)
  # iterate over all models
  for idx, row in ds_list.iterrows():
    # Get pattern candidates
    dataset_name = row['Id']
    pattern_candidates = import_tlv_folder(f"{pattern_candidate_path}/{dataset_name}/mining/", is_directed=False, parse_support=True)
    pattern_candidate_objects = [PatternBase(key, value) for key, value in pattern_candidates.items()]

    # different ranking metrics
    ranks_compression= get_rank(pattern_candidate_objects, lambda x: (x.frequency-1)*(len(x.parsed.edges())+len(x.parsed.nodes())), correct_graphs, "compression")
    ranks_freq = get_rank(pattern_candidate_objects, lambda x: x.frequency, correct_graphs, "freqeuncy")

    # Write the ranks into the dataframe
    for key in ranks_compression.keys():
       ds_list.loc[idx, key] = ranks_compression[key]

    for key in ranks_freq.keys():
      ds_list.loc[idx, key] = ranks_freq[key]

    # write results to file, i.e., ranks of correct pattern, incorrect graph, incorrect mm, duplicates
    ds_list.to_csv(output_folder + "/results/all_results.csv", sep=";", index = False)


if __name__ == "__main__":
  if len(sys.argv) == 4:
    main(sys.argv[1], sys.argv[2], sys.argv[3])
  else:
    print(f"Unexpected number of arguments: {len(sys.argv)}. At least path to training results file, graph to the correct graphs, and data set folder is needed.")