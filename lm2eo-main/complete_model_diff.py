import os
import sys
import pandas as pd
from edgel_utils import parse_lm_format_graph
from mining.isograph import IsoGraph
from model_specific.simple_component_model.simple_component_model import correct_mm_edges
from lm_miner import algo
from metrics import *


#######################
# Steps to complete a model:
# Step 1: Generate completion candidates (n best candidates)
# Step 2: Check if extension leads to a valid model (already done by algo)
# Step 3: Rank the candidates 
# Step 4: Return ranked candidates
def complete(context_string, model_id, ranking=pattern_score_edges_scaled):
      # Generate completion candidates
      patterns, _, _, _, _ = algo(model_id=model_id, sample_edges=True, initial_pattern=context_string, mm_edges=correct_mm_edges)     

      # Rank them (given the score function)
      for pattern in patterns:
        pattern.score = ranking(pattern)

      patterns = sorted(patterns, key=lambda x: x.score, reverse=True)

      # return ranked candidates
      return patterns


def eval_candidate(candidate_pattern, ground_truth_pattern_parsed, prompt, completion):
  '''
  Computes several metrics to evalute how well a candidate pattern fits the ground truth.
  '''

  lines_context = len(prompt.split('\n')) - 1
  # check to which extend it fits the ground truth (edge-wise)
  candidate_is_subgraph = IsoGraph(ground_truth_pattern_parsed).contains(candidate_pattern.parsed)
  ground_truth_is_subgraph = IsoGraph(candidate_pattern.parsed).contains(ground_truth_pattern_parsed)
  is_isomorphic = False
  is_too_small = False
  is_too_large = False
  if candidate_is_subgraph and ground_truth_is_subgraph:
      print("Isomorphic.")
      correct_edges = len(candidate_pattern.parsed.edges) - lines_context
      lines_missing = 0
      additional_lines = 0
      is_isomorphic = True
  elif candidate_is_subgraph:
      print("Too Small.")
      correct_edges = len(candidate_pattern.parsed.edges) - lines_context
      lines_missing = len(ground_truth_pattern_parsed.edges) - len(candidate_pattern.parsed.edges)
      additional_lines = 0
      is_too_small = True
  elif ground_truth_is_subgraph:
      print("Too Large.")
      correct_edges = len(candidate_pattern.parsed.edges) - lines_context
      lines_missing = 0
      additional_lines = len(candidate_pattern.parsed.edges) - len(ground_truth_pattern_parsed.edges)
      is_too_large = True
  else:
      print("No match.")
      correct_edges = 0
      # TODO check if some of the lines are correct
      lines_missing = len(ground_truth_pattern_parsed.edges) - lines_context
      additional_lines = len(candidate_pattern.parsed.edges) - lines_context
  return correct_edges, lines_missing, additional_lines, is_isomorphic, is_too_small, is_too_large

def pattern_candidate_score(lines_missing, additional_lines, is_isomorphic, is_too_small, is_too_large):
  '''
  The single patterns can not be compared directly, because it is not clearly defined yet (multi-dimensional).
  E.g., what is better: A pattern that is 2 edges too large or a pattern that is 2 edges too small.
  In this method, we clearly define this by mapping every pattern evaluation to a single valued score.
  A pattern that matches exactly will get the highest score, one that is too large is considered better than a pattern that is too small.
  Worst is a pattern that does not match at all.

  Assuming that there are at most 49999 edges too much, we can than map the pattern to an exact score using this method. TODO get rid of the magic number 50.000 by using the max number of edges in the dataset.
  '''
  # 150000/100000/50000/0 - missing
  if is_isomorphic:
    return 50001
  elif is_too_small:
    return 50000 - lines_missing
  elif is_too_large:
    return 100000 - additional_lines
  else:
    return 0 # TODO maybe pattern could be partly correct, but we do not consider this for now

def to_result_string(is_isomorphic_best, is_too_small_best, is_too_large_best):
    if is_isomorphic_best:
      result_best = "ISOMORPHIC"
    elif is_too_small_best:
      result_best = "TOO_SMALL"
    elif is_too_large_best:
      result_best = "TOO_LARGE"
    else:
      result_best = "INCORRECT"
    return result_best

#######################
# Steps to evaluate a completion:
# Step 1: Load the test data file
# For each prompt-completion pair:
# Step 2: Create ground truth graphs
# Step 3: Generate ranked list of completion candidates (n best candidates)
# Step 4: For each candidate, check to which extend it fits the ground truth 
# Step 5: Evaluate the candidates
# Step 5: For the best-ranked and the best-evaluating candidate, save the evluation metrics
def evaluate_completion(test_dataset_path, model_id):
    # create empty dataframe to store the results
    result_dataframe = pd.DataFrame(columns=['prompt_gt', 'completion_gt', 'missing_edges', 'total_edges', 'total_nodes', 'prompt_edges', 'prompt_nodes', 'nb_completion_candidates', 'completion_best_rank', 'completion_best_eval', 'completion_best_rank_score', 'completion_best_eval_score', 'completion_best_rank_correct_edges', 'completion_best_eval_correct_edges', 'completion_best_rank_lines_missing', 'completion_best_eval_lines_missing', 'completion_best_rank_additional_lines', 'completion_best_eval_additional_lines', 'completion_best_rank_result', 'completion_best_eval_result', 'best_rank', 'completion_reason_best', 'completion_reason_best_rank'])
    
    # load test data file, lines=True because data stored in "jsonl" format
    test_data = pd.read_json(path_or_buf=test_dataset_path, lines=True)

    # iterate over the test samples
    for idx, row in test_data.iterrows():
        prompt = row["prompt"]
        #lines_context = prompt.split('\n')
        completion = row["completion"]
        # create ground truth graphs
        full_serialization = prompt + completion

        # Correct serialization string and parse prompt graph
        _, _, parsed_prompt = parse_lm_format_graph(prompt)

        # Correct serialization string and parse (add header and cut off stop symbol)
        correct_syntax, _, parsed_gt = parse_lm_format_graph(full_serialization)
        if not correct_syntax:
            print("WARN: Illegal format encountered during parsing test set.")
            continue        

        # generate ranked list of completion candidates
        completion_candidates_ranked = complete(prompt, model_id)
        if len(completion_candidates_ranked) == 0:
            print("WARN: No completion candidates found.")
            print("Prompt: " + prompt)
            print("Completion: " + completion)
            continue
        nb_completion_candidates = len(completion_candidates_ranked)

        # Remove the best candidate according to the ranking
        best_rank_candidate = completion_candidates_ranked.pop(0)
        # Evaluate the best rank candidate
        correct_edges_best_rank, lines_missing_best_rank, additional_lines_best_rank, is_isomorphic_best_rank, is_too_small_best_rank, is_too_large_best_rank = eval_candidate(best_rank_candidate, parsed_gt, prompt, completion)
        # Get score for the best rank candidate
        best_rank_score = pattern_candidate_score(lines_missing_best_rank, additional_lines_best_rank, is_isomorphic_best_rank, is_too_small_best_rank, is_too_large_best_rank)


        best_pattern = best_rank_candidate
        best_score = best_rank_score
        rank_best = 0

        correct_edges_best = correct_edges_best_rank
        lines_missing_best = lines_missing_best_rank
        additional_lines_best = additional_lines_best_rank
        is_isomorphic_best = is_isomorphic_best_rank
        is_too_small_best = is_too_small_best_rank
        is_too_large_best = is_too_large_best_rank

        # Find the best candidate among all candidates
        for idx, candiate in enumerate(completion_candidates_ranked):
          correct_edges, lines_missing, additional_lines, is_isomorphic, is_too_small, is_too_large = eval_candidate(candiate, parsed_gt, prompt, completion)
          pattern_score = pattern_candidate_score(lines_missing, additional_lines, is_isomorphic, is_too_small, is_too_large)
          if pattern_score > best_score:
            rank_best = idx + 1
            best_score = pattern_score
            best_pattern = candiate
            correct_edges_best = correct_edges
            lines_missing_best = lines_missing
            additional_lines_best = additional_lines
            is_isomorphic_best = is_isomorphic
            is_too_small_best = is_too_small
            is_too_large_best = is_too_large

        result_best = to_result_string(is_isomorphic_best, is_too_small_best, is_too_large_best)
        result_best_rank = to_result_string(is_isomorphic_best_rank, is_too_small_best_rank, is_too_large_best_rank)


        # replace newlines in prompt and completion by escape sequences
        prompt = prompt.replace('\n', '\\n')
        completion = completion.replace('\n', '\\n')
        best_pattern_text = best_pattern.pattern_text.replace('\n', '\\n')
        best_pattern_rank_text = best_rank_candidate.pattern_text.replace('\n', '\\n')


        # Put the evaluation in the dataframe
        result_dataframe = pd.concat([result_dataframe, pd.DataFrame({'prompt_gt': [prompt], 'completion_gt': completion, 'missing_edges': len(parsed_gt.edges) - len(parsed_prompt.edges), 'total_edges': len(parsed_gt.edges), 'total_nodes': len(parsed_gt.nodes), 'prompt_edges': len(parsed_prompt.edges), 'prompt_nodes': len(parsed_prompt.nodes), 'nb_completion_candidates': nb_completion_candidates, 'completion_best_rank': best_pattern_rank_text, 'completion_best_eval':  best_pattern_text, 'completion_best_rank_score': best_rank_score, 'completion_best_eval_score': best_score, 'completion_best_rank_correct_edges': correct_edges_best_rank, 'completion_best_eval_correct_edges': correct_edges_best, 'completion_best_rank_lines_missing': lines_missing_best_rank, 'completion_best_eval_lines_missing': lines_missing_best, 'completion_best_rank_additional_lines': additional_lines_best_rank, 'completion_best_eval_additional_lines': additional_lines_best, 'completion_best_rank_result': result_best_rank, 'completion_best_eval_result': result_best, 'best_rank': rank_best, 'completion_reason_best': best_pattern.complete_reason, 'completion_reason_best_rank': best_rank_candidate.complete_reason})], ignore_index=True)
        
    # return the eval
    return result_dataframe


def main(test_dataset, model_id, results_folder):
    #results_file = root_folder + "/results/all_results.csv"
    # Create output folder
    os.makedirs(results_folder, exist_ok=True)
    # Results completion eval file


    results_completion_eval_file = results_folder + '/completion_eval.csv'
    # replace colons in save path by underscore (to avoid saving issues with model name which includes colons)
    results_completion_eval_file = results_completion_eval_file.replace(':', '_')

    # Read in results file
    #training_results = pd.read_csv(results_file, sep=";", header=0, keep_default_na=False)

    # iterate over all models
    #for idx, row in training_results.iterrows():
    #    # Compute the location of the test data file
    #    dataset_identifier = row["Id"]
    #    test_dataset_path = root_folder + '/results/finetune_ds/' + row["Id"]
    #    serialization_strategy = row["Strategy"]
    #    test_data_file =   test_dataset_path + '/finetune_training/dataset_' + serialization_strategy + '_test.jsonl' 
    #    model_id = row["Finetune_Id"]
    #    evaluation_results = evaluate_completion(test_data_file, model_id)
    #    evaluation_results.to_csv(results_completion_eval_file, sep=";", index = True)
    
    evaluation_results = evaluate_completion(test_dataset, model_id)
    evaluation_results.to_csv(results_completion_eval_file, sep=";", index = True)
    
    #training_results.to_csv(results_file, sep=";", index = False)

        
if __name__ == "__main__":
  if len(sys.argv) == 4:
    main(sys.argv[1], sys.argv[2], sys.argv[3])
  else:
    print(f"Unexpected number of arguments: {len(sys.argv)}. At least path to the test file, the id of the fine-tuned model, and the path to the results folder is necessary.")