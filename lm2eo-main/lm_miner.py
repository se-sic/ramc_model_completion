#!/usr/bin/python  
"""_summary_
  Implements a LLM-based pattern growth algorithm.

"""

import os
import re

from config import *
import openai

# some other utils

from functools import reduce 

# we use this to make deep copies of dicts in our edge extension
from copy import deepcopy

from transformers import GPT2TokenizerFast

# import isograph for graph isomorphism checks
from mining.isograph import IsoGraph
from edgel_utils import parse_graph, parse_lm_format_graph


# the pattern candidate interface
from pattern_candidate import Pattern

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# Load API key and setup OPEN-AI Lib
if not os.path.exists('./secrets/openai.key'):
    print("WARN: You need to provide your OpenAI API-Key in a file /secrets/openai.key")

with open('./secrets/openai.key', 'r') as f:
    api_key = f.read().strip()
    os.environ['OPENAI_API_KEY']=api_key
    openai.api_key=api_key
                   

def token_counter(result):
  # get token from result
  tokens = result['usage']['total_tokens']


class TokenCounter:
  def __init__(self, token_count):
    self.token_count = token_count
  
  def update_counter(self, new_tokens):
    self.token_count += new_tokens

def check_meta_model(graph_string, correct_mm_edges, skip_header=True):
  ''' Example for edge regex: r'e \d+ \d+ .*_bla .*_Blubb .*_Blubb'
  '''
  for edge in graph_string.split('\n'):
    if edge is None or len(edge) == 0:
      continue
    if skip_header:
      if re.match(r't # \d+', edge):
        continue
    if sum([re.match(regex_edge, edge) is not None for regex_edge in correct_mm_edges]) > 0:
      continue
    else:
      print(f"WARN: Invalid according to metamodel: Edege {edge} is not valid.")
      return False
  return True
     

def algo(model_id, sample_edges=True, initial_pattern="e", callbacks=None, mm_edges=None):
  '''
  params:
  sample_edges True, if the most probable edges should be directly be retrieved via API. Otherwise, there will be a token-wise sampling.
  '''

  init_pattern = Pattern.from_string(initial_pattern)
  patterns_old = [init_pattern]
  patterns_complete = []
  parsed_graphs = set()
  # Some metrics
  # TODO externalize these metrics (e.g., via callbacks)
  total_tokens_used = TokenCounter(0)
  faulty_graphs = 0
  faulty_mm = 0
  duplicates = 0
  while True:
    # only if there are still incomplete patterns, we continue to extend
    if len([pattern for pattern in patterns_old if not pattern.complete]) == 0:
      break

    patterns_new = []
    #print([pattern.pretty_print() for pattern in patterns_old])
    for pattern in patterns_old:

      # incomplete patterns will be extended  
      complete_edges_text = pattern.complete_edges_text
      current_edge = pattern.current_edge
      current_edge.previous_prompt = complete_edges_text

      if sample_edges:        
        new_edges = current_edge.sample_edges_r(model_id, min_token_prob, total_tokens_used)
      else:
        print("WARN: Token-wise sampling is not implemented yet.")
      # TODO the token sampling is not correctly implemented yet
      #else:
      #  res = openai.Completion.create(model=model_id, prompt=prompt, max_tokens=1, temperature=temperature, logprobs=logprobs, stop=stop_tokens)
      #  total_tokens_used.update_counter(res['usage']['total_tokens'])
      
      patterns_new += pattern.extend_pattern(new_edges, sample_edges)
    
    # store complete patterns separately
    patterns_incomplete = []
    # For the pruning in the next step, we want to ensure that the more probable pattern come first, therefore we sort here
    for pattern in sorted(patterns_new, key=lambda pt: pt.probability, reverse=True):
      # complete patterns are stored separately, since we don't want to parse them 
      if pattern.complete or pattern.edge_complete:
        correct_syntax, corrected_syntax_graph, parsed_pattern = parse_lm_format_graph(pattern.pattern_text)
        if mm_edges is not None:
          correct_meta_model = check_meta_model(corrected_syntax_graph, mm_edges)
        if not correct_syntax:
          print(f"Incorrect syntax: {corrected_syntax_graph}")
          faulty_graphs += 1
          continue
        elif not correct_meta_model:
          print(f"Incorrect graph according to meta-model: {corrected_syntax_graph}")
          faulty_mm += 1
          continue
        else:
          # check for isomorphism
          if IsoGraph(parsed_pattern) in parsed_graphs:
            duplicates += 1
            continue
          else:
            pattern.parsed = parsed_pattern
            parsed_graphs.add(IsoGraph(parsed_pattern))
            if pattern.complete:
              # reason EDGE_DECAY and EDGE_LABEL_DECAY has to be handled specially here, since then the last edge has to be thrown away -> TODO similar also for token corrections. MINOR PRIORITY, SINCE WE DO NOT USE TOKEN-WISE GENERATION IN PRACTISE
              if pattern.complete_reason in after_the_fact_correction_reasons:

                # remove last edge
                pattern_edge_removed = Pattern.from_string("\n".join(pattern.pattern_text.split('\n')[:-2])+'\ne')
                patterns_complete.append(pattern_edge_removed)
              else:
                patterns_complete.append(pattern)

              # Some complete pattern will still be extended
              if pattern.complete_reason in continue_extension_complete_reasons:
                pattern_copy = deepcopy(pattern)
                pattern_copy.complete = False
                patterns_incomplete.append(pattern_copy)
            else:
              patterns_incomplete.append(pattern)
      else:
        patterns_incomplete.append(pattern)

    patterns_old = patterns_incomplete
  
  print(f"The algorithm used {total_tokens_used.token_count} tokens.")
  print(f"There are {faulty_graphs} faulty graphs.")
  print(f"There are {faulty_mm} graphs which don't conform to the meta-model.")
  print(f"The algorithm encountered {duplicates} duplicates.")

  return patterns_old + patterns_complete, total_tokens_used.token_count, faulty_graphs, faulty_mm, duplicates