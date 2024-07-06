from copy import deepcopy
from functools import reduce
# json for printing pattern objects
import json
from math import exp
from config import *
import openai
import re

from edgel_utils import parse_lm_format_graph

def prod(given_list):
  return reduce(lambda x,y:x*y,given_list)

class PatternBase():
  def __init__(self, pattern_graph, frequency):
    self.parsed = pattern_graph
    self.frequency = frequency


class Edge:
  def __init__(self, previous_prompt, tokens, token_logprobs):

    if tokens is None:
      tokens = []
    if token_logprobs is None:
      token_logprobs = []

    self.tokens = tokens
    self.token_logprobs = token_logprobs
    self.previous_prompt = previous_prompt

  def update_tokens(self, new_tokens, new_token_logprobs):
    self.tokens += new_tokens
    self.token_logprobs += new_token_logprobs

  @property
  def complete_edge_text(self):
    return  "".join(self.complete_edge_tokens)	

  @property
  def complete_edge_tokens(self):
    if not self.is_complete():
      return []
    
    new_line_index = self.get_new_line_index()
    return self.tokens[:new_line_index + 1]
    
  @property
  def complete_edge_logprobs(self):
    if not self.is_complete():
      return []
    
    new_line_index = self.get_new_line_index()
    return self.token_logprobs[:new_line_index + 1]

  def get_current_edge(self, first_token=True):
    '''
    Edges are sampled in a way, such that they already contain the next lines tokens. E.g. "e 0 1 foo Foo Bar\ne"
    This method can help split the "new edge" from an edge. I.e., it returns the incomplete edge "e".
    '''
    if not self.is_complete():
      return Edge()
    
    new_line_index = self.get_new_line_index()

    # Sometime, the API samples behind the stop token. we have to ensure in this case, that we only return the first token in the new edge.
    if first_token:
      return Edge("", self.tokens[new_line_index+1:new_line_index+2], self.token_logprobs[new_line_index+1:new_line_index+2])
    else:
      return Edge("", self.tokens[new_line_index+1:], self.token_logprobs[new_line_index+1:])

  def is_complete(self):
    # TODO probably possible to speed this up
    return '\n' in self.tokens or len([token for token in self.tokens if '\n' in token]) > 0 

  def get_text(self):
    return "".join(self.tokens)

  def get_new_line_index(self):
    if '\n' not in self.tokens:
      # first token that contains a new line symbol
      new_line_index = [idx for idx, token in enumerate(self.tokens) if '\n' in token][0]
    else:
      new_line_index = self.tokens.index('\n')
    return new_line_index

  def edge_logprob(self):
    if not self.is_complete():
      return sum(self.token_logprobs)
    else:
      new_line_index = self.get_new_line_index()
      return sum(self.token_logprobs[:new_line_index + 1])

  # TODO we have to further refactor this part. All Language Model Related stuff has to be abstracted and handled here via a stable interface.
  @classmethod
  def _get_branching_options(cls, result, branching_threshold=min_token_prob, max_n=10):
    '''
    If there are alternative token probs above a certain threshold, we use this to further branch.
    params:
      result: The GPT-3 API result
      branching_threshold: A floating value probability threshold
    '''
    branching_options = []
    top_logprobs = result['choices'][0]['logprobs']['top_logprobs']
    for token_pos, token_alternatives in enumerate(top_logprobs):
      alternative_logprobs = list(token_alternatives.values())
      alternatives = list(token_alternatives.keys())
      for alt_i, alternative in enumerate(alternatives):
        if result['choices'][0]['logprobs']['tokens'][token_pos] == alternative:
          # Only new alternatives shall be considered
          continue
        if exp(alternative_logprobs[alt_i]) > branching_threshold:
          branching_options.append(BranchingOption(token_pos, alternative, alternative_logprobs[alt_i]))

  
    # Now we return only the max_n best branching options
    branching_options.sort(key=lambda option: option.token_logprob, reverse=True)
    print([bo.token_logprob for bo in branching_options])
    
    return branching_options[:max_n]

  def sample_edges_r(self,finetuned_model ,branching_threshold=min_token_prob, token_counter=None):
    '''
    Recursively generate edges by branching "good tokens"
    '''
    edges = []
    prompt = self.previous_prompt + self.get_text()
    # Here we call the Language Model to create completions
    # TODO the LM specific logic has to be abstracted here
    result = openai.Completion.create(model=finetuned_model, prompt=prompt, max_tokens=max_edge_tokens, n=1, top_p=top_p, logprobs=logprobs, stop=stop_tokens)
    if token_counter is not None:
      token_counter.update_counter(result['usage']['total_tokens'])
    # "best edge"
    best_edge = self.copy()
    best_edge.update_tokens(result['choices'][0]['logprobs']['tokens'], result['choices'][0]['logprobs']['token_logprobs'])
    edges.append(best_edge)
    # get branching options
    branching_options = self._get_branching_options(result, branching_threshold)
    if len(branching_options) > 0 :
      for branching_option in branching_options:
        if '\n' in result['choices'][0]['logprobs']['tokens'][:branching_option.position]:
          # We need to ensure that no extensions happen beyond a new edge
          continue
        new_edge = self.copy()
        new_edge.update_tokens(result['choices'][0]['logprobs']['tokens'][:branching_option.position] + [branching_option.token], result['choices'][0]['logprobs']['token_logprobs'][:branching_option.position] + [branching_option.token_logprob])
        print(new_edge.tokens)
        edges += new_edge.sample_edges_r(finetuned_model, branching_threshold, token_counter)
    return edges

  def _check_edge(self): 
    # edge should end with new_line
    if '\n' not in self.tokens:
      print(f"WARN: Stop token found but no new line in tokens. Tokens: {self.tokens}")
      return False

    new_line_index = self.tokens.index('\n')
    next_line_index = new_line_index + 1

    # There are only two valid solutions: Either, the line completes an edge and a new edge starts or the whole pattern is complete
    if not (self.tokens[next_line_index] == 'e' or self.tokens[next_line_index] == '$$'):
      # We expect that we either have \ne at the end of a line or \n$$
      print(f"WARN: Broken pattern. Expected line to end with \ne or \n$$. Tokens: {self.tokens}")
      return False

    return True

  def edge_prob(self):
    return exp(self.edge_logprob())

  def copy(self):
    return Edge(self.previous_prompt, list(self.tokens), list(self.token_logprobs))
    
  def get_edge_label(self):
    edge_string = self.get_text()
    regex_edge = r"e (\d+) (\d+) (.+) (.+) (.+)"
    matches_edge = re.match(regex_edge, edge_string)
    if not matches_edge:
      print(f"WARN: Not an edge: {edge_string}")
      return None
    return matches_edge.group(3)

class Pattern(PatternBase):
  def __init__(self, edges=None, current_edge=Edge("", ['e'], [0]), complete=False, complete_reason=None, edge_complete=False, decays=[], edge_decays=[], edge_label_decays=[], parsed=None):
    if edges is None:
      edges = []
    else:
      self.edges = edges

    self.parsed = parsed
    self.current_edge = current_edge

    ###
    self.complete = complete
    self.complete_reason = complete_reason
    self.edge_complete = edge_complete
    self.decays = decays
    self.edge_decays = edge_decays
    self.edge_label_decays = edge_label_decays         
          
  @property
  def pattern_text(self):
    return "".join(self.tokens)

  @property
  def complete_edges_text(self):
    return "".join([edge.complete_edge_text for edge in self.edges])

  @property
  def probability(self):
    return prod(self.token_probs)

  @property
  def tokens(self):
    # return all tokens from complete edges + current tokens
    all_tokens = [token for edge in self.edges for token in edge.complete_edge_tokens]
    all_tokens += self.current_edge.tokens
    return all_tokens

  @property
  def token_probs(self):
    # return all probs tokens from complete edges + current tokens
    all_token_probs = [exp(token_logprob) for edge in self.edges for token_logprob in edge.complete_edge_logprobs]
    all_token_probs += [exp(token_logprob) for token_logprob in self.current_edge.token_logprobs]
    return all_token_probs

  @classmethod
  def from_string(cls, pattern_string):
    # parse and check pattern
    correct_syntax, _, parsed_pattern = parse_lm_format_graph(pattern_string)
    assert correct_syntax, "Pattern syntax is not correct"

    # Extract edges
    pattern_edges = pattern_string.split('\n')
    edges = [Edge("", [text+'\n'], [0]) for text in pattern_edges[:-1]]
    return Pattern(edges=edges, current_edge=Edge("", [pattern_edges[-1]], [0]), parsed=parsed_pattern)

  def extend_pattern(self, result, edge_extension=False, max_decay=max_new_edge_decay, max_edge_decay=max_edge_decay, max_edge_label_decay=max_edge_label_decay): 
    # Extend old patterns
    if edge_extension:
      patterns = self.edge_extend_pattern(result)
    else:
      patterns = self.token_extend_pattern(result)

    # check if patterns are complete
    for pattern in patterns:
      pattern._check_pattern_complete(max_decay=max_decay, max_edge_decay=max_edge_decay, max_edge_label_decay=max_edge_label_decay)

    # if no new valid patterns are generated (for what reason ever), return the old pattern as "complete" pattern
    if len(patterns) == 0:
      self.complete = True
      self.complete_reason = 'NO_EXTENSIONS'
      return [self]
    else:
      return patterns
    
  def _check_pattern_complete(self, max_decay=max_new_edge_decay, max_edge_decay=max_edge_decay, max_edge_label_decay=max_edge_label_decay):
    pattern = self.pattern_text
    prob = self.probability
    tokens = self.tokens
    token_probs =  self.token_probs

    print(f"Checking pattern {pattern} with prob {prob} and tokens {tokens} with probs {token_probs}")

    # FIRST: Token/Edge probability sufficient?
    if self.edge_complete is not None and self.edge_complete:
      if self.get_last_edge_prob() < min_edge_prob:
        self.complete = True
        self.complete_reason = 'UNLIKELY_EDGE'
        return
    else:
      if self.get_last_token_prob() < min_token_prob:
        self.complete = True
        self.complete_reason = 'UNLIKELY_TOKEN'
        return

    # SECOND: Too large, we don't want to extend further
    if self.get_num_edges() > max_edges:
      self.complete = True
      self.complete_reason = 'TOO_MANY_EDGES'

    # THIRD: Too unlikely (we don't want to continue extending the pattern)
    if prob < min_total_prob:
      self.complete = True
      self.complete_reason = 'TOO_UNLIKELY'
      return
   
    # FOURTH: The pattern is already finished
    for stop_token in pattern_stop_tokens:
      if pattern.endswith(stop_token):
        self.complete = True
        self.complete_reason = 'STOP_TOKEN'
        return

    # FIFTH: NEW_EDGE, EDGE_LABEL, and EDGE decay. The decay for the new edge is to large (we consider 3 indicators: new edge token prob drop, new edge prob drop, new edge label prob drop)
    if self.edge_complete:
      assert tokens[-1] == "e"
      new_edge_index = -1
      # All new line tokens:
      new_line_index = [index for index, item in enumerate(tokens) if "\n" in item]

    # Find the previous "new edge token" and handle the case that we have "full edge tokens"
    if len(new_line_index) >= 2 and ("e" in tokens[new_line_index[-1]] or "e" in tokens[new_line_index[-2]]):
      # in this case one of the last two edges is a "full edge token"
      edge_token_decay = token_probs[new_edge_index]/1
      edge_decay = 1
      edge_label_decay = 1      
    else:
      if len(tokens) <= 1 or len(new_line_index) < 2:
        # probably we only have one line
        p_edge_token_index = 0
        pp_edge_token_index = None
      else:
        if len(new_line_index) < 3:
          # in this case we only have two full edges and therefore the start of the pre- previous edge is token number 0
          pp_edge_token_index = 0
        else:
          pp_edge_token_index = new_line_index[-3] + 1

        p_edge_token_index = new_line_index[-2] + 1
        # because we have e \d+ \d+ [EDGE_LABEL] [SRC_LABEL] [TGT_LABEL], the edge label is within the 3rd (incl.) and 4th (excl.) space
        pp_edge_label_token_index_start = [index for index, item in enumerate(tokens[pp_edge_token_index:]) if " " in item][2]
        pp_edge_label_token_index_end = [index for index, item in enumerate(tokens[pp_edge_token_index:]) if " " in item][3] - 1
        p_edge_label_token_index_start = [index for index, item in enumerate(tokens[p_edge_token_index:]) if " " in item][2]
        p_edge_label_token_index_end = [index for index, item in enumerate(tokens[p_edge_token_index:]) if " " in item][3] - 1

      new_edge_token_prob = token_probs[new_edge_index]
      last_edge_token_prob = token_probs[p_edge_token_index]

      if pp_edge_token_index is None:
        edge_decay = 1
        edge_label_decay = 1
      else:
        pp_edge_prob = prod(token_probs[pp_edge_token_index:p_edge_token_index])
        p_edge_prob = prod(token_probs[p_edge_token_index:new_edge_index])
        pp_edge_label_prob = prod(token_probs[pp_edge_label_token_index_start:pp_edge_label_token_index_end])
        p_edge_label_prob = prod(token_probs[p_edge_label_token_index_start:p_edge_label_token_index_end])
        edge_decay = p_edge_prob/pp_edge_prob
        edge_label_decay = p_edge_label_prob/pp_edge_label_prob

      edge_token_decay = new_edge_token_prob/last_edge_token_prob

    # Add decay also to the pattern specific (for metrics)
    if self.decays is None:
      self.decays = []
    if self.edge_decays is None:
      self.edge_decays = []
    if self.edge_label_decays is None:
      self.edge_label_decays = []

    self.decays.append(edge_token_decay)
    self.edge_decays.append(edge_decay)
    self.edge_label_decays.append(edge_label_decay)

    if edge_token_decay < max_decay:
      self.complete = True
      self.complete_reason = 'NEW_EDGE_DECAY'
      return

    if edge_decay < max_edge_decay:
      self.complete = True
      self.complete_reason = 'EDGE_DECAY'
      return    

    if edge_label_decay < max_edge_label_decay:
      self.complete = True
      self.complete_reason = 'EDGE_LABEL_DECAY'
      return

    self.complete = False
    return

  #def token_extend_pattern(self, result):
  #  pattern = self.pattern
  #  pattern_probability = self.probability
  #
  #  patterns = []
  #  top_logprobs = result['choices'][0]['logprobs']['top_logprobs'][0]
  #
  #  for key in top_logprobs.keys():
  #    new_pattern = pattern + key
  #    for stop_token in edge_stop_tokens:
  #      edge_complete = new_pattern.endswith(stop_token)
  #      if edge_complete:
  #        break
  #    probability = exp(top_logprobs[key])
  #
  #    # update token and token_probs
  #    tokens = self.tokens + [key]
  #    token_probs = self.token_probs + [probability]
  #
  #    new_pattern_obj = deepcopy(self)
  #    new_pattern_obj.pattern = new_pattern
  #    new_pattern.probability =  pattern_probability*probability
  #    new_pattern.complete = False
  #    new_pattern.complete = edge_complete
  #    new_pattern.tokens = tokens
  #    new_pattern.token_probs = token_probs
  #
  #    patterns.append(new_pattern_obj)
  #      
  #
  #  return patterns

  def update_edges(self, edges):
    self.edges += edges
    last_edge = edges[-1]
    # get latest token
    self.current_edge = last_edge.get_current_edge(first_token=True)

  def edge_extend_pattern(self, result):
    patterns = []
    for edge in result:

      new_pattern = deepcopy(self)
      # update edges
      new_pattern.update_edges([edge])
      new_pattern.complete = False
      new_pattern.edge_complete = True

      patterns.append(new_pattern)
  
    return patterns
  
  def get_last_edge_prob(self):
    if len(self.edges) < 1:
      print("WARN: No edge has been completed yet for this pattern. Can not compute edge probability.")
      return None
    return self.edges[-1].edge_prob()

  def get_last_token_prob(self):
    return self.token_probs[-1]
    
  def get_num_edges(self):
    return len(self.edges)

  def pretty_print(self):
    return json.dumps(self.__dict__, indent=4, sort_keys=True, default=lambda o: '<not serializable>')

class BranchingOption:
  def __init__(self, position, token, token_logprob):
    self.position = position
    self.token = token
    self.token_logprob = token_logprob