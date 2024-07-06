#!/usr/bin/python  

# TODO this class should be refactored (more object oriented). I.e., we have A graph and edge classes and each has a parsing and serialization cababilities.

import os
import shutil
import sys
from typing import Any, Callable, Dict, List, Set, Tuple

# graph library
import networkx as nx

# regular expressions (for parsing)
import re

# logging
import logging
import json
import re
LOGGER = logging.getLogger("EdgeL-Utils")

# FORMAT VERSION
V_ORIGINAL = 1 # Supports only string without whitespace characters for edge and node labels
V_JSON = 2 # Supports JSON labels for edges and nodes. Furthermore, "_" labels can be used to not repeat any label

DUMMY_NODE_LABELS = ["_", "\"{}\"", "{}"] # Node labels used to avoid repetition in edge serializations


############################ v1 -> v2 ######################################
def transform_token_node_attributes(token):
  change_type, obj_type = token.split('_')
  return {
    'changeType': change_type,
    'type': "object",
    'className': obj_type,
    'attributes': {
      'id': '',
      'name': '',
      'fqn': '',
      'maximumCountInCar': '',
      'coachAttributeValues': []
    }
  }
def replace_last_occur(text):
  return text.rsplit("}", 1)[0] + "\\}" + text.rsplit("}", 1)[1] if "}" in text else text

def transform_v1_to_v2(file_path_source, file_path_destination):

  all_transformed_data = []
  e=0

  with open(file_path_source, 'r') as f:
    transformed_data_one_example = {}
    for line in f:

        original_data = json.loads(line.strip())
        prompt_id = original_data["id"]

        for part in ["prompt","completion" , "completion_string" ]:
                data = original_data[part]
                edges = [line.strip() for line in data.split("\n") if line.strip()]
                if (edges[0].startswith("t")):
                  edges.pop(0)


                transformed_edges = []

                # Loop through the edges to transform them
                for edge in edges:
                      tokens = re.split(r'\s+', edge)

                      if(tokens==["$$"] ):
                        transformed_edges.append(tokens[0])
                      elif ( tokens==["---"] ):
                        transformed_edges.append(tokens[0])
                        transformed_edges.append("t # " + json.dumps(prompt_id))


                      elif (part=="completion_string" and not re.fullmatch(r'(e\s+)?(\d+) (\d+) (\w+) (\w+) (\w+)', edge)):
                        #usually at the very end, cpt produced too many edges
                        print("not syntactically correct completion string")


                      elif ( tokens!=["e"]):
                        if ("e" not in tokens ):
                          tokens.insert(0, "e")
                          e += 1
                          print("e for edge is missing")
                        edge_type =  tokens[3].split('_')
                        dict_edge = {}
                        dict_edge["changeType"] = edge_type[0]
                        dict_edge["type"] = "reference"
                        dict_edge["referenceTypeName"] = edge_type[1]

                        node_first_new_attribute = transform_token_node_attributes(tokens[4])
                        node_second_new_attribute = transform_token_node_attributes(tokens[5])

                        transformed_edge = 'e ' + tokens[1] + ' ' + tokens[2] + ' \"' + json.dumps(
                          dict_edge) +  '\" \"' + json.dumps(node_first_new_attribute) +  '\" \"' + json.dumps(
                          node_second_new_attribute) + '\"'

                        transformed_edges.append(transformed_edge)

                transformed_data_one_example[part] = "\n".join(transformed_edges)

        final_prompt={}
        final_prompt["sample_id"]=  prompt_id
        final_prompt["change_type"] ="None"
        final_prompt["detail_type"] = "None"
        final_prompt["context_type"] = "None"
        final_prompt["similar_few_shot"] = "None"
        final_prompt["comment"] = "None"
        final_prompt["id"] =  prompt_id
        final_prompt["prompt"]=   transformed_data_one_example["prompt"]
        final_prompt["completion"] =   transformed_data_one_example["completion"]
        final_prompt["completion_string"] = transformed_data_one_example["completion_string"]
        all_transformed_data.append(json.dumps(final_prompt))

    result = "\n".join(all_transformed_data)
    print(e)
    with open(file_path_destination, 'w') as f:
      f.write(result)



  ####################### edgeL (edge list) serialization ###################
# TODO implement other strategies
def serialize_graph(graph: nx.Graph, serialization_strategy: Callable[[nx.Graph],Tuple[nx.Graph, List[Tuple]]], is_completion: bool, serialized_nodes=set(), version=V_ORIGINAL):
  graph_string = ''
  # Add header
  if not is_completion:
    graph_string += f't # {graph.name}\n'
  # Serialize edges
  graph, edges = serialization_strategy(graph)

  for edge in edges:
    graph_string+=serialize_edge(edge, graph, serialized_nodes, version=version)+'\n'
  
  return graph_string, serialized_nodes


def serialize_edge(edge: Tuple, graph: nx.Graph, serialized_nodes=set(), version=V_ORIGINAL):
  if 'label' not in edge[2].keys():
    LOGGER.warn("Unlabeled edges in graph data for graph %s." % graph.name)
    label = "UNKNOWN_LABEL"
  else:
    label = edge[2]['label']

  graph_nodes = graph.nodes(data=True)
  
  if version == V_ORIGINAL:
    dummy_label = "_" 
  elif version == V_JSON:
    dummy_label = "\"{}\""
  else:
    LOGGER.warn(f"Unkown EdgL version: {version}")
    dummy_label = "_"

  if int(edge[0]) in serialized_nodes:
    src_label = dummy_label
  elif 'label' in graph_nodes[int(edge[0])].keys():
    src_label = graph_nodes[int(edge[0])]['label']
    serialized_nodes.add(int(edge[0]))
  else:
    LOGGER.warn("Unlabeled nodes in graph data for graph %s." % graph.name)
    src_label = "UNKNOWN_LABEL"
    serialized_nodes.add(int(edge[0]))
    
  if int(edge[1]) in serialized_nodes:
    tgt_label = dummy_label
  elif 'label' in graph_nodes[int(edge[1])].keys():
    tgt_label = graph_nodes[int(edge[1])]['label']
    serialized_nodes.add(int(edge[1]))
  else:
    LOGGER.warn("Unlabeled nodes in graph data for graph %s." % graph.name)
    tgt_label = "UNKNOWN_LABEL"
    serialized_nodes.add(int(edge[1]))

  return f'e {edge[0]} {edge[1]} {label} {src_label} {tgt_label}'



  
def serialize_edgeL_database(nx_database: List[nx.Graph], save_path: str, serialization_strategy: Callable[[nx.Graph],Tuple[nx.Graph, List[Tuple]]], single_file=False, is_completion =False, uncompleted_graph_set=dict()):
  ''' For a given database of networkX graphs, serializes them in the edgeL format and saves them.
  If single_file is True, all graphs are written to one file separated by an empty line. If single_file is False, there is one file per graph.
  '''
  # if it is the modified graph, we have a new serialisation strategy
  if (not is_completion):
      shutil.rmtree(save_path, ignore_errors=True)
  os.makedirs(save_path, exist_ok=True)

  if single_file:
    graph_db_string = ''
  for graph in nx_database:
    if is_completion:
      serialized_graph,_ = serialize_graph(graph, serialization_strategy=serialization_strategy,is_completion=is_completion, serialized_nodes=uncompleted_graph_set[graph.name] )
    else:
      serialized_graph,graph_set = serialize_graph(graph, serialization_strategy=serialization_strategy,is_completion=is_completion, serialized_nodes=set())
      uncompleted_graph_set[graph.name]= graph_set
      
    if single_file:
      graph_db_string += ('$$\n' if is_completion else '') + serialized_graph +  '\n\n'
    else:
      with open(f'{save_path}/{graph.name}.edgel', 'a') as f:
        f.write( ('$$\n' if is_completion else '') + serialized_graph)
  if single_file:
    with open(f'{save_path}/database.edgel', 'a') as f:
        f.write(graph_db_string)

  return uncompleted_graph_set
################# END edgeL serialization #################################################

####################### BEGIN: edgeL (edge list) parsing ###################
def parse_graph(graph_string, synthetic_dataset=False, directed=True, version: int=V_ORIGINAL,
                parse_labels_json=False, reduce_labels=False, serialized_ids: Set[int] = None) -> Tuple[bool, nx.Graph]:
  ''' 
  Parses a graph in the form of a list of edges separated by a new line symbold. Each edge has a label and the id of source and target node as well as source and target node labels. 
  Every graph starts with a header that includes the id or name of the graph. Example:
  t # 0
  e 0 1 c A B
  e 0 2 b A C
  e 1 2 a B C

  Since node labels are redundant, the consistency has to be checked, if a node appears in multiple edges.
  
  This method also supports adding ids of serialized nodes to the labels (to ensure that they are matched in a graph matching.))
  To enable this, a set with the corresponding node id's has to be given as "serialized_ids".
  
  returns True, Graph if the graph could be parsed correctly
  '''
  if directed:
    G = nx.DiGraph()
  else:
    LOGGER.error("Only DiGraphs supported currently. Use directed=True.")
    raise Exception("Only DiGraphs supported currently.")
  
  # t # graph_name/id
  regex_header = r"t # (.*)"

  lines = graph_string.split('\n')
  matches_header = re.match(regex_header, lines[0])
  
  if not matches_header:
    return False, None
  
  G.name = matches_header.group(1)

  is_correct=True
  for line in lines[1:]:
    add_edge = True
    if (line == "$$"):
      continue
    if len(line) == 0:
      break
    correct, src_id, tgt_id, edge_label, src_label, tgt_label = parse_edge(line, version=version, parse_labels_json=parse_labels_json, reduce_labels=reduce_labels, serialized_ids=serialized_ids)
        
    if not correct:
      is_correct= False
      add_edge =False
      if not synthetic_dataset:
        LOGGER.warning(f"Incorrect format. Couldn't parse edge: {line}")

    # add source node if not available
    if src_id in G.nodes:
      # verify consistency
      if src_label not in DUMMY_NODE_LABELS and not G.nodes(data=True)[src_id]['label'] == src_label:

        if not synthetic_dataset:
          LOGGER.warning(f"Nodes labels not consistent {G.nodes(data=True)[src_id]['label']} and {src_label}")
          return False, None
        else:
          is_correct = False
          add_edge = False

    elif add_edge:
      # add node
      G.add_node(src_id, label=src_label)


    # add target node if not available
    if tgt_id in G.nodes:
      # verify consistency
      if tgt_label not in DUMMY_NODE_LABELS and not G.nodes(data=True)[tgt_id]['label'] == tgt_label:
       if not synthetic_dataset:
         LOGGER.warning(f"Nodes labels not consistent {G.nodes(data=True)[tgt_id]['label']} and {tgt_label}")
         return False, None
       else:
          is_correct = False
          add_edge= False
    elif add_edge:
      # add node
      G.add_node(tgt_id, label=tgt_label)

    # add edge
    if (add_edge):
      G.add_edge(src_id, tgt_id, label=edge_label)
  return is_correct, G
 
def is_header(input: str):
  regex_header = r"t # (.*)"

  matches_header = re.match(regex_header, input)
  
  if not matches_header:
    return False 
  return True

def parse_edge(edge_string: str, version: int=V_ORIGINAL, parse_labels_json=False, reduce_labels=False, serialized_ids: Set[int] = None):
  # e src_id tgt_id edge_label src_label tgt_label
  if version == V_ORIGINAL:
    regex_edge = r"e (\d+) (\d+) (.+) (.+) (.+)"
  elif version == V_JSON:
    regex_edge = r"e (\d+) (\d+) (\"?\{.+\}\"?) (\"?\{.+\}\"?|_) (\"?\{.+\}\"?|_)"
  else:
    LOGGER.warn(f"Version not supported: {version}")   
    return False, None, None, None, None, None
  matches_edge = re.match(regex_edge, edge_string)
  
  if not matches_edge:
    return False, None, None, None, None, None
  
  src_id = int(matches_edge.group(1))
  tgt_id = int(matches_edge.group(2))

  edge_label = str(matches_edge.group(3))
  src_node_label = str(matches_edge.group(4))
  tgt_node_label = str(matches_edge.group(5))
  
  # Special handling for V_JSON version of serialization
  if version == V_JSON:
    #underscores are mapped to {} for easier handling
    if src_node_label == "_":
      src_node_label = "\"{}\""
    if tgt_node_label == "_":
      tgt_node_label = "\"{}\""
    

    # Special handling in case node ids have to be added to the node labels:
    src_add_attributes = dict()
    tgt_add_attributes = dict() 
    if serialized_ids is not None and len(serialized_ids) > 0 and (src_id in serialized_ids or tgt_id in serialized_ids):
      if src_id in serialized_ids:
        src_add_attributes['serialized_node_id'] = src_id
      
      if tgt_id in serialized_ids:
        tgt_add_attributes['serialized_node_id'] = tgt_id     
    
    # Transform the edge and node labels to valid json (due to historical reason, the V_JSON is not valid json yet)  
    if parse_labels_json:
      edge_label = ChangeGraphEdge.to_json(edge_label, reduce=reduce_labels)
      if not src_node_label in DUMMY_NODE_LABELS:
        src_node_label = ChangeGraphNode.to_json(src_node_label, reduce=reduce_labels, add_fields=src_add_attributes)
      else:
        src_node_label = "{}"
      if not tgt_node_label in DUMMY_NODE_LABELS:
        tgt_node_label = ChangeGraphNode.to_json(tgt_node_label, reduce=reduce_labels, add_fields=tgt_add_attributes)
      else:
        tgt_node_label = "{}"
        
  elif version == V_ORIGINAL:
    # Special handling in case node ids have to be added to the node labels:
    if serialized_ids is not None and len(serialized_ids) > 0 and (src_id in serialized_ids or tgt_id in serialized_ids):
      if src_id in serialized_ids:
        src_node_label = str(src_id) + "_" + src_node_label
      if tgt_id in serialized_ids:
        tgt_node_label = str(tgt_id) + "_" + tgt_node_label
  
  return True, src_id, tgt_id, edge_label, src_node_label, tgt_node_label


def parse_lm_format_graph(graph_string):
  '''
  For training the language model and reading out motifs, we throw away some unnecessary information in the edgeL format.
  We correct for this here and parse this format.

  returns syntax_correct, corrected_syntax, parsed_graph. Syntax correct is True, if the graph is correct syntax, corrected_syntax contains the possibly parsable string. parsed_graph contains the networkx graph.
  '''
  corrected_syntax_graph = "t # 0\n" + graph_string[:graph_string.rindex('\n')+1]# We have to add header and cut off the additional characters (end of graph and new edge) here
  correct_syntax, parsed_pattern = parse_graph(corrected_syntax_graph) 
  return correct_syntax, corrected_syntax_graph, parsed_pattern

def get_prompt_graphs(prompt: str, seperator: str ="$$\n---\n", synthetic_dataset: bool =False) -> List[str]:
  """
  Splits the given prompt by the given "graph seperator" and corrects the last graph in the list.

  Args:
      prompt (str): A list of serialized graphs for language model based completion.
      seperator (str, optional): The seperator between serialized (partial) graphs.. Defaults to "$$\n---\n".

  Returns:
      List[str]: A list of serialized (partial) graphs.
  """
  
  # Split the prompt (which probably has more than one graph)
  partial_graphs = prompt.split(seperator)
  # Last graph has some additional beginning new line and "e" prompt, which we cut of
  if (not synthetic_dataset):
    partial_graphs[-1] = partial_graphs[-1][:-2] # cut off new edge prompt
  
  return partial_graphs

def get_used_node_ids(graph_serialization: str, version=V_ORIGINAL) -> Set[int]:
  """
  Determines node ids used in a serialized (partial) graph.

  Args:
      graph_serialization (str): The serialized (partial) graph.

  Returns:
     Set(int): A set of used node ids.
  """
  used_node_ids = set()
  for line in graph_serialization.split('\n'):
      is_edge, src_id, tgt_id, edge_label, src_label, tgt_label = parse_edge(line, version=version)
      if is_edge:
          used_node_ids = used_node_ids.union({src_id, tgt_id})
  return used_node_ids

def get_prompt_only_nodes(prompt: str, completion: str, version=V_ORIGINAL) -> Set[int]:
  """For a (context) graph (serialized) and a possible completion for that serialized graph, this method determines
  the nodes that are only part of the prompt. but not of the completion.

  Args:
      prompt (str): The serialized context graph.
      completion (str): The serialized completion (candidate).

  Returns:
      Set[int]: The set of node ids of nodes that are only part of the the prompt, but not part of the completion.
  """
  return  get_used_node_ids(prompt, version) - get_used_node_ids(completion, version)


def get_anchor_node_ids(prompt: str, completion: str, version=V_ORIGINAL) -> Set[int]:
  """
  For a (context) graph (serialized) and a possible completion for that serialized graph, this method determines
  the anchor nodes (by their ids), i.e., the nodes in the context graph/prompt that the completion is attached to.
  
  This can be used to reduce graph comparison to the graph completions, by ensuring that the anchor nodes are correctly matched.

  Args:
      prompt (str): The serialized context graph.
      completion (str): The serialized completion (candidate).

  Returns:
      Set(int): A set of anchor node ids.
  """
  return get_used_node_ids(completion, version).intersection(get_used_node_ids(prompt, version))

def get_anchored_completion_graph(prompt: str, completion: str,synthetic_dataset: bool= False, version=V_ORIGINAL) -> nx.Graph:
  """
  
  Parses the graph given by the prompt (context) and the completion but returns only the nodes of the completion. 
  To ensure also correct "glueing" to the context, the ids of the context graph are added to the label.

  Args:
      prompt (str): The serialized prompt (partial) graph (or context graph).
      completion (str): A serialized completion candiate for the prompt graph.
      version (int, optional): The version of the edgel format used for prompt and completion. Defaults to V_ORIGINAL.

  Returns:
      nx.Graph: A networkx graph for the completion. 
  """
  
  # Firstly, we need a way do determine "new nodes" and "anchor nodes"
  anchor_node_ids = get_anchor_node_ids(prompt, completion, version=version)
  prompt_only_nodes = get_prompt_only_nodes(prompt, completion, version=version)
  
  full_graph = prompt + "\n" + completion

  # Secondly, we need a way to augment the labels for anchor nodes to ensure that anchor nodes are matched correctly
  # This functionality is available in the parse graph function via the attribute "serialized_ids"
  # Thirdly, we "reduce" the labels. Basically, attributes are very unlikely to be 100% correct. We compare these manually. 
  # Automatically, we can check structural correctness and type correctnes. We therefore reduce the label to only carry type information.
  # This "reduction" is also available as a feature in the "parse_graph" function.
  correct, graph = parse_graph(full_graph, synthetic_dataset, directed=True, version=version, parse_labels_json=True, reduce_labels=True, serialized_ids=anchor_node_ids)
  if not correct and not synthetic_dataset:
    LOGGER.warn(f"Incorrect graph completion {completion}.")

    return None

  # Fouthly, after parsing the entire graph, we remove all nodes that are only part of the prompt (i.e., not part of the completion).
  graph.remove_nodes_from(prompt_only_nodes)
  return graph, correct
  
################# END edgeL parsing #################################################

################# BEGIN JSON Graph Label Parser #####################################
import ast, json

# TODO move this to another module and try to better integrate with networkx
# TODO probably even bettern than working with json strings would be to work with the classes directly as data and defining __eq__ method.
class ChangeGraphElement():
  def __init__():
    pass
  
  @classmethod
  def from_string(cls, input: str):
    input = input.strip('"')
    input = clean_up_string(input)
    try:
      obj_dict = ast.literal_eval(input)
    except Exception as e:
      LOGGER.error(f"Object couldn't be parsed: {input}")
      raise e
    return cls(**obj_dict)

class ChangeGraphEdge(ChangeGraphElement):
  def __init__(self, changeType: str=None, type: str=None, referenceTypeName: str=None, attributes: Any=None):
    self.changeType = changeType
    self.type = type
    self.referenceTypeName = referenceTypeName
    self.attributes = attributes
    
  @classmethod
  def to_json(cls, original_edge_string: str, reduce=True) -> str:
    """
    
    Parse the input edge string and and transform it to valid json.
    This method also outputs a valid json, removes unneccesary quotes.

    Args:
        original_edge_string (str): The original edge label

    Returns:
        str: The JSON edge label
    """
    change_graph_edge = cls.from_string(original_edge_string)
    
    if reduce:
      if change_graph_edge.type == "attribute": # We have a Change Attribute Edge
        change_graph_edge = {"changeType": change_graph_edge.changeType, "type": change_graph_edge.type} 
      else:
        change_graph_edge = {"changeType": change_graph_edge.changeType, "referenceTypeName": change_graph_edge.referenceTypeName} 

    return json.dumps(change_graph_edge, sort_keys=True)
      
  
class ChangeGraphNode(ChangeGraphElement):
  def __init__(self, changeType: str=None, type: str=None, className: str=None, attributeName: str=None, attributes: Any = None, valueBefore: str = None, valueAfter: str = None):
    self.changeType = changeType
    self.type = type
    self.className = className
    self.attributeName = attributeName
    self.attributes = attributes
    self.valueBefore = valueBefore
    self.valueAfter = valueAfter
    
  def _to_json(self, reduce=True, additional_keep_fields: List[str] = []):
    change_graph_node = self
    if reduce:
      if self.changeType == "Change": # We have a Change Attribute Node
        change_graph_node = {"changeType": self.changeType, "className": self.className, "attributeName": self.attributeName} 
      else:
        change_graph_node = {"changeType": self.changeType, "className": self.className}

      for field in additional_keep_fields:
        change_graph_node[field] = getattr(self, field)
        
    return json.dumps(change_graph_node, sort_keys=True)
    
  @classmethod
  def to_json(cls, original_node_string: str, reduce=True, add_fields: Dict=dict()) -> str:
    """
    
    Parse the input node string and extract changeType, type, className, and attributeName if applicable.
    This method also outputs a valid json, removes unneccesary quotes.

    Args:
        original_node_string (str): The original node label
        reduce (bool, Optional): True, if only specific values should be serialized.
        add_fields (Dict, Optional): A dictionary of attributes and values that should be added to the serialization.


    Returns:
        str: A probably reduced JSON node label, throwing away attribute specific information.
    """
    change_graph_node = cls.from_string(original_node_string)
    for key, value in add_fields.items():
      setattr(change_graph_node, key, value)

    return change_graph_node._to_json(reduce=reduce, additional_keep_fields=list(add_fields.keys()))
  
def clean_up_string(input: str) -> str:
    input = input.replace('\\\'', '"') # used for quotes in strings
    input = re.sub('\'\'([^\']+)\'\'', '"\\1"', input) # double single quotes also used for quotes in strings
    input = re.sub('\'(\\w+)\'\\\\\\\\nVersion .', '\\1', input) # one-off thing, don't know how this string actuall is created but it's there, so we have to handle it.
    input = re.sub('\'(\\w+)\'\\\\nVersion .', '\\1', input) # one-off thing, don't know how this string actuall is created but it's there, so we have to handle it.
    input = re.sub('\'\\\\\\\\nVersion .', '', input) # one-off thing, don't know how this string actuall is created but it's there, so we have to handle it.
    input = re.sub('\'s ', 's ', input) # one-off thing
    input = re.sub('\'stack\' ', 'stack ', input) # one-off thing
    input = re.sub('\'selected\' ', 'selected ', input) # one-off thing
    input = re.sub(': \'ecore::EDoubleObject\'', ': ecore::EDoubleObject', input) # one-off thing
    input = re.sub('\'in\' ', 'in ', input) # one-off thing


    return input
  
  ################# END JSON Graph Label Parser #####################################



if __name__ == '__main__':

  if __name__ == "__main__":
    """
    Executes when called as python module.
    """
    if len(sys.argv) == 3:

      transform_v1_to_v2(sys.argv[1], sys.argv[2])
    else:
      print("Unexpected number of arguments. Call like python eval_completions.py [input_path] [output_path].")

