from mining.graph_lattice import Lattice, LatticeNode, Statistics
from mining.isograph import IsoGraph
from pattern_candidate import PatternBase

def get_rank(patterns, ranking_function, correct_graphs, postfix):
  # Rank them (probability x size? + other metrics)
  for pattern in patterns:
    pattern.score = ranking_function(pattern)

  # get rank of correct pattern
  patterns = sorted(patterns, key=lambda x: x.score, reverse=True)
  parsed_patterns = [IsoGraph(pattern.parsed) for pattern in patterns]
  ranks = dict()
  for correct_graph in correct_graphs:
    if correct_graph not in parsed_patterns:
      ranks["Pattern"+"_" + correct_graph.name + "_Rank_" + postfix] = -1
    else:
      ranks["Pattern"+"_" + correct_graph.name + "_Rank_" + postfix] = parsed_patterns.index(correct_graph) + 1
  
  return ranks

def count_pattern_occurrences(patterns, graph_db):
  # First create the lattice node for the subgraphs
  # extract networkx graphs from patterns
  pattern_graphs = [pattern.parsed for pattern in patterns]
  # Create lattice nodes
  lattice_nodes = [LatticeNode(subgraph) for subgraph in pattern_graphs]
  # Create lattice (this might take some time)
  lattice = Lattice(lattice_nodes)
  # Compute statistics
  stats = Statistics(graph_db, pattern_graphs, lattice=lattice)
  stats.compute_occurrences_lattice_based()
  # build up the correct pattern classes (having the compression key) and compute the ranks of the correct patterns
  return [PatternBase(lattice_node.graph, len(lattice_node.occurrences)) for lattice_node in stats.lattice.nodes]
  