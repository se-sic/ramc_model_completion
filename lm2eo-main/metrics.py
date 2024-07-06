import math
from pattern_candidate import PatternBase


def pattern_score_factorial(pattern: PatternBase):
    '''
    Score of the pattern based on the number of nodes and the probability of the pattern
    '''
    return math.factorial(len(pattern.parsed.nodes())) * pattern.probability

def pattern_score_probability(pattern: PatternBase):
    '''
    Score of the pattern based on the probability of the pattern
    '''
    return pattern.probability

def pattern_score_edges_scaled(pattern: PatternBase):
    '''
    Score of the pattern based on the number of edges and the probability of the pattern
    '''
    return pattern.probability * (len(pattern.parsed.edges()))

def patter_score_compresion(pattern: PatternBase):
    '''
    Score of the pattern based on the compression of the pattern
    '''
    return (pattern.frequency-1)*(len(pattern.parsed.edges())+len(pattern.parsed.nodes()))