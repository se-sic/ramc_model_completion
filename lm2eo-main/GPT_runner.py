"""
    A runner script for the GPT API.
    This script loads the datasets and generated completions for the given prompts
    and adds them to the datasets.
"""
# TODO write a wrapper for LM Apis (abstraction layer that also allows us to use multiple LLMs interchangeabliy)
# TODO use adapter pattern or similar to make language model interchangeable
import sys, os
import logging
import openai
from openai import Completion
import pandas as pd
import numpy as np
from config import model_prices, load_openai_key
from common import ask_for_proceed
import time

LOGGER = logging.getLogger("LLM-Runner")

# Sleep period for rate limiter
SLEEP_PERIOD = 50
SLEEP_EVERY = 5 # sleep every x samples

CHAT_MODELS = ["gpt-4"]
COMPLETION_MODELS = ["text-davinci-003", "code-davinci-002", "text-ada-001"]

EDGE_STOP_TOKEN = "\n"
GRAPH_STOP_TOKEN = "\n\n"

MAX_EDGE_TOKENS = 450
TEMPERATURE = 0 # The randomness of the answers
TOP_P = 0 # The top x(=0) percent of tokens are considered, 0 means no randomness.
LOGPROPS = 5 # there are the 5 best tokens at most -- for more, sales can be contacted

CHAT_MODEL_INSTRUCTION = """
You are an assistant that is given a list of change graphs in an edge format. That is, the graph is given edge by edge. The graphs are directed, labeled graphs. An edge is serialized as
"e src_id tgt_id edge_label src_label tgt_label"

Labels are dictionaries. If a node appears in more than one edge, the second time it appears it is replaced by "_" to avoid repetition. 

E.g.:
e 0 1 a b bar
e 1 2 bla _ foo

The second edge here would be equivalent to:
"e 1 2 bla bar foo"

There are some change graphs given as examples. Graphs are separated by "\n\n$$\n---\n".

The last graph in this list of graphs is not yet complete. Exactly one edge is missing. 
Your task is it, to complete the last graph by guessing the last edge. You can guess this typically by looking at the examples and trying to deduce the patterns in the examples. Give this missing edge in the format
"e src_id tgt_id edge_label src_label tgt_label". Note that the beginning "e" is already part of the prompt.
"""

CHAT_MODEL_INSTRUCTION_MULTI_EDGE = """
You are an assistant that is given a list of change graphs in an edge format. That is, the graph is given edge by edge. The graphs are directed, labeled graphs. An edge is serialized as
"e src_id tgt_id edge_label src_label tgt_label"

Labels are dictionaries or concatenations of change type and node/edge type. If a node appears in more than one edge, the second time it appears it can be replaced by "_" to avoid repetition. 

E.g.:
e 0 1 a b bar
e 1 2 bla _ foo

The second edge here would be equivalent to:
"e 1 2 bla bar foo"

There are some change graphs given as examples. Graphs are separated by "\n\n$$\n---\n".

The last graph in this list of graphs is not yet complete. Some edges are missing. 
Your task is it, to complete the last graph by guessing the missing edges. You can guess this typically by looking at the examples and trying to deduce the patterns in the examples. Give the missing edges in the format
"e src_id tgt_id edge_label src_label tgt_label". Note that the beginning "e" is already part of the prompt. After the last edge of the change graph, add two new lines.
"""

load_openai_key()

def token_counter_gpt(result: Completion):
  # get token from result
  total_tokens = result['usage']['total_tokens'] # completion_tokens
  completion_tokens = result['usage']['completion_tokens']
  return total_tokens, completion_tokens

def completion_gpt(result: Completion):
    return "".join(result['choices'][0]['logprobs']['tokens'])

def completion_gpt_chat(result: Completion):
    return result['choices'][0]['message']['content']

def token_logprobs_gpt(result: Completion):
    return result['choices'][0]['logprobs']['token_logprobs']

def get_model_id_gpt(result: Completion):
    return result['model']

def compute_suprisal(token_logprops):
    return np.product(token_logprops)

# TODO for sampling edges we could use the algorithm for samplign provided in the lm miner
def sample_edge_gpt(model_id: str, prompt, multi_edge: bool):       
    if multi_edge:
        stop_token = GRAPH_STOP_TOKEN         
    else:
        stop_token = EDGE_STOP_TOKEN
    result = openai.Completion.create(model=model_id, prompt=prompt, max_tokens=MAX_EDGE_TOKENS, n=1, top_p=TOP_P, logprobs=LOGPROPS, stop=stop_token)
    total_tokens, completion_tokens = token_counter_gpt(result)
    completion_string = completion_gpt(result)
    token_logprobs = token_logprobs_gpt(result)
    return total_tokens, completion_tokens, completion_string

# TODO for sampling edges we could use the algorithm for samplign provided in the lm miner
def sample_edge_gpt_chat(model_id: str, prompt: str, system_instruction: str, multi_edge: bool):
    if multi_edge:
        stop_token = GRAPH_STOP_TOKEN         
    else:
        stop_token = EDGE_STOP_TOKEN
    result = openai.ChatCompletion.create(
        engine=model_id, # Actually this looks like a bug with the library and we should not need to give this parameter.
        model=model_id,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt}
        ], 
        max_tokens=MAX_EDGE_TOKENS, 
        n=1, 
        top_p=TOP_P, 
        stop=stop_token
    )
    total_tokens, completion_tokens = token_counter_gpt(result)
    specific_model_used = get_model_id_gpt(result)
    completion_string = completion_gpt_chat(result)
    LOGGER.info(f"Using model {model_id}")
    #token_logprobs = token_logprobs_gpt(result) token logprobs not available for chat models
    return total_tokens, completion_tokens, completion_string
    
def main(path_input_file: str, path_output_file: str, model_id: str, multi_edge: bool):
    input_df = pd.read_json(path_input_file, lines=True)
    
     # calculate pricing
    total_tokens = sum(input_df['total_token_count'])
    expected_price = total_tokens * model_prices[model_id]
    LOGGER.info(f"The total cost for this experiment are expected to be: {expected_price}USD")
    
    ask_for_proceed()
    
    #Create output directory
    os.makedirs(path_output_file, exist_ok=True)
    
    output_df = input_df.copy()
    counter = 0
    for idx, row in input_df.iterrows():
        counter += 1
        if counter % SLEEP_EVERY == 0:
            # artificial sleep to handle rate limitter
            time.sleep(SLEEP_PERIOD)
        
        prompt = row['prompt']
        if model_id in COMPLETION_MODELS:
            total_tokens, completion_tokens, completion_string = sample_edge_gpt(model_id , prompt, multi_edge)
        elif model_id in CHAT_MODELS:
            
            if multi_edge:
                instruction = CHAT_MODEL_INSTRUCTION_MULTI_EDGE
            else:
                instruction = CHAT_MODEL_INSTRUCTION
            
            total_tokens, completion_tokens, completion_string = sample_edge_gpt_chat(model_id , prompt, instruction, multi_edge)
        else:
            LOGGER.error(f"Model {model_id} currently not supported.")
            raise Exception(f"Model {model_id} currently not supported.")
        output_df.at[idx,'total_tokens'] = total_tokens
        output_df.at[idx,'completion_tokens'] = completion_tokens
        output_df.at[idx,'completion_string'] = completion_string
        output_df.to_json(path_output_file + '_snapshot_' + str(counter), orient='records', lines=True)


    # Store output
    output_df.to_json(path_output_file, orient='records', lines=True)


if __name__ == "__main__":
    """
    Executes when called as python module.
    """
    logging.basicConfig(level=logging.INFO)
    model_id = "gpt-4" #"text-davinci-003" #"code-davinci-002" #"text-ada-001"
    if len(sys.argv) == 4:
        multi_edge = sys.argv[3] == "MULTI_EDGE"
        main(sys.argv[1], sys.argv[2], model_id, multi_edge)
    else:
        LOGGER.info("Unexpected number of arguments. Call like python GPT_runner.py [input_path] [output_path] [SINGLE_EDGE/MULTI_EDGE].")
