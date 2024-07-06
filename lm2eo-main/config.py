# TODO move the configuration in a configuration class
# Min probabilities for pattern, edges, and tokens
min_token_prob = 0.15 # should be somehow computed from the size of the set of labels
min_edge_prob = 0.005 # 0,5% probability for edges
min_total_prob = 0.0001 # total prob of the entire continuation (product of all token probs)

# Decays
max_new_edge_decay = 0.995 # The relative decrease in probability (and thus confidence) from one generated edge to the next, measured as the beginning edge token "e" probability. If this drops to quickly, this might signal that the "pattern" is complete.
max_edge_decay = 1
max_edge_label_decay = 1

# Stop tokens
stop_tokens = ["\n\n", "\n$$", "\ne"]
pattern_stop_tokens = ["\n\n", "\n$$"]
edge_stop_tokens = ["\ne"]

# Some API parameters. TODO actually we should try to not rely too much on API specific things.
#num_edge_sampled = 5
max_edge_tokens = 25 # TODO compute from meta-model or input data
# max_pattern_tokens = 150 # Substituted by max edges
max_edges = 15
max_generated_tokens = 1 # how many tokens to retrieve... See the API description
temperature = 0 # The randomness of the answers
top_p = 0 # The top x(=0) percent of tokens are considered, 0 means no randomness.
logprobs = 5 # there are the 5 best tokens at most -- for more, sales can be contacted

# Some patterns, even if complete will additionally be extended, if the reason for their completion is one of the following
continue_extension_complete_reasons = ["NEW_EDGE_DECAY", "EDGE_DECAY", "EDGE_LABEL_DECAY"]
# Some patterns, if complete have to be "after-the-fact" corrected, i.e., the latest edge has to be cut off.
after_the_fact_correction_reasons = ["EDGE_DECAY", "EDGE_LABEL_DECAY", "UNLIKELY_EDGE"]

# Prices for the OpenAI models
model_prices = {
"text-ada-001": 0.0004 / 1000,
"text-babbage-001": 0.0005 / 1000,
"text-curie-001": 0.002 / 1000,
"text-davinci-001": 0.02 / 1000,
"text-davinci-002": 0.02 / 1000,
"text-davinci-003": 0.02 / 1000,
"ada": 0.0004 / 1000,
"babbage": 0.0005 / 1000,
"curie": 0.002 / 1000,
"davinci": 0.02 / 1000,
"code-davinci-002": 0,
"embedding": 0.0001 / 1000,
"gpt-4": 0.06 / 1000
}

def load_openai_key(azure=True):
    import openai
    import os
    
    # Load API key and setup OPEN-AI Lib
    if not os.path.exists('./secrets/openai.key'):
        print("WARN: You need to provide your OpenAI API-Key in a file /secrets/openai.key")
        raise Exception("Failed to load openai key.")
    with open('./secrets/openai.key', 'r') as f:
        api_key = f.read().strip()
        
    if azure and not os.path.exists('./secrets/azure.url'):
        print("WARN: You need to provide a url to your Azure deployment in a file /secrets/azure.url")
        raise Exception("Failed to load Azure API endpoint.")

    if azure:
        with open('./secrets/azure.url', 'r') as f:
            azure_url = f.read().strip()

    if not azure:
        os.environ['OPENAI_API_KEY']=api_key
        openai.api_key=api_key
    else:
        # Set OpenAI configuration settings
        print("Using azure.")
        openai.api_type = "azure"
        openai.api_base = azure_url
        openai.api_version = "2023-03-15-preview" # chat completions only available with preview
        openai.api_key = api_key