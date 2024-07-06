

#!/usr/bin/python  

"""
    From given training and test set of single graph prompt completion pairs, this script filters them and creates few-shot samples, i.e., several training graphs and a test sample.
    
    There are two modes: 
        presample -> This filters the initial samples (e.g., based on token count of one sample) and then selects a certain fraction of the samples.
        finalsample -> This consumes the data from the presampling (probably a manual curation takes place in-between), and then create few-shot and test prompts and some additional meta-info.
    
    The presampling is thought to be used too downsample the data for a manual curation.
    After manual curation of the dataset, finalsampling can be used
    
    Call as python ./sample.py [presample|finalsample] ./../model_completion_dataset/Siemens_Mobility/results/experiment_samples/ ./../model_completion_dataset/Siemens_Mobility/results/experiment_selected_prompts/
    
"""

import math
import re
import sys, os
from typing import List
import pandas as pd
import random
import logging

from vector_database import ChangeGraphVectorDB


# TODO write sampling results (e.g., number of removed samples etc) to a csv-file (extend the one from the dataset generation step)

# Load the logger
LOGGER = logging.getLogger()

# For reproducability, we select a fixed seed for the random number generator
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Some further default configs TODO use some configuration mechanism to clean this script up
DEFAULT_ATTRIBUTE_RATIO = 0.2
DEFAULT_ALL_CHANGES_TRAIN_RATIO = 0.5
DEFAULT_NO_ATTRIBUTES = False
DEFAULT_MAX_FEW_SHOT = 20



class Document():
    """Class for storing a piece of text and associated metadata."""

    page_content: str
    """String text."""
    metadata: dict
    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """

    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata =metadata


def pre_sample(train_dataset_path: str, test_dataset_path: str, output_path: str, ratio: float = 1, token_max: int = 1000):
    
    train_df: pd.DataFrame = pd.read_json(train_dataset_path, lines=True)
    test_df: pd.DataFrame = pd.read_json(test_dataset_path, lines=True)
    
    # Calculate initial size
    size_train_beginning = train_df.size
    size_test_beginning = test_df.size
    
    # Filter dataset (too large and no completion or empty context we do not want)
    LOGGER.info(f"Filtering dataset. Only samples < {token_max} tokens and with non-empty prompt or completion are considered.")
    train_df = train_df.query('token_count < ' + str(token_max) + ' and number_of_removed_items > 0 and number_of_edges_graph > 0')#and completion_count > 0')
    test_df = test_df.query('token_count < ' + str(token_max) + ' and number_of_removed_items > 0 and number_of_edges_graph > 0')#and completion_count > 0')

    # Calculate size after filtering
    size_train_after = train_df.size
    size_test_after = test_df.size
    
    LOGGER.info(f"Removed {size_train_beginning - size_train_after} training samples for {train_dataset_path}.")
    LOGGER.info(f"Removed {size_test_beginning - size_test_after} training samples for {test_dataset_path}.")
    
    # instead of fix number, take a ratop x % of the dataset
    LOGGER.info(f"Dowsampling of dataset with a ratio of: {ratio}. (Ratio of 1 means no downsampling).")
    nb_samples_train = int(len(train_df) * ratio)
    nb_samples_test = int(len(test_df) * ratio)

    LOGGER.info(f"Training samples for {train_dataset_path} ({len(train_df)} samples in total): {nb_samples_train}")

    # Sample train and test samples
    train_samples = train_df.sample(n = nb_samples_train, replace=False, random_state=RANDOM_SEED)
    test_samples = test_df.sample(n = nb_samples_test, replace=False, random_state=RANDOM_SEED)
    
    # Concat prompt + completion
    train_samples['full_graph']  = train_samples.apply(lambda row: row['prompt'] + row['completion'], axis=1) # along axis 1 means row-wise
    test_samples['full_graph']  = test_samples.apply(lambda row: row['prompt'] + row['completion'], axis=1) # along axis 1 means row-wise

    # store dataset.
    train_samples.to_json(output_path + '/train_samples.jsonl', orient='records', lines=True)
    test_samples.to_json(output_path + '/test_samples.jsonl', orient='records', lines=True)
    
def select_few_shot_samples(vector_db: ChangeGraphVectorDB, nb_few_shot_samples: int, ratio_arbitrary_change_train: float, test_sample, 
                            similarity_based_sampling: bool, no_attributes: bool = DEFAULT_NO_ATTRIBUTES):
    # We sample 50% attribute value changes and 50% other changes (or other ratio given as ratio parameters)
    #include_filter={"last_edge_change_type": "Change_attribute"}
    exclude_filter={"last_edge_change_type": {"$ne": "Change_attribute"}}
    scope_filter={"scope": "train"}
    nb_arbitrary_samples = max(int(ratio_arbitrary_change_train*nb_few_shot_samples), 1) # at least 1 arbitrary sample (in case nb_few_shot_samples = 1)
    nb_no_change_attribute = nb_few_shot_samples - nb_arbitrary_samples
        
    # There are two modi: Random sampling of few shot samples or similarity based
    if similarity_based_sampling:
        if no_attributes:
            selected_few_shot_samples = vector_db.query_k_most_diverse_strong(test_sample.metadata['prompt'], scope="train",  k=nb_few_shot_samples, k_retrieve=nb_few_shot_samples*2, num_of_iterations=100)
        else:
            arbitrary_few_shot_samples = vector_db.query_k_most_diverse_strong(test_sample.metadata['prompt'], scope="train", k=nb_arbitrary_samples, k_retrieve=nb_arbitrary_samples*2)
            no_change_attribute_few_shot_samples = vector_db.query_k_most_diverse_strong(test_sample.metadata['prompt'], scope="train", additional_filter=exclude_filter, k=nb_no_change_attribute, k_retrieve=nb_no_change_attribute*2)
            selected_few_shot_samples = arbitrary_few_shot_samples + no_change_attribute_few_shot_samples
    else:
        # We sample nb_few_shot_samples samples randomly from the test dataset
        if no_attributes:
            selected_few_shot_samples = random.sample(vector_db.get_filtered(where=scope_filter), nb_few_shot_samples)
        else:
            arbitrary_few_shot_samples = random.sample(vector_db.get_filtered(where=scope_filter), nb_arbitrary_samples) #.sample(nb_few_shot_samples, replace=False, random_state=RANDOM_SEED+id_pair)
            no_change_attribute_few_shot_samples = random.sample(vector_db.get_filtered(where={"$and": [scope_filter, exclude_filter]}), nb_no_change_attribute) #.sample(nb_few_shot_samples, replace=False, random_state=RANDOM_SEED+id_pair)
            selected_few_shot_samples = arbitrary_few_shot_samples + no_change_attribute_few_shot_samples
            
    return selected_few_shot_samples

def select_test_samples(vector_db: ChangeGraphVectorDB, nb_pairs: int, ratio_attribute_changes: float, no_attributes: bool = DEFAULT_NO_ATTRIBUTES):
    # We want to ensure, that the distribution of test samples across the datasets is similar to the original distribution
    # Get the distribution of test datasets first
    ds_distribution = vector_db.get_distribution("source", where={"scope": "test"})
    test_samples = []
    for source, frequency in ds_distribution.items():
        nb_samples = math.ceil(nb_pairs*frequency)
    
        # select nb_pairs test samples with diversity sampling
        # additionally, we often have a biased dataset in the direction of "Change_attribute" changes. We try to debias here, by only selecting xx% (default 25%) of changes to be attribute value changes and the remaining samples we exclude attribute value changes
        if no_attributes:
            source_samples = vector_db.k_most_diverse_elements_strong(nb_samples, 'test', additional_filter={"source": source}, num_of_iterations=100)  # .sample(nb_pairs, replace=False, random_state=RANDOM_SEED).iloc[0]
        else:
            if nb_samples == 1:
                    source_samples = vector_db.k_most_diverse_elements_strong(1, 'test', additional_filter={"$and": [{"last_edge_change_type": "Change_attribute"}, {"source": source}]}, num_of_iterations=100)
                    source_samples += vector_db.k_most_diverse_elements_strong(math.ceil(1/ratio_attribute_changes), 'test', additional_filter={"$and":[{"last_edge_change_type": {"$ne": "Change_attribute"}}, {"source": source}]}, num_of_iterations=100)  # .sample(nb_pairs, replace=False, random_state=RANDOM_SEED).iloc[0]
                    source_samples = random.sample(source_samples, 1)
            else:       
                nb_attribute_changes = math.ceil(nb_samples*ratio_attribute_changes)
                source_samples_attribute_changes = vector_db.k_most_diverse_elements_strong(nb_attribute_changes, 'test', additional_filter={"$and": [{"last_edge_change_type": "Change_attribute"}, {"source": source}]}, num_of_iterations=100)  # .sample(nb_pairs, replace=False, random_state=RANDOM_SEED).iloc[0]
                nb_others = nb_samples - len(source_samples_attribute_changes)
                source_samples_others = []
                if nb_others > 0:
                    source_samples_others = vector_db.k_most_diverse_elements_strong(nb_others, 'test', additional_filter={"$and":[{"last_edge_change_type": {"$ne": "Change_attribute"}}, {"source": source}]}, num_of_iterations=100)  # .sample(nb_pairs, replace=False, random_state=RANDOM_SEED).iloc[0]
                source_samples = source_samples_attribute_changes + source_samples_others
        LOGGER.info(f"Sampled {len(source_samples)} for datasset {source}.")
        test_samples += source_samples

    return test_samples

def select_few_shot_samples_for_test_samples(test_samples: List[Document], vector_db: ChangeGraphVectorDB, max_few_shot_samples: int = DEFAULT_MAX_FEW_SHOT, 
                 max_tokens: int = 4096, similarity_based_sampling: bool=True, ratio_arbitrary_change_train=DEFAULT_ALL_CHANGES_TRAIN_RATIO, no_attributes: bool = DEFAULT_NO_ATTRIBUTES):
    nb_pairs = len(test_samples)
    
    # Create dataframes for results
    df_few_shot_samples = pd.DataFrame(index=range(nb_pairs), columns=["id", "source", "prompt", "completion", "few_shot_token_count", "test_token_count", "total_token_count", "few_shot_count", "test_edge_count", "test_completion_edge_count", "completion_type"])
    # to keep track of too long prompts (because of limited context). We count how many of the few-shots we have to remove again.
    excess_samples = pd.DataFrame(index=range(nb_pairs), columns=["excess_count"])

    # randomly sample nb_pairs "pairs" of few-show samples and test sample
    for id_pair, test_sample in enumerate(test_samples):
        # randomly select a number between 1 and max_few_shot_samples =: nb_few_shot_samples
        nb_few_shot_samples = random.randint(1, max_few_shot_samples)
        
        # Select few-shot samples
        selected_few_shot_samples = select_few_shot_samples(vector_db, nb_few_shot_samples, ratio_arbitrary_change_train, test_sample, similarity_based_sampling)

        # shuffle samples
        random.shuffle(selected_few_shot_samples)
        
        # ensure context size sufficient
        selected_few_shot_samples_token_counts = [doc.metadata['token_count'] for doc in selected_few_shot_samples]
        total_few_shot_tokens = sum(selected_few_shot_samples_token_counts)
        total_test_tokens = test_sample.metadata['token_count']
        
        # randomly remove sample until token count < max_tokens
        total_tokens = total_test_tokens + total_few_shot_tokens
        
        # We store the number of few shot samples that are too long and have to be adjusted (for reporting purposes)
        excess_samples.loc[id_pair] = 0
        
        while total_tokens > max_tokens:
            # keep track of the number of removed samples
            excess_samples.loc[id_pair] = excess_samples.loc[id_pair] + 1
            nb_few_shot_samples = nb_few_shot_samples - 1
            selected_few_shot_samples = random.sample(selected_few_shot_samples, nb_few_shot_samples)
            total_few_shot_tokens = sum([doc.metadata['token_count'] for doc in selected_few_shot_samples])
            total_tokens = total_test_tokens + total_few_shot_tokens
    
        # generate the final prompt and expected result
        final_prompt = ""
        few_shot_samples = "\n---\n".join([sample.page_content for sample in selected_few_shot_samples])
        final_prompt += few_shot_samples
        final_prompt += "\n---\n"
        final_prompt += test_sample.metadata["prompt"]
        
        df_few_shot_samples.loc[id_pair]["id"] = id_pair
        df_few_shot_samples.loc[id_pair]["prompt"] = final_prompt
        df_few_shot_samples.loc[id_pair]["completion"] = test_sample.metadata["completion"]
        df_few_shot_samples.loc[id_pair]["few_shot_token_count"] = total_few_shot_tokens
        df_few_shot_samples.loc[id_pair]["test_token_count"] = test_sample.metadata["token_count"]
        df_few_shot_samples.loc[id_pair]["total_token_count"] = total_tokens
        df_few_shot_samples.loc[id_pair]["few_shot_count"] = nb_few_shot_samples
        df_few_shot_samples.loc[id_pair]["source"] = test_sample.metadata["source"]

        df_few_shot_samples.loc[id_pair]["test_edge_count"] = len(re.findall(r'\ne \d+ \d+', test_sample.metadata["prompt"])) # actually the number of context edges (of the change graph)
        
        if "number_of_removed_items" in test_sample.metadata.keys():
            df_few_shot_samples.loc[id_pair]["test_completion_edge_count"]= test_sample.metadata["number_of_removed_items"] # the number of removed edges

        if not no_attributes:
            df_few_shot_samples.loc[id_pair]["completion_type"] = test_sample.metadata["change_type"]
        

    return excess_samples, df_few_shot_samples

def load_test_samples_from_data_frame(input_path: str) -> List[Document]:
    samples_dataframe = pd.read_json(input_path, lines=True)
    
    test_samples: List[Document] = []
    for idx, row in samples_dataframe.iterrows():
        # we need the following information:
        # prompt, completion, token_count, test_completion_edge_count (alias "number_of_removed_items"), completion_type (alias "change_type")
        
        # prompt, completion we can get from the completion and the "last graph" 
        prompt = row['prompt'].split("$$\n---\n")[-1] # we have a set of graphs in the dataframe seperated by $$\n---\n and only the last one is the test sample we are interested in
        completion = row['completion']
        
        token_count = row['test_token_count']
        test_completion_edge_count = row['test_completion_edge_count']
        completion_type = row['completion_type']
        metadata = {'prompt': prompt, 'completion': completion, 'token_count': token_count, 'number_of_removed_items': test_completion_edge_count, 'change_type': completion_type}
        # These will then be part of a dictionary 'metadata' because this is how the interface is designed here.
        doc = Document("", metadata)
        test_samples.append(doc)
    
    return samples_dataframe['sample_id'].tolist(), test_samples
    

def few_shots_for_existing_test_samples(input_samples_path: str, path_db: str, output_path: str, max_few_shot_samples: int = DEFAULT_MAX_FEW_SHOT, max_tokens: int = 4096, 
                                        use_openai_embeddings: bool=False, similarity_based_sampling: bool=True, ratio_arbitrary_change_train=DEFAULT_ALL_CHANGES_TRAIN_RATIO, no_attributes: bool = DEFAULT_NO_ATTRIBUTES):
    """Same as #final_sample() but for an already given set of test samples. It only generates a set of few-shots for given test samples.

    Args:
        input_samples_path (str): Path to a dataframe JSON of test samples.
        path_db (str): Path to the sample database.
        output_path (str): The outputpath for the results.
        max_few_shot_samples (int, optional): Maximum number of few-shot samples. Defaults to 20.
        max_tokens (int, optional): Token limit. Defaults to 4096.
        use_openai_embeddings (bool, optional): If openai embeddings are used for the database. Defaults to False.
        similarity_based_sampling (bool, optional): If similarity based sampling should be used. If not, random sampling is used. Defaults to True.
        ratio_arbitrary_change_train (float, optional): The ratio of arbitrary changes in the few-shot set. The rest will be non-attribute changes (i.e., adds and removes). Defaults to 0.5.
        no_attributes (bool, Optional): Indicates whether there are attributes in the dataset (or types only), defaults to False
    """
    
    LOGGER.info(f"Loading samples from {input_samples_path}.")
    
    index, test_samples = load_test_samples_from_data_frame(input_samples_path)
    
    LOGGER.info(f"Starting final sampling with {len(test_samples)} samples, " +
                f" and up to {max_few_shot_samples} few-shot examples.")
    # First, we load the (vector() database with the documents.
    vector_db = ChangeGraphVectorDB(path_db, use_openai_embeddings)
    vector_db.load_existing()
    
    count_train = len(vector_db.get_filtered(where={"scope": "train"}))
       
    # Get max of train samples available and compute max of available samples and desired max prompt samples
    max_few_shot_samples = min(count_train, max_few_shot_samples)
         
    # For the test samples, retrieve some few shot samples and return the dataset as a dataframe. 
    # During sampling to large context can happen and is tracked in the excess_samples dataframe
    excess_samples, df_few_shot_samples = select_few_shot_samples_for_test_samples(test_samples, vector_db, max_few_shot_samples, max_tokens, 
                                                                                   similarity_based_sampling, ratio_arbitrary_change_train, no_attributes=no_attributes)
       
    # store samples and other meta-data
    os.makedirs(output_path, exist_ok=True)
    df_few_shot_samples.to_json(output_path + '/few_shot_dataset.jsonl', orient='records', lines=True)
    df_few_shot_samples.loc[:, 'sample_id'] = index
    excess_samples.to_csv(output_path + '/excess_samples.csv')

    
def final_sample(path_db: str, output_path: str, max_few_shot_samples: int = DEFAULT_MAX_FEW_SHOT, nb_pairs: int = 200, 
                 max_tokens: int = 4096, use_openai_embeddings: bool=False, similarity_based_sampling: bool=True, 
                 ratio_attribute_changes=DEFAULT_ATTRIBUTE_RATIO, ratio_arbitrary_change_train=DEFAULT_ALL_CHANGES_TRAIN_RATIO, no_attributes: bool = DEFAULT_NO_ATTRIBUTES):
    """
    This method generates the final few-shot learning prompts, consisting of some 
    complete (change) graph serializations (few-shot training examples)
    and one incomplete sample. Furthermore, the correct completion is stored in the dataset.
    The dataset is stored to disk.
    
    There are two modes for sampling. 
        - random: Randomly drawing few-shot samples from the entire train dataset
        - similarity: Draw the most similar (e.g., according to embedding similarity) ones.
    
    In the final dataset, we have 
        - the number of tokens
        - the number of prompt edges of the candidate
        - the number of completion edges of the candidate (should be one always)
        #- origin dataset -> Not used when datasets are merged
        - number of few show examples
        - type of completion (add/remove node, add/remove edge, change attribute)
        - the prompt (few-show and incomplete samples)
        - the correct completion

    Args:
        input_path (str): Path to the dataset folders
        output_path (str): Path to store the final sampled data to
        max_few_shot_samples (int, optional): The maximum number of few shot samples. Defaults to 20.
        nb_pairs (int, optional): How many pairs the dataset should contain. Defaults to 200.
        max_tokens (int, optional): The maximum number of tokens for the prompt (few-shot + incomplete sample). Defaults to 4096.
        use_openai_embeddings (bool, optional): If True, OpenAI API is used to compute embeddings. This can be expensive for large databases. 
                                                Note that the same setting should be used as for the creation of the database. Defaults to False.
        similarity_based_sampling (bool, optional): If True, few-shot samples are selected by similarit instead of random. Defaults to True.
        ratio_attribute_changes (float, optional): The ratio of attribute changes (of the last edge) in the test samples. Defaults to 0.25.
        ratio_arbitrary_change_train (float, optional): The ratio of attribute changes (of the last edge) in the train samples. Defaults to 0.5.
        no_attributes (bool, Optional): Indicates whether there are attributes in the dataset (or types only), defaults to False
    """
    LOGGER.info(f"Starting final sampling with {nb_pairs} samples, " +
                f"a ratio of {ratio_attribute_changes} attribute value changes" +
                f" and up to {max_few_shot_samples} few-shot examples.")
    # First, we load the (vector() database with the documents.
    vector_db = ChangeGraphVectorDB(path_db, use_openai_embeddings)
    vector_db.load_existing()
    
    count_train = len(vector_db.get_filtered(where={"scope": "train"}))
    count_test = len(vector_db.get_filtered(where={"scope": "test"}))
        
    # If there are less test samples than nb_pairs, only sample all test samples
    if count_test < nb_pairs:
        LOGGER.warning("There are less test samples available in the dataset than required samples. Only sampling from all test samples.")
        nb_pairs = count_test
    
    # Get max of train samples available and compute max of available samples and desired max prompt samples
    max_few_shot_samples = min(count_train, max_few_shot_samples)
    
   
    # Select test samples
    test_samples = select_test_samples(vector_db, nb_pairs, ratio_attribute_changes, no_attributes=no_attributes)
    LOGGER.info(f"Selected {len(test_samples)} in total.")
    
    # For the test samples, retrieve some few shot samples and return the dataset as a dataframe. 
    # During sampling to large context can happen and is tracked in the excess_samples dataframe
    excess_samples, df_few_shot_samples = select_few_shot_samples_for_test_samples(test_samples, vector_db, max_few_shot_samples,
                                                                                   max_tokens, similarity_based_sampling, ratio_arbitrary_change_train,
                                                                                   no_attributes=no_attributes)
       
    # store samples and other meta-data
    os.makedirs(output_path, exist_ok=True)
    df_few_shot_samples.to_json(output_path + '/few_shot_dataset.jsonl', orient='records', lines=True)
    excess_samples.to_csv(output_path + '/excess_samples.csv')

def pre_sample_folder(input_path: str, output_path: str):
    # Loop over all datasets
    for folder_name in os.listdir(input_path):
        input_ds_path = input_path + '/' + folder_name
        # Skip files in the input_path
        if not os.path.isdir(input_ds_path):
            continue
        
        # Create output folder
        output_ds_path = output_path + '/' + folder_name
        os.makedirs(output_ds_path, exist_ok=True)

        # TODO check if files exist    
        train_path = input_ds_path + '/finetune_training/dataset_dfs_edges.jsonl'
        test_path = input_ds_path  + '/finetune_training/dataset_dfs_edges_test.jsonl'
        LOGGER.info(f"Processing dataset: {folder_name}")
        pre_sample(train_path, test_path, output_ds_path)

def merge_datasets(input_path: str):
    train_samples = dict()
    test_samples = dict()
    
    # Loop over all datasets
    for folder_name in os.listdir(input_path):
        input_ds_path = input_path + '/' + folder_name
        # Skip files in the input_path
        if not os.path.isdir(input_ds_path):
            continue

        # TODO check if files exist    
        train_path = input_ds_path + '/train_samples.jsonl'
        test_path = input_ds_path  + '/test_samples.jsonl'
        
        # Put all dataframes in a dictionary
        train_samples[folder_name] = pd.read_json(train_path, lines=True)
        test_samples[folder_name] = pd.read_json(test_path, lines=True)
        

    # finally, concat all dataframes
    df_train_all = pd.concat(train_samples.values(), keys=train_samples.keys(),
          names=['dataset', 'id'])
    df_test_all = pd.concat(test_samples.values(), keys=test_samples.keys(),
          names=['dataset', 'id'])
    
    df_train_all.reset_index(level=['dataset','id'], inplace=True)
    df_test_all.reset_index(level=['dataset','id'], inplace=True)
    df_train_all.drop(columns=['id'], axis=1, inplace=True)
    df_test_all.drop(columns=['id'], axis=1, inplace=True)

    
    return df_train_all, df_test_all


def downsample_samples(input_folder, output_folder):

    # Merge the two datasets into a single dataframe and save the combined result.
    # Establish clusters based on unique combinations of the attributes: change_type, detail_type,
    # context_type, similar_few_shot, and few_shot_count.
    # From each cluster, select one sample.
    # Save this sampled dataset,  it's the finalized input for the upcoming experiment.
    classification_file = input_folder + '/' +'stats.csv'
    completion_data_set_file = input_folder + '/' +'few_shot_dataset.jsonl'
    output_file = output_folder + '/' + 'downsampled_few_shot_dataset.jsonl'
    
    df_csv = pd.read_csv(classification_file)
    df_jsonl = pd.read_json(completion_data_set_file, lines=True)
    #"inner" keep only when there's a match in both dataframes
    merged_df = df_csv.merge(df_jsonl, left_on='sample_id', right_on='id', how='inner')


    #Group (because from every group we want to sample afterwards)
    grouped = merged_df.groupby(['change_type', 'detail_type', 'context_type', 'similar_few_shot', 'few_shot_count'], group_keys=False)

    # Sample from each group
    select_one_per_group = grouped.apply(lambda x: x.sample(1))

    select_one_per_group.to_json(output_file, orient='records', lines=True)


def main(sampling: str, input_path: str, output_path: str, path_existing_samples: str=None, 
         no_attributes: bool = False):
    """
    
    The main method of this module.

    Args:
        sampling (str): Sampling mode [presample|finalsample]
        input_path (str): Path to the training and test sets.
        output_path (str): To store the sampled datasets.
        path_existing_samples (str, Optional): Path to existing samples, in the case only few-shot training samples should be sampled.
        no_attributes (bool, Optional): Indicates whether there are attributes in the dataset (or types only), defaults to False
    """
    
    # Logger settings
    logging.basicConfig(level=logging.INFO)
        
    if sampling == 'presample':
        LOGGER.info("Starting presampling and filtering.")
        pre_sample_folder(input_path, output_path)
    elif sampling == 'finalsample':
        LOGGER.info("Creating samples with few-shot samples and completion task.")
        final_sample(input_path, output_path, no_attributes=no_attributes)
    elif sampling == 'downsample':
        LOGGER.info("Selecting representetives from the completion dataset for the completion task.")
        downsample_samples(input_path, output_path)
    elif sampling == 'add_few_shot':
        similarity_based_sampling = False
        LOGGER.info(f"Selecting few-shot examples for given samples. Similarity based sampling: {similarity_based_sampling}")
        few_shots_for_existing_test_samples(path_existing_samples, input_path, output_path, similarity_based_sampling=similarity_based_sampling)    
    else:
        LOGGER.error(f"Illegal parameter '{sampling}' for sampling. Only 'presample' and 'finalsample' supported.")

if __name__ == "__main__":
    """
    Executes when called as python module.
    """
    # For some datasets we have no attributes and can therefore not sample attributes changes here
    # Set this switch to True (e.g., for the synthetic dataset)
    # TODO it's kind of ridicoulus how this simple hack to select the number of attributes (for balancing purposes) messes up the entire sampling process. this has to be cleaned up.
    no_attributes = DEFAULT_NO_ATTRIBUTES
    
    if len(sys.argv) == 4:
        main(sys.argv[1], sys.argv[2],sys.argv[3], no_attributes=no_attributes)
    elif len(sys.argv) == 5:
        assert sys.argv[1] == 'add_few_shot'
        main(sys.argv[1], sys.argv[2],sys.argv[3], path_existing_samples=sys.argv[4], no_attributes=no_attributes)
    else:
        LOGGER.error("Unexpected number of arguments. Call like python sample.py [presample|finalsample|add_few_shot] [input_path] [output_path] [optional: path_to_existing_samples].")
