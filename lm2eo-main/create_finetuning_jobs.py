#!/usr/bin/python  
import sys, os
import re
import openai
from common import call_subprocess_timeout
import pandas as pd
import numpy as np
from common import load_results
import logging

LOGGER = logging.getLogger()


# Load API key and setup OPEN-AI Lib
if not os.path.exists('./secrets/openai.key'):
    LOGGER.info("WARN: You need to provide your OpenAI API-Key in a file /secrets/openai.key")

with open('./secrets/openai.key', 'r') as f:
    api_key = f.read().strip()
    os.environ['OPENAI_API_KEY']=api_key
    openai.api_key=api_key


regex_finetune_start = r"^.*Created fine-tune:\s(.*).*$"

serialization_strategies = ["dfs_edges"] # ["random_edges", "as_is", "dfs_edges"] #
base_models = ["davinci"] #["ada", "curie", "davinci"]

def get_existing_model(jobs_list, dataset, strategy, base_model):
    lookup_rows = jobs_list.loc[(jobs_list['Id'] == dataset) & (jobs_list['Strategy'] == strategy) & (jobs_list['Base_Model'] == base_model)]
    epochs_max = 0
    max_id = None       
    for idx, row in lookup_rows.iterrows():
        epochs = int(row['Epochs'])
        if epochs > epochs_max:
            epochs_max = epochs
            max_id = finetune_id = row['Finetune_Id']
        
    return load_results(max_id)


def fine_tune(input_path, results_path, dataset, strategy, base_model, nb_epochs, fine_tuned=None, ft_epochs=None):                
    training_data_file = input_path + "/" + dataset + "/finetune_training/dataset_" + strategy + ".jsonl"
    test_data_file = input_path + "/" + dataset + "/finetune_training/dataset_" + strategy + "_test.jsonl"
    
    if fine_tuned is not None:
        model = fine_tuned
    else:
        model = base_model
        
    if ft_epochs is not None:
        epochs = nb_epochs - ft_epochs
    else:
        epochs = nb_epochs
                   
    # Trigger fine-tuning
    openai_cli_output = call_subprocess_timeout(["openai", "api", "fine_tunes.create", "-t", training_data_file, "-v", test_data_file, "-m", model, "--n_epochs", str(epochs), "--no_check_if_files_exist"], timeout_s=20)
    # Extract Id
    all_matches = list(re.finditer(regex_finetune_start, openai_cli_output, re.MULTILINE))
    if len(all_matches) == 0:
        LOGGER.info("WARN: Output of finetuning could not be parsed correctly")
        finetune_id = "NA"
    else:
        matches = all_matches[0]
        finetune_id = matches.group(1).strip()
                
    with open(results_path, 'a') as f:
        f.write(f"{dataset};{strategy};{nb_epochs};{base_model};{finetune_id}\n")


def train_fintuned(input_path, results_path, epochs):
    jobs_list = pd.read_csv(results_path, sep=";", header=0, keep_default_na=False)    
   
    for strategy in serialization_strategies:
        for base_model in base_models:
            for dataset in sorted(os.listdir(input_path)):
                # Skip files in the input_path
                if not os.path.isdir(input_path + '/' + dataset):
                    continue
                # lookup the right model, then load the mode name
                model_data = get_existing_model(jobs_list, dataset, strategy, base_model)

                # Check if finetune results are ready yet.
                if model_data['model_name'] is None or model_data['model_name'] == 'NA':
                    LOGGER.info("WARN: Finetuned model not available yet. Maybe try later.")
                    raise Exception(f"Finetune results for {dataset} and strategy {strategy} and base model {base_model} not available yet.")
                else:
                    LOGGER.info(f"For {dataset} and strategy {strategy} and base model {base_model}, found {model_data}")
                    LOGGER.info(f"Still have to train {int(epochs) - int(model_data['nb_epochs'])} epochs!")
                    fine_tune(input_path, results_path, dataset, strategy, base_model, epochs, fine_tuned=model_data['model_name'], ft_epochs=int(model_data['nb_epochs']))

def ft_base_models(input_path, results_path):
    if os.path.exists(results_path):
        LOGGER.info(f"WARN: There was already a results file in {results_path}.")
        os.remove(results_path)
        
        
    # Write the header of the results file
    
    with open(results_path, 'w') as f:
        f.write("Id;Strategy;Epochs;Base_Model;Finetune_Id\n")

    for strategy in serialization_strategies:
        for base_model in base_models:
            nb_epochs = 4         
            for dataset in sorted(os.listdir(input_path)):
                # Skip files in the input_path
                if not os.path.isdir(input_path + '/' + dataset):
                  continue
                fine_tune(input_path, results_path, dataset, strategy, base_model, nb_epochs)


def main(input_path, output_path, epochs=4):
    # Create output folder
    os.makedirs(output_path, exist_ok=True)
    # path for the output of the csv file
    results_path = output_path + '/results.csv'
    
    if int(epochs) == 4:
        ft_base_models(input_path, results_path)
    elif int(epochs) < 4:
        LOGGER.info("WARN: Only epochs >= 4 allowed.")
    else:
        if not os.path.exists(results_path):
            LOGGER.info("WARN: No fine-tuned models found. For epochs > 4 at least one model with epcohs >= 4 has to be available.")
        
        train_fintuned(input_path, results_path, int(epochs))
        #Fined the closest fine-tuned model
        # For n > 4 epochs:
        # Find out model nae for same parameters but 4 epochs
        # Train for n-4 epochs with base model equals the one


if __name__ == "__main__":
  if len(sys.argv) == 4:
    main(sys.argv[1], sys.argv[2], sys.argv[3])
  else:
    LOGGER.info("Unexpected number of arguments. At least input path, output path, and number of epochs are expected")
