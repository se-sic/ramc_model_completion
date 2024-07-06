import re
import subprocess
from subprocess import TimeoutExpired
# For UTC timestring parsing
import dateutil.parser
import openai
import os, sys
import pandas as pd
import numpy as np

# StringIO for directly feeding a csv string into pandasimport sys
if sys.version_info[0] < 3: 
    from StringIO import StringIO
else:
    from io import StringIO

regex_finetune_str = r"^.*\sCreated fine-tune:\s(ft-.*)\n.*Fine-tune costs\s\$(.*)\n(?:.*\n)*\[(.*)\]\sCompleted epoch 1\/(\d)\n(?:.*\n)*\[(.*)\]\sCompleted epoch \d\/\d\n(?:.*\n)*.* Uploaded model:\s((.*):.*)\n(?:.*\n)*$"

# Load API key and setup OPEN-AI Lib
if not os.path.exists('./secrets/openai.key'):
    print("WARN: You need to provide your OpenAI API-Key in a file /secrets/openai.key")

with open('./secrets/openai.key', 'r') as f:
    api_key = f.read().strip()
    os.environ['OPENAI_API_KEY']=api_key
    openai.api_key=api_key

def parse_dataset_params(folder_name):
    regex = r"^\D*(\d+)_eo(\d+)_p(\d\.\d)$"
    match = re.match(regex, folder_name)
    if match is None:
      return "", "", ""
    nb_diffs = int(match.group(1))
    nb_eos = int(match.group(2))
    pertubation = float(match.group(3).replace(",", "."))
    return nb_diffs, nb_eos, pertubation
    
def call_subprocess_timeout(args, timeout_s=5):
  try:
    result = subprocess.run(args, timeout=timeout_s, capture_output=True)
    output = result.stdout
    if output is None:
      print("WARN: No output from subprocess.")
      output = b""
    if len(result.stderr) > 0:
      print(f"WARN: Error from subprocess: {result.stderr}")
  except TimeoutExpired as e:
    print("INFO: Timout during subprocess call. This can still be okay, if the process is running asynchronously (e.g., for OpenAI fine-tuning).")
    if e.stdout is not None:
      output = e.stdout
    else:
      raise e
  except Exception as e:
    print(f"ERROR: Failed to run subprocess: {str(e)}")
    raise e
  
  return output.decode("UTF-8")
  
def load_results(finetunes_id):
    # Load fine-tuning results
    cli_output =  call_subprocess_timeout(["openai", "api", "fine_tunes.follow", "-i", finetunes_id.strip()], timeout_s=20)
    # fix line endings
    cli_output = cli_output.replace('\r\n','\n')
    cli_output = cli_output.replace('\r','\n')
    # Parse information
    all_matches = list(re.finditer(regex_finetune_str, cli_output, re.MULTILINE))

    if len(all_matches) == 0:
      print("WARN: Output of finetuning could not be parsed correctly. Maybe job not finished?")
      model_id = finetunes_id
      cost = "NA"
      total_time = "NA"
      epochs_str = "NA"
      model_name = "NA"
      base_model = "NA"
    else:
      matches = all_matches[0]
      model_id = matches.group(1)
      cost = matches.group(2)
      start_time = dateutil.parser.parse(matches.group(3), dayfirst=True).timestamp()
      epochs_str = matches.group(4)
      end_time = dateutil.parser.parse(matches.group(5), dayfirst=True).timestamp()
      model_name = matches.group(6)
      base_model = matches.group(7)
      total_time = end_time - start_time
    
    # Now we collect the training_token_accuracs and average the last ten runs
    results_cli_output = call_subprocess_timeout(["openai", "api", "fine_tunes.results", "-i", finetunes_id.strip()], timeout_s=20)
    # fix line endings
    results_cli_output = results_cli_output.replace('\r\n','\n')
    results_cli_output = results_cli_output.replace('\r','\n')
    results_data_string = StringIO(results_cli_output)
    try:
        results_df = pd.read_csv(results_data_string, sep=",", header=0)
        acc = results_df['training_token_accuracy'].to_numpy()
        average_token_acc = np.average(acc[:-10])
    except Exception as e:
        print("WARN: Couldn't read results file.")
        print(str(e))
        average_token_acc = "NA"
    
    return {"finetune_id": model_id, "cost": cost, "finetune_time": total_time, "model_name": model_name, "nb_epochs": epochs_str, "base_model": base_model, "average_token_acc": average_token_acc}

def ask_for_proceed():
  # Only continue if this is okay for the user
  # Ask the user for agreement
  answer = input("Do you want to proceed? (yes/no) ")

  # Check the answer
  if answer.lower() == "yes" or answer.lower() == "y":
    # Continue the program
    print("You agreed. The program continues.")
    # Exit the program otherwise
  else:
    print("You disagreed. The program exits.")
    exit()
    
def choose_pytorch_device():
  import torch
  if torch.cuda.is_available():
    torch.set_default_device('cuda')
    device = 'cuda'
  elif torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    torch.set_default_device('mps')
    device = 'mps'
  else:
    torch.set_default_device('cpu')
    device = 'cpu'
  return device
