# Software Model Completion with Large Language Models

This repository includes supplementary information and detailed descriptions related to our paper, located in the **"MC_Preprint_and_appendix" pdf document**. This includes details about our prompt formulation (serialization format, ChatGPT instructions, and few-shot examples from our running example), information about our candidate generation, insights into our retrieval mechanism and an additional overview about related work. Please scroll down in the document to **appendix** to find this additional information. The main part of the paper is the same as in the submission. 

Furthermore this repository contains all files to reproduce the results for the paper **Software Model Evolution with Large Language Models: Experiments on Simulated, Public, and Industrial Datasets**.

## How to install

Simply run 

```bash
python -m pip install -r requirements.txt
```
to install all the necessary requirements. Furthermore, for using the OpenAI API, you nead to provide your API key, that you can see in your OpenAI account. 

Just copy this key in a file `/secrets/openai.key` in the trunk of this repository.
Now, you are ready to go!

If you are using OpenAI via Microsoft Azure, you can proide two files in the `secret` folder.
The first `api_key_openai.key` is the API key, the second `azure.url` file, that contains the url to the subscription URL.

The current scripts use Azure by default. If you do not want to use Azure, you have to set the `azure` parameter of the `load_openai_key()` method to `False` in the script `GPT_runner.py``.

## Datasets
This replication package contains a folder `datasets`, which includes all input (and output) data from the experiments. We can not disclose the `Industry` dataset, because, as a productive system, this data is highly confidential.

Inside the `datasets` folder, there is one folder per `dataset`. These folder themselves contain a folder named `diffgraphs` that contains the simple change graph input. We hade to **cure** the `RepairVision` (Revision) dataset, because some models did not contain any history. The curated version of the original input dataset for `RepairVision` is contained in the folder `orig_curated`.

The folders `component`, `fine_tuned_ds`, `experiment_samples`, and `few_shot_samples` include the intermediate results that we obtained when running the pipeline. Inside the folder `few_shot_samples`, there is a file `few_shot_dataset.jsonl`. This file is contains the final input data that is then used together with the large language model, to provide the software model completions.

The folder `results` include the results that we got from `GPT-4` and (`GPT-3` in case of fine-tuning).
The files `results_iso_check.jsonl`, additional include the evaluation via the graph isomorphism check. 

For the `Synthetic`  dataset, the `results` folder additionally includes two folders for **Experiment 3** and **Experiment 4** results, that is the results of the fine-tuning, the results of the completion evaluation of the fine-tuned language models, and the results of the comparison of few-shot learning and fine-tuning.

## How to Run

The scripts in this repository consumes the JSON simple change graphs that are part of the replication package.

This repository also contains some functionality for edit operation mining that we do not use directly for this paper.

**Step 1:**
The main script for this paper is the script `pipeline_few_shot_example_generation.sh`. In this script, you can set the path to the input files (i.e., the simple change graphs). The script then automatically computes the graph components, creates training and test data and computes the vector database for semantic retrieval.

**Step 2:** 
At the end, the files for the experiments are generated that are consumed by the script `GPT_runner.py` that calls the Language Model API to generate completions. This script is called like 

```bash
python GPT_runner.py [input_path] [output_path] [SINGLE_EDGE/MULTI_EDGE]
```
, where `input_path` is the same path that has been used as `LOCATION_FEW_SHOT_DATASET` parameter in the `pipeline_few_shot_example_generation.sh` script + the filename `few_shot_dataset.jsonl`. `output_path` can be chosen arbitrarily. The current default setting for the pipeline generate "single edge" completion examples only, so you should choose `SINGLE_EDGE`.

**Step 3:**
This runner gives you the completion results in the form of a JSON List. You can evaluate the structural and type correctness of this via the `eval_scripts/eval_completions.py`. Run this script as
```bash
eval_completions.py [input_path] [output_path] [synthetic_dataset_switch (optional: defaults to False)]
```
, where `input_path` is points to the output of the `GPT_runner.py`. `output_path` can be chosen arbitrarily. The last parameter can be left empty or should be set to `True` for the *Synthetic* dataset.

This last step gives you the files that we then evaluate statistically in the Jupyter Notebooks (see Section `Statistical Evaluation``.)

### Edit Operation Mining (Not relevant for this paper)
There are three main steps to learn edit operations (or network motivs in general) from a set of labeled graphs (representing model differencs as described in the paper).
-  Create the fine-tuning datasets
-  Collect the fine-tuning results
-  Read out the edit operations (~ network motifs)

These three steps correspond to the three shell scripts in the trunk of this repository:
-  pipeline_part_1.sh
-  pipeline_part_2.sh
-  pipeline_part_3.sh

Since the fine-tuning can take some time (it can even happen that the queue is blocked for long jobs), the second part should be executed when all the fine-tuning jobs are finished.
You can check this by fetching the results for the last dataset from the API:

```bash
openai api fine_tunes.follow -i [YOUR_FINETUNE_ID]
```

You have to replace the id for the finetuning jobs by the id of the last job, which you will find in the file `/results/finetune/jobs/results.csv` in the subfolder for your dataset.

## Statistical Evaluation
**Note that the experiments numbering in the code and repository is different from the paper.
In the paper, for better presentation purposes, we spilt the Experiment 1 in this repository into two experiments (Experiment 1 (RQ1) and Experiment 2 (RQ2)). The Experiment refernced here corresponds to Experiment 3 (RQ3) in the paper. Finally, in the paper, we merge Experiment 3 and 4 in this paper to Experiment 4 (RQ4). We do remove the focus on fine-tuning the in the paper and therefore condense the results from Experiment 3 and 4 in this repository.**

We have provided our evaluation scripts as Jupyter Notebook (and for one evaluation as R Script) in this repository in the folder `eval_scripts`.

### Experiment 1:
The evaluation for this experiment is done in `eval_scripts/llm4model_completion_paper_experiment_1.ipynb`.
For all of our three datasets, this evaluation works on a `csv`-file provided as part of the results data.
This includes the final manual (for the industry dataset) and automatic isomorphism test results for the other two datasets.
We are currently discussing with our industry partner, if the results for the industry dataset can be disclosed.

### Experiment 2:
The results of the second experiment were obtained by a manual analysis. The comments (evaluating every single sample) were recorded in a CSV file.
We are currently discussing with our industry partner, if these comments can be disclosed.

### Experiment 3:
The evaluation for this experiment is done in `eval_scripts/llm4model_completion_paper_experiment_3.ipynb` and `EOMining_Experiment_2.R`.
The evaluation uses the following results from the data and results folder (part of the `Synthetic` dataset): `results_completion_ada_6_D10E31p0.csv`,
`results_completion_curie_6_D20E51p0.csv`,
`all_results_ada.csv`,
`all_results_curie.csv`,
`all_results_davinci.csv`.

### Experiment 4: 
To run the final evaluation, run the jupyter notebook `eval_scripts/llm4model_completion_paper_experiment_4.ipynb`
The scripts use the four files to the load all post-processed results of the Ada, Curie and RAG model. We have provided these input files separately.
The final result includes an additional evaluation of the completions generated by the models. The completions have been processed and converted into a uniform format.
Subsequently, we transformed this data into graphs to compare the completions with the ground truth, checking for isomorphism and sub-isomorphism among other metrics.
