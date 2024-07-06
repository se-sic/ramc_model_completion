#!/bin/bash


################## Step 1 ########################
# Generate pattern candidates via FSM            #
# Input: Components                              #
# Output: pattern candidates + meta-info         # 
##################################################
echo "Step 1: Generating components via FSM"
python_version=$(python3 --version)
echo "Using python version $python_version"
# TODO automatically make sure that python 3 is used

target_subtree_count_for_threshold_estimation=300
threshold=6 # 0 means read threshold from files
timeout_mining=120
min_size=4
max_size=15


run_dataset(){
	echo "Input:  $1"
	echo "Output:  $2"
	echo "Libs:  $3"

	# Definition of output directories
	data_set_name="$4"
	input_path="$1"
	output_mining="$2mining/"
	output_mining_no_duplicates="$2mining_no_duplicates/"
	lib_path="$3"
	parsemis_path="$3parsemis.jar"

	echo "Running dataset: $data_set_name"
	mkdir -p "$2"

	# Step 2 - Compute thresholds - Not better than fixed threshold for Linux dataset
	#python bisect_threshold_search.py $lib_path $output_filtered $target_subtree_count_for_threshold_estimation

	# Step 3 - Mining
	python ./mining/run_parsemis.py "$parsemis_path" "$input_path" "$output_mining" $threshold $min_size $max_size $timeout_mining

	# Step 4 - Remove duplicates
	python ./mining/remove_duplicates.py "$output_mining" "$output_mining_no_duplicates" "$data_set_name" 
}

# Dataset location
LOCATION='./case_studies/paper_fsm_hard/'

COMPONENTS_PATH=$LOCATION'results/components/'
OUTPUT_BASE=$LOCATION'results/pattern_candidates/'
COMPONENTS_PATH=$LOCATION'results/components/'
CORRECT_PATTERNS_PATH=$LOCATION'correct_graphs/'
LIB_PATH='./mining/lib/'

echo "Input: $COMPONENTS_PATH"
echo "Output:  $OUTPUT_BASE"
echo "Libs:  $LIB_PATH"

mkdir -p "$OUTPUT_BASE"


# Run for every dataset in input folder
for input_folder in "$COMPONENTS_PATH"/*/ ; do
	dataset="$(basename "$input_folder")"
	output_base="$OUTPUT_BASE/$threshold/$dataset/"
	run_dataset "$input_folder" "$output_base" "$LIB_PATH" "$dataset"
done

# Run evaluation
python ./generate_model_transformation_candidates_fsm.py $COMPONENTS_PATH'results.csv' $CORRECT_PATTERNS_PATH $LOCATION