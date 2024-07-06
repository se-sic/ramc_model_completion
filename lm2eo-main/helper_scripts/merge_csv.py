# Helper script to handle the completion sample description csv file.

import pandas as pd

PATH_RESULTS='./model_completion_dataset/Siemens_Mobility/results/few_shot_samples/downsampled_few_shot_dataset_results.jsonl'
PATH_CSV='./model_completion_dataset/Siemens_Mobility/results/few_shot_samples/stats.csv'

SELECTED_COLUMNS = ['correct_format', 'type_isomorphic_completion', 'change_type_isomorphic_completion', 'structural_isomorphic_completion', 'completion_type', 'change_type', 'detail_type', 'context_type', 'few_shot_count',  'similar_few_shot', 'comment' , 'in_sample', 'comment_completion', 'correctness', 'interesting']

df_csv = pd.read_csv(PATH_CSV, index_col='sample_id')
df_results = pd.read_json(PATH_RESULTS, lines=True)

# ensure id is integer
df_results['sample_id'] = df_results['id'].astype(int)

# Left join (csv is leading)
merged_df = df_csv.merge(df_results, left_on='sample_id', right_on='sample_id', how='outer', suffixes=(None,'_right'))

sampled_ids = set(df_results['sample_id'])

for idx, row in merged_df.iterrows():
    if row['sample_id'] in sampled_ids:
        merged_df.loc[idx, 'in_sample'] = 'yes'
    else:
        merged_df.loc[idx, 'in_sample'] = 'no'

# Set index
merged_df['sample_id'] = merged_df.sample_id.astype(int)
merged_df.set_index('sample_id', inplace=True)
        
# select columns to keep
merged_df = merged_df[SELECTED_COLUMNS]
 
merged_df.to_csv(PATH_CSV, index_label='sample_id')