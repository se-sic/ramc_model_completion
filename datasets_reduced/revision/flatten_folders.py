import os
import shutil

'''
We have a folder with projects. 
Each project contains multiple models which are represented by folder contained in the project folder.
The model folder contains a history of models files.
'''

def flatten_folders(input_directory, output_directory):
    # root folder contains projects
    for project in os.listdir(input_directory):
        full_project_path = os.path.join(input_directory, project)
        if os.path.isfile(full_project_path):
            print(f"Warn: Unexpected file {full_project_path}")
            continue
        #print(f"Project: {project}")
        for model in os.listdir(full_project_path):
            full_model_path = os.path.join(full_project_path, model)
            if os.path.isfile(full_model_path):
                print(f"Warn: Unexpected file {full_model_path}")
                continue
            #print(f"Model: {model}")
            for revision in os.listdir(full_model_path):
                full_revision_path = os.path.join(full_model_path, revision)
                if os.path.isfile(full_revision_path):
                    print(f"Warn: Unexpected file {full_revision_path}")
                    continue
                #print(f"Revision: {revision}")
                for model_file in os.listdir(full_revision_path):
                    #print(f"File: {model_file}")

                    # We finally want to move the files to 
                    # output/{project}!!{model}/{revision}/file
                    destination_path = os.path.join(output_directory, f"{project}!!{model}", revision)
                    os.makedirs(destination_path, exist_ok=True)
                    shutil.copy(os.path.join(full_revision_path, model_file), os.path.join(destination_path, model_file))
                    
if __name__ == "__main__":
    input_directory = "/Users/z003zckm/projects/model_completion_dataset/revision/orig_curated"  # Replace this with the path to your input directory
    output_directory = "/Users/z003zckm/projects/model_completion_dataset/revision/orig_flat"  # Replace this with the path to your output directory

    flatten_folders(input_directory, output_directory)