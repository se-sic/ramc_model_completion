#!/usr/bin/python  
import os, sys
import re

# The Trafomap
scm_to_sscm = {"Add": "Add", "Remove": "Remove", "Preserve": "Pres", "Component": "Component", "component": "component", "SwImplementation": "Im", "swimplementation": "im", "Requirement": "Requ", "requirement": "requ", "Package": "Package", "package": "package",
  "Connector": "Connector", "connector": "con", "Port": "Port", "port": "port", "src": "src", "_tgt\n": "_t\n"}

sscm_to_scm = dict(zip(scm_to_sscm.values(), scm_to_sscm.keys()))

def replace_dict(input_string, repl_dict):
    for word, replacement in repl_dict.items():
        input_string = re.sub(word, replacement, input_string)
    return input_string

def encode(input_string):
    return replace_dict(input_string, scm_to_sscm)
    
def decode(input_string):
    return replace_dict(input_string, sscm_to_scm)
    
def apply_to_dataset_folder(folder_path, apply_func):
    for data_set in os.listdir(folder_path):
        # Skip files in the input_path
        if not os.path.isdir(folder_path + '/' + data_set):
          continue
        
        transform_folder(folder_path + '/' + data_set + '/', apply_func)


def transform_folder(folder, apply_func):
    for file_to_transform in os.listdir(folder):
        # Read 
        with open(folder + file_to_transform, 'r') as f:
            input_str = f.read()
        # transform
        output_str = apply_func(input_str)

        # Write
        with open(folder + file_to_transform, 'w') as f:
            f.write(output_str)

def encode_folder(folder_path, ds_folder=True):
    if ds_folder:
        apply_to_dataset_folder(folder_path, encode)
    else:
        transform_folder(folder_path, encode)

def decode_folder(folder_path, ds_folder=True):
    if ds_folder:
        apply_to_dataset_folder(folder_path, decode)
    else:
        transform_folder(folder_path, decode)

def main(folder, method, ds_folder=True):
    if method == "enc":
        encode_folder(folder, ds_folder)
    elif method == "dec":
        decode_folder(folder, ds_folder)
    else:
        print("ERROR: Second argument has to be 'enc' for encoding or 'dec' for decoding.")

if __name__ == "__main__":
  if len(sys.argv) == 3:
    main(sys.argv[1], sys.argv[2])
  elif len(sys.argv) == 4:
    ds_folder = sys.argv[3] == "True"
    main(sys.argv[1], sys.argv[2], ds_folder)
  else:
    print("Unexpected number of arguments. At least data_set folder, and enc/dec neccesary")
