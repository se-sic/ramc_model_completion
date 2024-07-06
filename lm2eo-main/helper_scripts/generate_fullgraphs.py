import json
import os


def generated_full_graphs_and_change_type(folder_input, folder_output):
    overall_prompts=[]
    overall_graph_ids = 0
    for file in os.listdir(folder_input):
        file_input = os.path.join(folder_input, file)
        if os.path.isfile(file_input):
            with open(file_input, 'r') as f:

                for line in f:
                    try:
                        overall_prompts.append(json.loads(line.strip()))

                    except json.JSONDecodeError as e:
                        print(f"Invalid JSON: {e}")

            for entry in overall_prompts:
                entry['full_graph'] = entry['prompt'] + ' ' + entry['completion']
              #  entry['change_type'] = "add_edge"



                entry['graph_id'] = str(overall_graph_ids)
                overall_graph_ids+=1


               # metadata["token_count"] = record.get("token_count")
             #   metadata["number_of_edges_graph"] = record.get("number_of_edges_graph")
              #  metadata["number_of_removed_items"] = record.get("number_of_removed_items")
              #  metadata["last_edge_change_type"] = record.get("change_type")[-1]


            #updated_train_version.jsonl'
            file_name ="/updated_"
            if "test" in file:
                file_name+="test"
            else:
                file_name += "train"

            file_name += "_version.jsonl"



            with open(folder_output+ file_name , 'w') as f:
                for entry in overall_prompts:
                    json_str = json.dumps(entry)
                    f.write(f"{json_str}\n")



if __name__ == '__main__':

    generated_full_graphs_and_change_type("../input_new_shot_original_curie", "../input_few_shot_database_curie/curie_examples")
