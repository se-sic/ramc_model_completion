###############################################
###############################################
#### Module: Remove duplicates ################
### and compute counting statistics ###########
###############################################
###############################################
from parse_utils import import_tlv_folder, export_TLV
from isograph import IsoGraph, remove_duplicates
import sys
import os
import time
from graph_lattice import LatticeNode, Lattice, Statistics


def main(graph_db_path: str, subgraphs_path:str, results_dir:str):
    subgraphs = import_tlv_folder(subgraphs_path, parse_support=False)
    
    #TODO REMOVE THIS AGAIN, THIS IS ONLY TO FIT TO THE TEST DATA
    #subgraphs = [graph.reverse() for graph in subgraphs]
    
    #TODO Workaround since a dummy root has been added by a previous steps
    subgraphs = [IsoGraph(graph).cut_root() for graph in subgraphs]
        
    # Get rid of clones
    nb_initial_subgraphs = len(subgraphs)
    print("Removing duplicates. This might take some time...")
    subgraphs = remove_duplicates(subgraphs)
    nb_pruned_subgraphs = len(subgraphs)
    removed_duplicates = nb_initial_subgraphs - nb_pruned_subgraphs
    print("Removed %d duplicates" % removed_duplicates)
    
    print("Creating subgraph lattice for lattice-based counting...")
    # First create the lattice node for the subgraphs
    lattice_nodes = [LatticeNode(subgraph) for subgraph in subgraphs]
    # Create lattice (this might take some time)
    lattice = Lattice(lattice_nodes)  
    
    print("Exporting lattice.")
    nx_lattice = lattice.to_networkx()
    export_TLV([nx_lattice], results_dir + 'lattice.lg')
    #plot_graphs([nx_lattice], results_dir + 'lattice.png')
    #plot_graph_dot(nx_lattice, results_dir + 'lattice_dot.png')
    with open(results_dir + 'lattice.graphml', 'w') as f:
        f.write(lattice.to_graphml())
                       
    # Write subgraphs without clones
    print("Writing subgraphs without occurrences.")
    export_TLV(subgraphs, results_dir + 'subgraph_candidates.lg')

    for folder in os.listdir(graph_db_path):
        if not os.path.isdir(graph_db_path + "/" + folder + "/mining"):
            continue
        # Read db
        print(f"Parsing graph database for data set {folder}")
        graph_db = import_tlv_folder(graph_db_path+"/"+folder+"/mining/", parse_support=False)
        compute_statistics(graph_db, subgraphs, lattice, results_dir + "/" + folder + "/", folder)

def get_url_for_project(project): 
    ''' 
    Looks up the given project in a list of projects and returns the corresponding repository url.
    '''
    # TODO project list could be cached
    with open('project_list.md', 'r') as project_list:
        while True:
            line = project_list.readline()
            if not line:
                break
            tokens = line.split('|')
            # project name is token 0, repository url is token 4
            if tokens[0].strip() == project:
                return tokens[4]

def write_as_md(stats, save_path, project_name):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w', newline='') as mdfile:
        for subgraph in stats.subgraphs:
            mdfile.write(f"# {subgraph.name}\n")
            mdfile.write(f"![{subgraph.name}]({subgraph.name}.png)\n")
            occurrences_transaction = stats.occurrences_transaction[subgraph.name] if subgraph.name in stats.occurrences_transaction.keys() else 0
            occurrences_references = stats.occurrences_references[subgraph.name] if subgraph.name in stats.occurrences_references.keys() else []
            mdfile.write(f"Frequency: {occurrences_transaction}\n\n")
            mdfile.write("Put your notes here\n\n")
            mdfile.write("<details><summary>Matches</summary><p>\n")
            for occurrence in occurrences_references:
                tokens = occurrence.split('$$$')
                if not len(tokens) >= 2:
                    print(f"Illegal occurence string {occurrence}")
                    continue
                commit_id = tokens[1].strip()‚
                file_path = tokens[0].strip()
                # only works for commits, which are not merge commits
                github_commit_url = get_url_for_project(project_name) + '/commit/'  + commit_id
                mdfile.write(f"Commit: {github_commit_url}\n")
                mdfile.write(f"File: {file_path}\n\n")
            mdfile.write("</p></details>\n\n")

def compute_statistics(graph_db, subgraphs, lattice, results_dir, dataset_name):
    # Compute statistics
    print("Counting the subgraph occurrences in the graph database. This might take some time...")
    stats = Statistics(graph_db, subgraphs, lattice=lattice)
    #start = time.time()
    #stats.compute_occurrences_brute_force()
    #print(stats.occurrences_transaction)
    #stop = time.time()
    #print("Computing occurrences brute force took %f seconds." % (stop-start))
    start = time.time()
    # TODO if this sill takes to long, think about some tricks, e.g., parallelizing,...
    stats.compute_occurrences_lattice_based()
    stop = time.time()
    print("Computing occurrences lattice based took %f seconds." % (stop-start))
    
    # Write statistics to file
    print("Write occurrence statistics...")
    stats.write_as_csv(results_dir + 'occurrence_stats.csv', dataset_name)
    write_as_md(stats, results_dir + 'occurrences_stats.md', dataset_name)

    print("Done")

 
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Three arguments expected: path to graph database folder, path to subgraph database folder, path to results directory")
    
    # Create output folder if it doesn't exist yet
    os.makedirs(sys.argv[3], exist_ok=True)    
    main(sys.argv[1], sys.argv[2], sys.argv[3])
