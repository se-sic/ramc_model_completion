import mining.plot_utils as plt
import mining.parse_utils as parse_utils
import os

def plot_results(graph_db_file, output_dir, labels=True):
    graph_db = parse_utils.import_tlv(graph_db_file, parse_support=False)
    plt.plot_graphs(graph_db, output_dir, labels)    


if __name__ == "__main__":
    os.makedirs('./case_studies/paper_davinci/results/pattern_candidates/Two_Eos20_eo81_p1.0_dfs_edges_davinci_6/linegraphs/plots/', exist_ok=True)
    plot_results('./case_studies/paper_davinci/results/pattern_candidates/Two_Eos20_eo81_p1.0_dfs_edges_davinci_6/linegraphs/pattern_candidates_parsed.lg', './case_studies/paper_davinci/results/pattern_candidates/Two_Eos20_eo81_p1.0_dfs_edges_davinci_6/linegraphs/plots/img')