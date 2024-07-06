from typing import List
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout

def get_color_by_label(label: str):
    if label.startswith("Add") or not label.find('"changeType": "Add"') == -1:
        return 'green'
    elif label.startswith("Del") or not label.find('"changeType": "Remove"') == -1:
        return 'red'
    elif label.startswith("Pres") or not label.find('"changeType": "Preserve"') == -1:
        return 'grey'
    elif label.startswith("Change") or not label.find('"changeType": "Change"') == -1:
        return 'blue'
    else:
        return 'black'

def json_pretty_print(input: str) -> str:
    import json
    input = json.loads(input)
    output = json.dumps(input, indent=2, sort_keys=True)
    return output

def get_stylebox(style="round"):
    return dict(boxstyle=style, fc="w", ec="k")

# Plot graphs
def plot_graphs(S: List[nx.Graph], file_path=None, labels=True, json_labels=False):
    for i in range(len(S)):
        plt.clf()
        plt.figure(i, dpi=1000)
        plt.margins(0.1, 0.1)

        pos = nx.spring_layout(S[i], scale=1, k=2)

        # Set colors
        if json_labels:
            node_labels = dict([(v, json_pretty_print(d['label'])) for v, d in S[i].nodes(data=True)])
        else:
            node_labels = dict([(v, d['label']) for (v, d) in S[i].nodes(data=True)])

        base_size=25
        node_size=max([len(label) * base_size for label in node_labels.values()])
        
        color_map = [get_color_by_label(node_labels[v]) for v in S[i].nodes()]
        edge_color_map = [get_color_by_label(data['label']) for _, _,  data in S[i].edges(data=True)]

        if labels:
            y_off = 0#0.1
            x_off = 0.1
            nx.draw_networkx_labels(S[i], pos={k: ([v[0]-x_off, v[1] + y_off]) for k, v in pos.items()}, font_size=4,
                                    labels=node_labels, bbox=get_stylebox(), horizontalalignment="left")
            nx.draw_networkx_edges(S[i], pos, edge_color=edge_color_map, arrows=True, arrowstyle='->', node_size=node_size)
            
            nx.draw_networkx_edge_labels(S[i], pos, font_size=3)
            nx.draw_networkx_nodes(S[i], pos,  node_size=node_size, node_color=color_map)

        else:
            nx.draw(S[i], pos, node_size=node_size)

        if file_path is not None:   # Save figure
            if len(S) > 1:
                save_path = file_path + "_" + str(i) + ".png"
            else:
                save_path = file_path + ".png"

            # Save
            plt.savefig(save_path, format="PNG")
        else: 
            plt.show()

def plot_graph_dot(G, file_path, labels=True):
    # TODO assert graphviz installed

    # same layout using matplotlib with no labels
    plt.title(G.name)
    pos = graphviz_layout(G, prog='dot')
    nx.draw(G, pos, with_labels=labels, arrows=True)
    plt.savefig(file_path)
    

def to_latex(G: nx.Graph, file_path: str, labels: bool=True):
    print(nx.to_latex_raw(G))