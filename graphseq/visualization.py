import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(graph, title):
    pos = nx.spring_layout(graph)
    labels = nx.get_node_attributes(graph, 'nucleotide')
    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos, with_labels=True, labels=labels, node_color='skyblue', node_size=1500, edge_color='black', linewidths=1, font_size=15)
    plt.title(title)
    plt.show()
