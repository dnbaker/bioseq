import networkx as nx
from Bio import SeqIO
import RNA

def read_fasta(file_path):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequences.append(str(record.seq))
    return sequences

def predict_secondary_structure(sequence):
    structure, _ = RNA.fold(sequence)
    return structure

def construct_knowledge_graph(sequence):
    G = nx.DiGraph()
    for i, nucleotide in enumerate(sequence):
        G.add_node(i, nucleotide=nucleotide, x=i)
    for i in range(len(sequence) - 1):
        G.add_edge(i, i + 1, weight=1)
    return G

def construct_secondary_structure_graph(sequence):
    structure = predict_secondary_structure(sequence)
    G = nx.DiGraph()
    for i, nucleotide in enumerate(sequence):
        G.add_node(i, nucleotide=nucleotide, x=i)
    stack = []
    for i, char in enumerate(structure):
        if char == '(':
            stack.append(i)
        elif char == ')':
            j = stack.pop()
            G.add_edge(j, i, weight=1)
    return G
