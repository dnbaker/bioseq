# scripts/evaluate.py
import networkx as nx
import numpy as np
from sklearn.metrics import jaccard_score, accuracy_score, precision_score, recall_score
import RNA
import difflib
import torch
from load_data import load_graphs, load_sequences

# Function to calculate Graph Edit Distance
def calculate_ged(graph1, graph2):
    return nx.graph_edit_distance(graph1, graph2)

# Function to calculate Jaccard Similarity
def calculate_jaccard_similarity(graph1, graph2):
    nodes1 = set(graph1.nodes())
    nodes2 = set(graph2.nodes())
    return jaccard_score(list(nodes1), list(nodes2))

# Function to calculate Sequence Similarity
def calculate_sequence_similarity(seq1, seq2):
    sm = difflib.SequenceMatcher(None, seq1, seq2)
    return sm.ratio()

# Function to predict RNA secondary structure using RNAfold
def predict_secondary_structure(sequence):
    (structure, mfe) = RNA.fold(sequence)
    return structure

# Function to calculate base pair distance
def calculate_base_pair_distance(structure1, structure2):
    return RNA.bp_distance(structure1, structure2)

# Evaluate Structural Similarity
def evaluate_structural_similarity(generated_graphs, original_graphs):
    ged_scores = []
    jaccard_scores = []
    for g_graph, o_graph in zip(generated_graphs, original_graphs):
        ged_scores.append(calculate_ged(g_graph, o_graph))
        jaccard_scores.append(calculate_jaccard_similarity(g_graph, o_graph))
    return np.mean(ged_scores), np.mean(jaccard_scores)

# Evaluate Sequence Similarity
def evaluate_sequence_similarity(generated_sequences, original_sequences):
    seq_similarities = []
    for g_seq, o_seq in zip(generated_sequences, original_sequences):
        seq_similarities.append(calculate_sequence_similarity(g_seq, o_seq))
    return np.mean(seq_similarities)

# Evaluate Functional Similarity
def evaluate_functional_similarity(generated_sequences, original_sequences):
    bp_distances = []
    for g_seq, o_seq in zip(generated_sequences, original_sequences):
        g_structure = predict_secondary_structure(g_seq)
        o_structure = predict_secondary_structure(o_seq)
        bp_distances.append(calculate_base_pair_distance(g_structure, o_structure))
    return np.mean(bp_distances)

# Evaluate Performance Metrics
def evaluate_performance_metrics(predictions, targets):
    accuracy = accuracy_score(targets, predictions)
    precision = precision_score(targets, predictions, average='weighted')
    recall = recall_score(targets, predictions, average='weighted')
    return accuracy, precision, recall

# Main evaluation function
def evaluate_all():
    original_graphs = load_graphs('../data/original_graphs')
    generated_graphs = load_graphs('../output/generated_graphs')
    original_sequences = load_sequences('../data/original_sequences')
    generated_sequences = load_sequences('../output/generated_sequences')

    # Structural Similarity
    ged_score, jaccard_score = evaluate_structural_similarity(generated_graphs, original_graphs)
    print(f"Graph Edit Distance: {ged_score}, Jaccard Similarity: {jaccard_score}")

    # Sequence Similarity
    seq_similarity = evaluate_sequence_similarity(generated_sequences, original_sequences)
    print(f"Sequence Similarity: {seq_similarity}")

    # Functional Similarity
    bp_distance = evaluate_functional_similarity(generated_sequences, original_sequences)
    print(f"Base Pair Distance: {bp_distance}")

    # Performance Metrics (Dummy example with predictions and targets)
    predictions = [1, 0, 1, 1]
    targets = [1, 0, 1, 0]
    accuracy, precision, recall = evaluate_performance_metrics(predictions, targets)
    print(f"Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}")

if __name__ == "__main__":
    evaluate_all()
