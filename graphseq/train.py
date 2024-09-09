# scripts/integrate_and_execute.py
import os
import torch
import dgl
import numpy as np
from data_preparation import read_fasta, construct_knowledge_graph, construct_secondary_structure_graph
from visualization import visualize_graph
from sequence_encoders.xlstm import xLSTM, save_xlstm_model, load_xlstm_model
from sequence_encoders.lstm import LSTM, save_lstm_model, load_lstm_model
from sequence_encoders.bilstm import BiLSTM, save_bilstm_model, load_bilstm_model
from sequence_encoders.attlstm import AttLSTM, save_attlstm_model, load_attlstm_model
from sequence_encoders.bert import BERTSequenceEncoder, save_bert_model, load_bert_model
from graph_encoders.gcn import GCN, save_gcn_model, load_gcn_model
from graph_encoders.graphsage import GraphSAGE, save_graphsage_model, load_graphsage_model
from graph_encoders.gat import GAT, save_gat_model, load_gat_model
from distillation import train_with_distillation

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define model parameters
input_size = 4  # Assuming one-hot encoding of RNA sequences (A, U, C, G)
hidden_size = 128
num_layers = 2
dropout = 0.5

# Initialize sequence models
sequence_models = {
    "xLSTM": xLSTM(input_size, hidden_size, num_layers).to(device),
    "LSTM": LSTM(input_size, hidden_size, num_layers).to(device),
    "BiLSTM": BiLSTM(input_size, hidden_size, num_layers).to(device),
    "AttLSTM": AttLSTM(input_size, hidden_size, num_layers).to(device),
    "BERT": BERTSequenceEncoder().to(device)
}

# Define graph model parameters
in_feats = 4  # Input feature size (e.g., one-hot encoded nucleotides)
hidden_feats = 128  # Hidden feature size
out_feats = 128  # Output feature size (latent space size)
num_layers_gcn = 3  # Number of GCN layers
activation = torch.nn.functional.relu  # Activation function
dropout = 0.5  # Dropout rate

# Initialize graph models
graph_models = {
    "GCN": GCN(in_feats, hidden_feats, out_feats).to(device),
    "GraphSAGE": GraphSAGE(in_feats, hidden_feats, out_feats, num_layers_gcn, activation, dropout).to(device),
    "GAT": GAT(in_feats, hidden_feats, out_feats, num_layers_gcn, num_heads=4, activation=activation,
               dropout=dropout).to(device)
}


# Function to train and save models with distillation
def train_and_save_models(data_loader, teacher_model, student_model, model_name, graph_model, graph_model_name):
    # Define loss criterion and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)

    # Train student model with distillation
    student_model = train_with_distillation(teacher_model, student_model, data_loader, criterion, optimizer, device)

    # Save student model
    if model_name == "xLSTM":
        save_xlstm_model(student_model, f'../output/distilled_{model_name}_model.pth')
    elif model_name == "LSTM":
        save_lstm_model(student_model, f'../output/distilled_{model_name}_model.pth')
    elif model_name == "BiLSTM":
        save_bilstm_model(student_model, f'../output/distilled_{model_name}_model.pth')
    elif model_name == "AttLSTM":
        save_attlstm_model(student_model, f'../output/distilled_{model_name}_model.pth')
    elif model_name == "BERT":
        save_bert_model(student_model, f'../output/distilled_{model_name}_model.pth')


# Process each .fa file in the samples directory
data_dir = '../data/samples'
output_dir = '../output/results'
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(data_dir):
    if filename.endswith('.fa'):
        file_path = os.path.join(data_dir, filename)
        sequences = read_fasta(file_path)

        for i, sequence in enumerate(sequences):
            # Construct knowledge and secondary structure graphs
            knowledge_graph = construct_knowledge_graph(sequence)
            secondary_structure_graph = construct_secondary_structure_graph(sequence)

            # Visualize graphs
            visualize_graph(knowledge_graph, f"Knowledge Graph - {filename} - Sequence {i + 1}")
            visualize_graph(secondary_structure_graph, f"Secondary Structure Graph - {filename} - Sequence {i + 1}")

            # Encode sequences
            encoded_sequence = encode_one_hot(sequence)

            for model_name, model in sequence_models.items():
                z_combined_sequence = encode_sequences(model, encoded_sequence, device)

                # Create DGL graph for GCN
                g = dgl.DGLGraph()
                g.add_nodes(len(sequence))
                for j in range(len(sequence) - 1):
                    g.add_edge(j, j + 1)
                g = dgl.add_self_loop(g)
                features = torch.tensor(encoded_sequence, dtype=torch.float32).to(device)

                for graph_model_name, graph_model in graph_models.items():
                    # Forward pass to get graph latent features
                    graph_latent_features = graph_model(g, features)

                    # Combine graph and sequence latent features
                    combined_latent_features = torch.cat([graph_latent_features, z_combined_sequence.unsqueeze(0)],
                                                         dim=1)

                    # Save combined latent features to file
                    output_file = os.path.join(output_dir,
                                               f"{filename}_seq{i + 1}_{model_name}_{graph_model_name}_latent_features.npy")
                    np.save(output_file, combined_latent_features.cpu().numpy())
                    print(
                        f"Saved combined latent features for {filename} sequence {i + 1} with {model_name} and {graph_model_name} to {output_file}")

                    # Initialize student model
                    student_model = model.__class__(*model.args).to(device)  # Assuming the model has an args attribute

                    # Train and save models with self-distillation
                    train_and_save_models(data_loader, model, student_model, model_name, graph_model, graph_model_name)
