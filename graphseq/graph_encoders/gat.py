import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv

class GAT(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers, num_heads, activation, dropout):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # input layer
        self.layers.append(GATConv(in_feats, hidden_feats, num_heads))
        # hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden_feats * num_heads, hidden_feats, num_heads))
        # output layer
        self.layers.append(GATConv(hidden_feats * num_heads, out_feats, 1))

    def forward(self, g, features):
        x = features
        for layer in self.layers[:-1]:
            x = self.dropout(x)
            x = layer(g, x).flatten(1)
            x = self.activation(x)
        x = self.layers[-1](g, x).mean(1)
        return x

def save_gat_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_gat_model(filepath, in_feats, hidden_feats, out_feats, num_layers, num_heads, activation, dropout):
    model = GAT(in_feats, hidden_feats, out_feats, num_layers, num_heads, activation, dropout)
    model.load_state_dict(torch.load(filepath))
    return model