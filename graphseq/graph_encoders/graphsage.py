import torch
import torch.nn as nn
from dgl.nn.pytorch.conv import SAGEConv

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats, num_layers, activation, dropout, aggregator_type='mean'):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # input layer
        self.layers.append(SAGEConv(in_feats, hidden_feats, aggregator_type))
        # hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_feats, hidden_feats, aggregator_type))
        # output layer
        self.layers.append(SAGEConv(hidden_feats, out_feats, aggregator_type))

    def forward(self, g, features):
        x = features
        for layer in self.layers:
            x = self.dropout(x)
            x = layer(g, x)
        return x

def save_graphsage_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_graphsage_model(filepath, in_feats, hidden_feats, out_feats, num_layers, activation, dropout, aggregator_type='mean'):
    model = GraphSAGE(in_feats, hidden_feats, out_feats, num_layers, activation, dropout, aggregator_type)
    model.load_state_dict(torch.load(filepath))
    return model