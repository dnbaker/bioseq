import torch
import torch.nn as nn
from torch_geometric.nn import GATConv

class GraphAttentionTransformer(nn.Module):
    def __init__(self, in_channels, out_channels, heads=8, dropout=0.6, num_layers=10):
        super(GraphAttentionTransformer, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_channels, out_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 1):
            self.layers.append(GATConv(out_channels * heads, out_channels, heads=heads, dropout=dropout))
        self.fc = nn.Linear(out_channels * heads, out_channels)

    def forward(self, x, edge_index):
        for layer in self.layers:
            x = layer(x, edge_index)
            x = nn.functional.elu(x)
        x = self.fc(x)
        return x
