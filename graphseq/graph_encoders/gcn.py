# scripts/graph_encoders/gcn.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, edge_index):
        row, col = edge_index
        deg = torch.bincount(row)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        out = torch.matmul(edge_index, x)
        out = norm.view(-1, 1) * out
        return self.linear(out)


class GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(GCN, self).__init__()
        self.layer1 = GCNLayer(in_features, hidden_features)
        self.layer2 = GCNLayer(hidden_features, out_features)

    def forward(self, x, edge_index):
        x = F.relu(self.layer1(x, edge_index))
        x = self.layer2(x, edge_index)
        return x


def save_gcn_model(model, filepath):
    torch.save(model.state_dict(), filepath)


def load_gcn_model(filepath, in_features, hidden_features, out_features):
    model = GCN(in_features, hidden_features, out_features)
    model.load_state_dict(torch.load(filepath))
    return model
