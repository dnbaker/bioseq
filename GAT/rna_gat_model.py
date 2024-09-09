import torch
import torch.nn as nn
from embedding_module import EmbeddingModule, embedding_dim, vocab_size, mask_token_id
from gat_module import GraphAttentionTransformer


class RNA_GAT_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, gat_out_channels, gat_heads, gat_layers, dropout):
        super(RNA_GAT_Model, self).__init__()
        self.embedding_module = EmbeddingModule(vocab_size, embedding_dim)
        self.gat_module = GraphAttentionTransformer(embedding_dim, gat_out_channels, heads=gat_heads, dropout=dropout,
                                                    num_layers=gat_layers)

    def forward(self, x, edge_index):
        x = self.embedding_module(x)
        x = x.view(-1, x.size(2))  # Flatten the sequence length dimension
        x = self.gat_module(x, edge_index)
        return x


gat_out_channels = 768
gat_heads = 8
gat_layers = 10
dropout = 0.6

rna_gat_model = RNA_GAT_Model(vocab_size + 1, embedding_dim, gat_out_channels, gat_heads, gat_layers,
                              dropout)  # +1 for the mask token
