import torch
import torch.nn as nn
import numpy as np

class EmbeddingModule(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingModule, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(5000, embedding_dim)  # Assuming max length of 5000

    def forward(self, x):
        seq_length = x.size(1)
        positions = torch.arange(0, seq_length, device=x.device).unsqueeze(0).expand_as(x)
        return self.embedding(x) + self.position_embedding(positions)

    def mask_input(self, x, mask_token_id, mask_prob=0.15):
        mask = np.random.rand(*x.shape) < mask_prob
        x_masked = x.clone()
        x_masked[mask] = mask_token_id
        return x_masked, mask

embedding_dim = 768
vocab_size = len("AGCUX-")
mask_token_id = vocab_size  # Assuming the last token in the vocabulary is used as the mask token
embedding_module = EmbeddingModule(vocab_size + 1, embedding_dim)  # +1 for the mask token
