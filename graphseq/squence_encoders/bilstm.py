import torch
import torch.nn as nn

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out

def save_bilstm_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_bilstm_model(filepath, input_size, hidden_size, num_layers):
    model = BiLSTM(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(filepath))
    return model
