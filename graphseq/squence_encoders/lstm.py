import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out

def save_lstm_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_lstm_model(filepath, input_size, hidden_size, num_layers):
    model = LSTM(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(filepath))
    return model