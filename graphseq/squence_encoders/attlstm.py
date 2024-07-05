import torch
import torch.nn as nn

class AttLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(AttLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        attn_weights = F.softmax(self.attention(out), dim=1)
        out = torch.bmm(attn_weights.transpose(1, 2), out)
        return out

def save_attlstm_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_attlstm_model(filepath, input_size, hidden_size, num_layers):
    model = AttLSTM(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(filepath))
    return model