import torch
import torch.nn as nn

class xLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(xLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.Wi = nn.Linear(input_size, hidden_size)
        self.Wf = nn.Linear(input_size, hidden_size)
        self.Wo = nn.Linear(input_size, hidden_size)
        self.Wc = nn.Linear(input_size, hidden_size)
        self.Ui = nn.Linear(hidden_size, hidden_size)
        self.Uf = nn.Linear(hidden_size, hidden_size)
        self.Uo = nn.Linear(hidden_size, hidden_size)
        self.Uc = nn.Linear(hidden_size, hidden_size)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, hidden):
        h, c = hidden
        i = torch.sigmoid(self.Wi(x) + self.Ui(h))
        f = torch.sigmoid(self.Wf(x) + self.Uf(h))
        o = torch.sigmoid(self.Wo(x) + self.Uo(h))
        c_hat = torch.tanh(self.Wc(x) + self.Uc(h))
        c = f * c + i * c_hat
        h = o * torch.tanh(c)
        return h, (h, c)

class xLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(xLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.layers = nn.ModuleList([xLSTMCell(input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)])

    def forward(self, x, hidden=None):
        seq_len, batch_size, _ = x.size()
        if hidden is None:
            h = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
            c = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        else:
            h, c = hidden

        hiddens = []
        for i in range(self.num_layers):
            h_i, c_i = h[i], c[i]
            layer_output = []
            for t in range(seq_len):
                h_i, (h_i, c_i) = self.layers[i](x[t], (h_i, c_i))
                layer_output.append(h_i.unsqueeze(0))
            x = torch.cat(layer_output, dim=0)
            hiddens.append((h_i, c_i))

        h, c = zip(*hiddens)
        h = torch.stack(h)
        c = torch.stack(c)
        return x, (h, c)

def save_xlstm_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_xlstm_model(filepath, input_size, hidden_size, num_layers):
    model = xLSTM(input_size, hidden_size, num_layers)
    model.load_state_dict(torch.load(filepath))
    return model
