import torch
from transformers import BertTokenizer, BertModel

class BERTSequenceEncoder(nn.Module):
    def __init__(self):
        super(BERTSequenceEncoder, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, sequences):
        inputs = self.tokenizer(sequences, return_tensors='pt', padding=True, truncation=True)
        outputs = self.bert(**inputs)
        return outputs.last_hidden_state.mean(dim=1)

def save_bert_model(model, filepath):
    torch.save(model.state_dict(), filepath)

def load_bert_model(filepath, pretrained_model_name='bert-base-uncased'):
    model = BERTSequenceEncoder()
    model.load_state_dict(torch.load(filepath))
    return model