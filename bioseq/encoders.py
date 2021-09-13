import bioseq
import torch.nn as nn
import linear_attention_transformer as linformer
import fast_transformer_pytorch as fast_transformer
from fast_transformer_pytorch import FastTransformer
from fast_transformer_pytorch.fast_transformer_pytorch import FastAttention
from linear_attention_transformer.linear_attention_transformer import SelfAttention as LinSelfAttention
from linear_attention_transformer.autoregressive_wrapper import AutoregressiveWrapper
from x_transformers import XTransformer, AutoregressiveWrapper as XAutoregressiveWrapper, Encoder as XEncoder, CrossAttender, Decoder as XDecoder

TransformerLMs = [linformer.LinearAttentionTransformerLM]
Transformers = [FastTransformer, linformer.LinearAttentionTransformer, XTransformer]
AutoTransformers = [linformer.AutoregressiveWrapper, XAutoregressiveWrapper]
Attentions = [FastAttention, LinSelfAttention]
Encoders = [XEncoder, FastTransformer]
Decoders = [XDecoder]


class TokenizerLayer(nn.Module):
    def __init__(self, tokenizer, *, padlen, batch_first=False, nthreads=-1):
        super().__init__(self)
        assert padlen >= 0
        self.tokenizer = tokenizer
        self.pad = padlen
        self.batch_first = batch_first
        self.nthreads = nthreads if nthreads > 0 else 1
    def forward(self, inputs):
        return self.tokenizer.batch_tokenize(inputs, padlen=self.pad, batch_first=self.batch_first, nthreads=self.nthreads)


class SeqEncoder(nn.Module):
    def __init__(self, tokenizer, embedding, encoder_type, *args, **kwargs):
        super().__init__(self)
        self.tokenizer = tokenizer
        self.embedding = embedding
        self.encoder = encoder_type(*args, **kwargs)
