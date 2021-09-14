import bioseq
import torch
import torch.nn as nn
from bioseq import cbioseq
import torch.nn as nn
import linear_attention_transformer as linformer
import fast_transformer_pytorch as fast_transformer
from fast_transformer_pytorch import FastTransformer
from fast_transformer_pytorch.fast_transformer_pytorch import FastAttention
from linear_attention_transformer.linear_attention_transformer import SelfAttention as LinSelfAttention
from linear_attention_transformer.autoregressive_wrapper import AutoregressiveWrapper
from x_transformers.x_transformers import DEFAULT_DIM_HEAD
from x_transformers import XTransformer, AutoregressiveWrapper as XAutoregressiveWrapper, Encoder as XEncoder, CrossAttender, Decoder as XDecoder
from h_transformer_1d import HTransformer1D


class FastEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,
        dim,
        depth,
        max_seq_len,
        heads = 8,
        dim_head = 64,
        ff_mult = 4,
        absolute_pos_emb = False
    ):
        from fast_transformer_pytorch.fast_transformer_pytorch import FeedForward, RotaryEmbedding, PreNorm
        super().__init__()
        self.token_emb = nn.Embedding(num_tokens, dim)

        # positional embeddings

        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if absolute_pos_emb else None

        layer_pos_emb = None
        if not absolute_pos_emb:
            assert (dim_head % 4) == 0, 'dimension of the head must be divisible by 4 to use rotary embeddings'
            layer_pos_emb = RotaryEmbedding(dim_head // 2)

        # layers

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            attn = FastAttention(dim, dim_head = dim_head, heads = heads, pos_emb = layer_pos_emb, max_seq_len = max_seq_len)
            ff = FeedForward(dim, mult = ff_mult)

            self.layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, ff)
            ]))

        # weight tie projections across all layers

        first_block, _ = self.layers[0]
        for block, _ in self.layers[1:]:
            block.fn.to_q_attn_logits = first_block.fn.to_q_attn_logits
            block.fn.to_k_attn_logits = first_block.fn.to_k_attn_logits

        # to logits

        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_tokens)
        )

    def forward(
        self,
        x,
        mask = None
    ):
        n, device = x.shape[1], x.device

        if self.abs_pos_emb is not None:
            pos_emb = self.abs_pos_emb(torch.arange(n, device = device))
            x = x + rearrange(pos_emb, 'n d -> () n d')

        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x

        return self.to_logits(x)

class TokenizerLayer(nn.Module):
    def __init__(self, tokenizer, *, padlen, batch_first=False, nthreads=-1, destchar='i'):
        super().__init__()
        assert padlen >= 0
        self.tokenizer = tokenizer
        self.pad = padlen
        self.batch_first = batch_first
        self.destchar = destchar
        self.nthreads = nthreads if nthreads > 0 else 1
    def forward(self, inputs):
        return self.tokenizer.batch_tokenize(inputs, padlen=self.pad, batch_first=self.batch_first, nthreads=self.nthreads, destchar=self.destchar)


class SeqEncoder(nn.Module):
    '''
        emb_dim = None,
        max_mem_len = 0.,
        emb_dropout = 0.,
        num_memory_tokens = None,
        tie_embedding = False,
        use_pos_emb = True
    '''
    def __init__(self, tokenizer, embedding, encoder_type, emb_dropout=.1, *args, **kwargs):
        super().__init__()
        self.tokenizer = tokenizer
        assert isinstance(self.tokenizer, Tokenizer) or isinstance(self.tokenizer, TokenizerLayer)
        assert hasattr(embedding, "forward")
        self.embedding = embedding
        self.emb_dropout = nn.Dropout(emb_dropout)
        self.encoder = encoder_type(*args, **kwargs)
        assert hasattr(self.encoder, "forward")

    def forward(self, inputs, device=None):
        """
            Tokenizes
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        from torch import from_numpy, long as torchlong, int as torchint
        from x_transformers import Encoder
        tokens = self.tokenizer(inputs)
        print(tokens.dtype, tokens.shape, tokens)
        tokens = from_numpy(tokens).to(device)
        print("tokens", tokens.shape)
        embs = self.embedding(tokens)
        print("embs", embs.shape)
        embs = self.emb_dropout(embs)
        encoding = self.encoder(embs)
        print("encoding", encoding.shape)
        return encoding

class FFArgs:
    ''' Helper class holdering FFArguments
    '''
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0., zero_init_output=False):
        self.dim = dim
        self.dim_out = dim_out
        self.mult = mult
        self.glu = glu
        self.dropout = dropout
        self.zero_init_output

    def dict(self):
        return {"ff_" + key: value for key, value in self.__dict__.items()}



class AttnArgs:
    def __init__(self, dim, dim_head = DEFAULT_DIM_HEAD,
                 heads = 8, causal = False, mask = None,
                 talking_heads = False, collab_heads = False,
                 collab_compression = .3, sparse_topk = None,
                 use_entmax15 = False, num_mem_kv = 0,
                 dropout = 0., on_attn = False, gate_values = False,
                 zero_init_output = False):
        self.dim = dim; self.dim_head = dim_head
        self.causal = causal
        self.mask = mask
        self.collab_compression = collab_compression
        self.collab_heads = collab_heads
        self.use_entmax15 = use_entmax15
        self.sparse_topk = sparse_topk
        self.num_mem_kv = num_mem_kv
        self.dropout = dropout
        self.on_attn = on_attn
        self.gate_values = gate_values
        self.zero_init_output = zero_init_output

    def dict(self):
        return {"attn_" + key: value for key, value in self.__dict__.items()}


Tokenizer = cbioseq.Tokenizer
TransformerLMs = [linformer.LinearAttentionTransformerLM]
Transformers = [FastTransformer, linformer.LinearAttentionTransformer, XTransformer]
AutoTransformers = [linformer.AutoregressiveWrapper, XAutoregressiveWrapper]
Attentions = [FastAttention, LinSelfAttention]
Encoders = [XEncoder, FastTransformer, FastEncoder, HTransformer1D]
Decoders = [XDecoder]


class DifferentiableSparseSoftmax(nn.Module):
    def __init__(self, alpha_init=1.5, n_iter=24, dtype=torch.float32, device=torch.device("gpu" if torch.cuda.is_available() else "cpu"),
                 reduction='sum'):
        super().__init__()
        from entmax import EntmaxBisectLoss
        self.loss = EntmaxBisectLoss(
            torch.tensor([alpha_init], dtype=dtype, device=device),
            n_iter=n_iter, reduction=reduction)

    def forward(self, *args, **kwargs):
        return self.loss(*args, **kwargs)



if __name__ == "__main__":
    # Test things!
    embdim = 32
    attndim = 128
    headdim = 64
    nlayers = 4
    nheads = 8
    emb_dropout = .15
    tokl = TokenizerLayer(bioseq.DNATokenizer, padlen=250, batch_first=True, destchar='i')
    emb = bioseq.make_embedding(bioseq.DNATokenizer, embdim, norm_type=2.0, sparse=True)

    encoder = SeqEncoder(tokl, emb, FastEncoder, num_tokens=tokl.tokenizer.alphabet_size(), dim=embdim, depth=nlayers, max_seq_len=tokl.pad, heads=nheads, dim_head=headdim, ff_mult=4, absolute_pos_emb=False)
    sfmax = DifferentiableSparseSoftmax()
    Xs = torch.randn(4, 10, dtype=torch.float64, requires_grad=True)
    Ys = torch.max(torch.randn_like(Xs), dim=1)[1]
    print(sfmax(Xs, Ys))
    from random import choice
    seqs = ["".join(choice("ACGT") for i in range(250)) for j in range(500)]
    output = encoder(seqs)
    print(output.shape)
    from timeit import timeit
    print("Time to compute: ", timeit(lambda: encoder(seqs), number=3))
