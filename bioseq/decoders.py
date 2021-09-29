import sys
import random


import bioseq
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax

from bioseq.hattn import HTransformer1D
from bioseq.softmax import SparseSoftmax


import linear_attention_transformer as linformer
import fast_transformer_pytorch as fast_transformer
from fast_transformer_pytorch import FastTransformer
from linear_attention_transformer.linear_attention_transformer import SelfAttention as LinSelfAttention
from linear_attention_transformer.autoregressive_wrapper import AutoregressiveWrapper as FAutoregressiveWrapper
from x_transformers.x_transformers import DEFAULT_DIM_HEAD
from x_transformers.autoregressive_wrapper import top_k, top_p, top_a
from x_transformers import XTransformer, Encoder as XEncoder, CrossAttender, Decoder as XDecoder
from rotary_embedding_torch import apply_rotary_emb, RotaryEmbedding
from product_key_memory import PKM


random.seed(0)

def exists(x):
    return x is not None


def param_count(x):
    return sum(map(lambda x: x.numel(), x.parameters()))


class FastAttention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        heads = 8,
        dim_head = 64,
        max_seq_len = None,
        pos_emb = None,
        query_sparse_softmax=False,
        key_sparse_softmax=False,
        tied_sparse_softmax=False,
        softmax_alpha=1.5
    ):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # rotary positional embedding

        assert not (exists(pos_emb) and not exists(max_seq_len)), 'max_seq_len must be passed in if to use rotary positional embeddings'

        self.pos_emb = pos_emb
        self.max_seq_len = max_seq_len

        # if using relative positional encoding, make sure to reduce pairs of consecutive feature dimension before doing projection to attention logits

        dim_kvproj = dim_head // (1 + (pos_emb is not None))

        self.to_q_attn_logits = nn.Linear(dim_head, 1, bias = False)  # for projecting queries to query attention logits
        self.to_k_attn_logits = nn.Linear(dim_kvproj, 1, bias = False)  # for projecting keys to key attention logits
        if query_sparse_softmax:
            self.query_softmax = SparseSoftmax(softmax_alpha)
        else:
            self.query_softmax = Softmax(dim=-1)
        self.key_softmax = (self.query_softmax if tied_sparse_softmax else SparseSoftmax(softmax_alpha)) if key_sparse_softmax else Softmax(dim=-1)

        # final transformation of values to "r" as in the paper

        self.to_r = nn.Linear(dim_kvproj, dim_head)

        self.to_out = nn.Linear(inner_dim, dim)

    def forward(self, x, mask = None):
        n, device, h, use_rotary_emb = x.shape[1], x.device, self.heads, exists(self.pos_emb)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: einops.rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        mask_value = -torch.finfo(x.dtype).max
        if mask is not None:
            mask = einops.rearrange(mask, 'b n -> b () n')

        # if relative positional encoding is needed

        if use_rotary_emb:
            freqs = self.pos_emb(torch.arange(self.max_seq_len, device = device), cache_key = self.max_seq_len)
            freqs = einops.rearrange(freqs[:n], 'n d -> () () n d')
            q_aggr, k_aggr, v_aggr = map(lambda t: apply_rotary_emb(freqs, t), (q, k, v))
        else:
            q_aggr, k_aggr, v_aggr = q, k, v

        # calculate query attention logits

        q_attn_logits = einops.rearrange(self.to_q_attn_logits(q), 'b h n () -> b h n') * self.scale
        if mask is not None:
            q_attn_logits = q_attn_logits.masked_fill(~mask, mask_value)
        q_attn = self.query_softmax(q_attn_logits)

        # calculate global query token

        global_q = torch.einsum('b h n, b h n d -> b h d', q_attn, q_aggr)
        global_q = einops.rearrange(global_q, 'b h d -> b h () d')

        # bias keys with global query token

        k = k * global_q

        # if using rotary embeddings, do an inner product between adjacent pairs in the feature dimension

        if use_rotary_emb:
            k = einops.reduce(k, 'b h n (d r) -> b h n d', 'sum', r = 2)

        # now calculate key attention logits

        k_attn_logits = einops.rearrange(self.to_k_attn_logits(k), 'b h n () -> b h n') * self.scale
        if mask is not None:
            k_attn_logits = k_attn_logits.masked_fill(~mask, mask_value)
        k_attn = self.key_softmax(k_attn_logits)

        # calculate global key token

        global_k = torch.einsum('b h n, b h n d -> b h d', k_attn, k_aggr)
        global_k = einops.rearrange(global_k, 'b h d -> b h () d')

        # bias the values

        u = v_aggr * global_k

        # if using rotary embeddings, do an inner product between adjacent pairs in the feature dimension

        if use_rotary_emb:
            u = einops.reduce(u, 'b h n (d r) -> b h n d', 'sum', r = 2)

        # transformation step

        r = self.to_r(u)

        # paper then says to add the queries as a residual

        r = r + q

        # combine heads

        r = einops.rearrange(r, 'b h n d -> b n (h d)')
        return self.to_out(r)


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
        absolute_pos_emb = False,
        query_sparse_softmax=False,
        key_sparse_softmax=False,
        tied_sparse_softmax=False,
        softmax_alpha=1.5
    ):
        softmaxdict = {"query_sparse_softmax": query_sparse_softmax, "key_sparse_softmax": key_sparse_softmax, "tied_sparse_softmax": tied_sparse_softmax, "softmax_alpha": softmax_alpha}
        from fast_transformer_pytorch.fast_transformer_pytorch import FeedForward, RotaryEmbedding, PreNorm
        super().__init__()

        # positional embeddings

        self.abs_pos_emb = nn.Embedding(max_seq_len, dim) if absolute_pos_emb else None

        self.max_seq_len = max_seq_len

        layer_pos_emb = None
        if not absolute_pos_emb:
            assert (dim_head % 4) == 0, 'dimension of the head must be divisible by 4 to use rotary embeddings'
            layer_pos_emb = RotaryEmbedding(dim_head // 2)

        # layers

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            attn = FastAttention(dim, dim_head = dim_head, heads = heads, pos_emb = layer_pos_emb, max_seq_len = max_seq_len, **softmaxdict)
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
        mask = None,
        return_embeddings = False
    ):
        n, device = x.shape[1], x.device

        if self.abs_pos_emb is not None:
            pos_emb = self.abs_pos_emb(torch.arange(n, device = device))
            x = x + einops.rearrange(pos_emb, 'n d -> () n d')

        for attn, ff in self.layers:
            x = attn(x, mask = mask) + x
            x = ff(x) + x

        if not return_embeddings:
            x = self.to_logits(x)

        return x


class XAutoregressiveWrapper(nn.Module):
    def __init__(self, net, ignore_index = -100, pad_value = 0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token = None, temperature = 1., filter_logits_fn = top_k, filter_thres = 0.9, **kwargs):
        device = start_tokens.device
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 1:
            start_tokens = start_tokens[None, :]

        b, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        mask = kwargs.pop('mask', None)

        if mask is None:
            mask = torch.full_like(out, True, dtype=torch.bool, device=out.device)

        for _ in range(seq_len):
            x = out[:, -self.max_seq_len:]
            mask = mask[:, -self.max_seq_len:]

            logits = self.net(x, mask=mask, **kwargs)[:, -1, :]

            if filter_logits_fn in {top_k, top_p}:
                filtered_logits = filter_logits_fn(logits, thres = filter_thres)
                probs = F.softmax(filtered_logits / temperature, dim=-1)

            elif filter_logits_fn is entmax:
                probs = entmax(logits / temperature, alpha = ENTMAX_ALPHA, dim=-1)

            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample), dim=-1)
            mask = F.pad(mask, (0, 1), value=True)

            if exists(eos_token):
                is_eos_token = (out == eos_token)

                if is_eos_token.any(dim = -1).all():
                    # mask out everything after the eos tokens
                    shifted_is_eos_tokens = F.pad(is_eos_tokens, (1, -1))
                    mask = shifted_is_eos_tokens.float().cumsum(dim = -1) >= 1
                    out = out.masked_fill(mask, self.pad_value)
                    break

        out = out[:, t:]

        if num_dims == 1:
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

    def forward(self, x, **kwargs):
        device = kwargs.get("device", torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        xi = x[:, :-1]
        xo = x[:, 1:]
        x = self.net.tokenize(x, device=device)
        xo = xo.to(device)

        # help auto-solve a frequent area of confusion around input masks in auto-regressive
        # if user supplies a mask that is only off by one from the source sequence, resolve it for them
        mask = kwargs.get('mask', None)
        if mask is not None and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs['mask'] = mask

        out = self.net(xi, tokenize=False, **kwargs)
        loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index = self.ignore_index)
        return loss
'''
    def forward(self, x, **kwargs):
        # if not isinstance(x, torch.Tensor) and isinstance(self.net, bioseq.encoders.SeqEncoder):
        device = kwargs.get("device", torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        x = self.net.tokenize(x, device=device)
        print(x)
        xi = x[:, :-1]
        xo = x[:, 1:]

        out = self.net(xi, tokenize=False, **kwargs)
        if xo.dtype is not torch.long:
            xo = xo.to(torch.long)
        loss = F.cross_entropy(out.transpose(1, 2), xo, ignore_index = self.ignore_index)
        return loss

'''


class TokenizerLayer(nn.Module):
    def __init__(self, tokenizer, *, padlen, batch_first=True, nthreads=-1, destchar='i'):
        super().__init__()
        assert padlen >= 0
        self.tokenizer = tokenizer
        self.pad = padlen
        self.batch_first = batch_first
        self.destchar = destchar
        self.nthreads = nthreads if nthreads > 0 else 1
    def forward(self, inputs):
        if isinstance(inputs, torch.Tensor):
            # No need to tokenize, already converted
            return inputs
        return self.tokenizer.batch_tokenize(inputs, padlen=self.pad, batch_first=self.batch_first, nthreads=self.nthreads, destchar=self.destchar)


class SeqEncoder(nn.Module):
    '''
        SeqEncoder - takes a tokenizer, an embedding layer, and an encoding layer
        and propagates arguments to it.
        Takes input strings/byte sequences and performs a transformer encoding.
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
        if hasattr(self.encoder, "max_seq_len"):
            self.max_seq_len = self.encoder.max_seq_len
        else:
            self.max_seq_len = kwargs['max_seq_len']
        # Use kaiming_normal to initialize embeddings
        torch.nn.init.kaiming_normal_(self.embedding.weight)

    def tokenize(self, inputs, *, device):
        assert device is not None
        from torch import from_numpy, long as torchlong, int as torchint
        if not isinstance(inputs, torch.Tensor):
            inputs = from_numpy(self.tokenizer(inputs))
        if device is not None:
            inputs = inputs.to(device)
        return inputs


    def forward(self, inputs, device=None, tokenize=True, **kwargs):
        """
            Tokenizes
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        from x_transformers import Encoder
        if tokenize:
            tokens = self.tokenize(inputs, device=device)
        else:
            tokens = inputs
        if inputs.device is not device:
            tokens = tokens.to(device)
        tokens = tokens.to(torch.long)
        embs = self.embedding(tokens)
        # print(f"embs {embs} {embs.device}")
        embs = self.emb_dropout(embs)
        encoding = self.encoder(embs, **kwargs)
        # print("encoding", encoding.shape)
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


Tokenizer = bioseq.Tokenizer
TransformerLMs = [linformer.LinearAttentionTransformerLM]
Transformers = [FastTransformer, linformer.LinearAttentionTransformer, XTransformer]
AutoTransformers = [linformer.AutoregressiveWrapper, XAutoregressiveWrapper]
Attentions = [FastAttention, LinSelfAttention]
Encoders = [XEncoder, FastTransformer, FastEncoder, HTransformer1D]
Decoders = [XDecoder]



if __name__ == "__main__":
    from timeit import timeit
    # Test things!
    nseqs = 10
    embdim = 32
    attndim = 16
    headdim = 16
    nlayers = 3
    nheads = 2
    emb_dropout = .15
    seqlen = 4096
    tokl = TokenizerLayer(bioseq.DNATokenizer, padlen=seqlen, destchar='i')
    emb = bioseq.make_embedding(bioseq.DNATokenizer, embdim, norm_type=2.0, sparse=True)

    encoder = SeqEncoder(tokl, emb, FastEncoder, num_tokens=tokl.tokenizer.alphabet_size(), dim=embdim, depth=nlayers,
                         max_seq_len=tokl.pad, heads=nheads, dim_head=headdim, ff_mult=4, absolute_pos_emb=False, key_sparse_softmax=True, tied_sparse_softmax=True, query_sparse_softmax=True)
    hencoder = SeqEncoder(tokl, emb, HTransformer1D, num_tokens=tokl.tokenizer.alphabet_size(), causal=False, dim=embdim, depth=nlayers,
                          max_seq_len=tokl.pad, heads=nheads, dim_head=headdim, ff_mult=4, block_size=8)
    xencoder = SeqEncoder(tokl, emb, XEncoder, dim=embdim, depth=nlayers,
                          max_seq_len=tokl.pad, heads=nheads, dim_head=headdim, ff_mult=4, gate_residual=True, gate_values=True, rotary_pos_emb = True)
    xdec = XDecoder(dim=embdim, depth=nlayers, dim_head=headdim, heads=nheads, ff_mult=4, gate_residual=True, gate_values=True, attn_talking_heads=True, rotary_pos_emb = True, cross_residual_attn=True)
    sfmax = SparseSoftmax()
    Xs = torch.randn(4, 10, dtype=torch.float64, requires_grad=True)
    Ys = torch.max(torch.randn_like(Xs), dim=1)[1]
    from random import choice
    seqs = ["".join(choice("ACGT") for i in range(seqlen)) for j in range(nseqs)]

    # FastEncoder
    output = encoder(seqs)
    print(output.shape)
    print("Time to compute FastEncoder: ", timeit(lambda: encoder(seqs), number=2))

    # XEncoder
    output = xencoder(seqs)
    print(output.shape)
    print("Time to compute XEncoder: ", timeit(lambda: xencoder(seqs), number=2))


    # Hierarchical Attention
    houtput = hencoder(seqs, return_embeddings=True)
    print("hout.shape (embeddings)", houtput.shape)
    houtput = hencoder(seqs)
    print("hout.shape", houtput.shape)
    print("Time to compute HEncoder: ", timeit(lambda: hencoder(seqs), number=2))
    decout = xdec(output)
    print("encoder output shape: ", output.shape)
    print("decoder output shape: ", decout.shape)
