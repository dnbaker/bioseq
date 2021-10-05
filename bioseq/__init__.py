import cbioseq
from cbioseq import *
from .tax import get_taxid
import bioseq.tax as tax
import bioseq.decoders as decoders
import bioseq.hattn as hattn
import bioseq.softmax as softmax
import bioseq.loaders as loaders
import bioseq.annotations as annotations
import bioseq.blosum as blosum

"""
bioseq provides tokenizers and utilities for generating embeddings
See the list of existing tokenizers:
    DNATokenizer
    AmineTokenizer
    Reduced6Tokenizer
    Reduced8Tokenizer
    Reduced10Tokenizer
    Reduced14Tokenizer
    DayhoffTokenizer
    LIATokenizer
    LIBTokenizer
These are pre-made, with no bos, eos, or padding characters.
Use the keys from this set:
    ("SEB6", "SEB8", "SEB10", "SEV10", "MURPHY", "LIA10", "LIB10", "SEB6", "DAYHOFF", "DNA4", "DNA", "DNA5", "KETO", "PURPYR", "BYTES")
to index specific tokenizers into the premade tokenizer dictionaries.

There are 'bos_tokenizers', where BOS is an additional character, but there is no padding or EOS char.
There are 'eos_tokenizers', where EOS is an additional character, but there is no padding or BOS char.
There are 'beos_tokenizers', where EOS and BOS are an additional character, but there is no padding.
There are 'pbeos_tokenizers', where EOS, BOS, and padchar are all characters. This means there are 3 additional alphabet characters.
"""


def onehot_encode(tokenizer, seqbatch, padlen=-1, destchar='B', batch_first=False, to_pytorch=False, device=None):
    """
        Args:
            tokenizer:
                cbioseq.Tokenizer
                This is the type doing the encoding
            seqbatch: Union[Iterable[Union[Str,Bytes,numpy.ndarray]],Str,Bytes,numpy.ndarray]
                Can be a set of sequences or a single sequence
                if a set of sequences, padlen required, and all sequences must be short enough to fit
                in that length, including EOS, BOS, and padding characters, if relevant
            padlen: Size to which to pad the sequence. Optional
            destchar: One of BbIiUuLlQq - specifies the data type of the encoded sequence
            to_pytorch:
                False, by default. Set to true to convert from numpy to pytorch.
            device:
                None by default. Set to a pytorch device or a string representing it to
                cause this function to copy to device after encoding.
    """
    if isinstance(seqbatch, str) or isinstance(seqbatch, bytes):
        res = tokenizer.onehot_encode(seqbatch, padlen, destchar)
    else:
        res = tokenizer.batch_onehot_encode(seqbatch, padlen, destchar)
        if batch_first:
            from einops import rearrange
            res = rearrange(res, 'seq batch base -> batch seq base')
    if to_pytorch:
        from torch import from_numpy
        res = from_numpy(res)
        if device is not None and res.device:
            res = res.to(device)
    return res


def f_encode(seqbatch, key="DNA", bos=False, eos=False, padchar=False, padlen=-1, destchar='B', batch_first=False, to_pytorch=False, device=None):
    """
        Functional encoding of sequence batch.
        Creates a tokenizer and then uses it
        Args:
            seqbatch: Union[Iterable[Union[Str,Bytes,numpy.ndarray]],Str,Bytes,numpy.ndarray]
        Kwargs:
            key:
                Alphabet with which to encode. Choose from
               "DNA" is the default, with 4 character encodings.


            Other options:
                DNA:
                    DNA - 4 characters - ACGT
                    DNA4 - alias for DNA
                    DNA5 - DNA plus character for N, separate from unexpected wildcards
                Reduced DNA:
                    KETO - split into keto/amine -- AC/GT
                    PURPYR - split into purines/pyrimidines (AG/CT)

            For amino acid sequences:
                Full alphabet:
                AMINO20,AMINO,PROTEIN
                     -- 20 amino acids
                Reduced alphabets:
                SEB6, SEB8, SEB10, SEB14 --
                    All reduced protein alphabets, used for long-range homology detection
                LIA10, LIB10
                DAYHOFF
                KETO
                PURPYR

            ,SEB8,SEB10,SEB14,SEV10,MURPHY,LIA10,LIB10,SEB6,DAYHOFF,KETO,PURPYR,DNA4,DNA,DNA5

            bos: To include bos as its own symbol [False]
            eos: To include eos as its own symbol [False]
            padchar: To include padchar as its own symbol [False]
            padlen: Size to which to pad the sequence. Optional
            destchar: One of BbIiUuLlQq - specifies the data type of the encoded sequence
            device:
                None by default. Set to a pytorch device or a string representing it to
                cause this function to copy to device after encoding.
    """
    from cbioseq import Tokenizer
    tokenizer = Tokenizer(key, bos=bos, eos=eos, padchar=padchar)
    return onehot_encode(tokenizer, seqbatch, padlen=padlen, destchar=destchar,
                         batch_first=batch_first, to_pytorch=to_pytorch, device=device)


keys = ("SEB6", "SEB8", "SEB10", "SEV10", "MURPHY", "LIA10", "LIB10", "SEB6", "DAYHOFF", "DNA4", "DNA", "DNA5", "KETO", "PURPYR", "BYTES", "AMINO20", "PROTEIN")
bkeys = keys + tuple(map(str.lower, keys))


DNATokenizer = cbioseq.Tokenizer("DNA")
AmineTokenizer = cbioseq.Tokenizer("AMINO20")
Reduced6Tokenizer = cbioseq.Tokenizer("SEB6")
Reduced8Tokenizer = cbioseq.Tokenizer("SEB8")
Reduced10Tokenizer = cbioseq.Tokenizer("SEB10")
Reduced14Tokenizer = cbioseq.Tokenizer("SEB14")
DayhoffTokenizer = cbioseq.Tokenizer("DAYHOFF")
LIATokenizer = cbioseq.Tokenizer("LIA10")
LIBTokenizer = cbioseq.Tokenizer("LIB10")
default_tokenizers = {"DNA": DNATokenizer,
                      "AMINO20": AmineTokenizer,
                      "AMINE": AmineTokenizer,
                      "PROTEIN": AmineTokenizer,
                      "SEB6": Reduced6Tokenizer,
                      "SEB8": Reduced8Tokenizer,
                      "SEB10": Reduced10Tokenizer,
                      "SEB14": Reduced14Tokenizer,
                      "LIA10": LIATokenizer,
                      "LIA": LIATokenizer,
                      "LIB10": LIBTokenizer,
                      "LIB": LIBTokenizer}
pbeos_tokenizers = {k: cbioseq.Tokenizer(k, bos=True, eos=True, padchar=True) for k in bkeys}
beos_tokenizers = {k: cbioseq.Tokenizer(k, bos=True, eos=True, padchar=False) for k in bkeys}
pbos_tokenizers = {k: cbioseq.Tokenizer(k, bos=True, eos=False, padchar=True) for k in bkeys}
bos_tokenizers = {k: cbioseq.Tokenizer(k, bos=True, eos=False, padchar=False) for k in bkeys}
peos_tokenizers = {k: cbioseq.Tokenizer(k, bos=False, eos=True, padchar=True) for k in bkeys}
eos_tokenizers = {k: cbioseq.Tokenizer(k, bos=False, eos=True, padchar=False) for k in bkeys}
pos_tokenizers = {k: cbioseq.Tokenizer(k, bos=False, eos=False, padchar=True) for k in bkeys}
total_tokenizer_dict = {}
for bos in [0, 1]:
    for eos in [0, 1]:
        for padchar in [0, 1]:
            for k in bkeys:
                total_tokenizer_dict[(bos, eos, padchar, k)] = cbioseq.Tokenizer(k.upper(), bos=bos, eos=eos, padchar=padchar)


def get_tokenizer_dict(bos, eos, padchar):
    if bos:
        if eos:
            return pbeos_tokenizers if padchar else beos_tokenizers
        else:
            return pbos_tokenizers if padchar else bos_tokenizers
    elif eos:
        return peos_tokenizers if padchar else eos_tokenizers
    else:
        return pos_tokenizers if padchar else default_tokenizers


def make_embedding(tok, embdim, maxnorm=None, norm_type=2.0, scale_grad_by_freq=False, sparse=False, _weight=None):
    """
        Args:
            tok: bioseq.Tokenizer
            embdim: Int, dimension for embeddings
        KWArgs:
            maxnorm = None: maximum norm for embeddings. If not None, embeddings will be scaled by the p-norm corresponding to norm_type
            norm_type = 2.: Sets p for the Lp norm. Must be See torch.nn.Embedding.
            scale_grad_by_freq = False: Whether to scale gradient by the count frequencies. See torch.nn.Embedding.
            sparse = False: Whether to use sparse embeddings. False by default.
            _weight = None: If providing tensors from pre-trained, set them here. Must match the tokenizer's number of tokens and the embedding dimension.
    """
    assert norm_type >= 1., f"{norm_type} is not >= 1., so it is not a norm."
    import torch.nn as nn
    return nn.Embedding(tok.alphabet_size(),
                        embdim,
                        padding_idx=tok.pad() if tok.is_padded() else None,
                        scale_grad_by_freq=scale_grad_by_freq, sparse=sparse, _weight=_weight)


def torchify(arr):
    '''Simply wrapper for torch.from_numpy, converts numpy array to pytorch.
    '''
    from torch import from_numpy
    return from_numpy(arr)


class PyViewFF:
    '''
    PyViewFF provides a pure-python view into the C++ FlatFile database.
    '''
    def __init__(self, path):
        fp = np.memmap(path, mode='r', dtype=np.uint8)
        self.nseqs = int(fp[:8].view(np.uint64)[0])
        self.offsets = fp[8:8 * (2 + self.nseqs)].view(np.uint64)
        self.seqs = fp[8 * (2 + self.nseqs):]
        self.fp = fp
    def access(self, idx):
        res = self.seqs[self.offsets[idx]:self.offsets[idx + 1]]
        return bytes(res)
    def __getitem__(self, idx):
        if isinstance(idx, int): return self.access(idx)
        elif isinstance(idx, slice):
            return [self.access(x) for x in range(idx.start, idx.stop, idx.step)]
        else:
            raise InvalidArgument("PyViewFF can only support slices and integers.")
    def __len__(self):
        return self.nseqs


__all__ = ["onehot_encode", "cbioseq", "f_encode", "Tokenizer", "tax",
           "make_embedding",
           "bos_tokenizers", "eos_tokenizers", "beos_tokenizers", "pbeos_tokenizers", "peos_tokenizers", "pbos_tokenizers", "pos_tokenizers", "default_tokenizers", "get_tokenizer_dict",
           "DNATokenizer", "AmineTokenizer", "Reduced6Tokenizer", "Reduced8Tokenizer", "Reduced10Tokenizer", "Reduced14Tokenizer", "DayhoffTokenizer", "LIATokenizer", "LIBTokenizer",
           "decoders", 'softmax', 'hattn', 'loaders', 'torchify']
