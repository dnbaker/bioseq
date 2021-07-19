import cbioseq
from cbioseq import *
from .tax import get_taxid
import bioseq.tax as tax


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



__all__ = ["onehot_encode", "cbioseq", "f_encode", "Tokenizer", "tax"]
