import os
import sys

import bioseq
import numpy as np
import torch
from bioseq.blosum import augment_seq


def FF2NP(x, tokenizer, destfile, *, batch_size=8192):
    import numpy as np
    assert isinstance(x, bioseq.FlatFile)
    assert isinstance(tokenizer, bioseq.Tokenizer)
    msl = x.maxseqlen
    total_msl = msl + tokenizer.includes_bos() + tokenizer.includes_eos()
    nseqs = x.nseqs()
    retmat = np.memmap(destfile, mode='w+', dtype=np.uint8, shape=(nseqs, total_msl))
    nbatches = (nseqs + batch_size - 1) // batch_size
    for i in range(nbatches):
        start = i * batch_size
        stop = min(start + batch_size, nseqs)
        seqs = x.access(i * batch_size, stop)
        retmat[start:stop] = tokenizer.batch_tokenize(
            seqs, padlen=msl, batch_first=True, destchar='B')
    return (retmat, destfile)
        

class FlatFileDataset(torch.utils.data.Dataset):
    """
        Creates a FlatFileDataset from a Tokenizer and a FlatFile.
        if keyword augment is provided, then sequences will be mutated <augment> times
        before tokenizing using BLOSUM62 substitution rates.
    """
    def __init__(self, ff, tokenizer, *, augment=0, augment_frac=0.5):
        super(FlatFileDataset).__init__()
        assert isinstance(ff, bioseq.FlatFile)
        assert isinstance(tokenizer, bioseq.Tokenizer)
        self.ff = ff
        self.tokenizer = tokenizer
        self.max_seq_len = ff.maxseqlen + tokenizer.includes_bos() + tokenizer.includes_eos()
        self.maxseqlen = self.max_seq_len
        self.augment = augment
        self.augment_frac = augment_frac
        from numpy.random import default_rng
        self.rng = default_rng(13)
    def __getitem__(self, index):
        import numpy as np
        from torch import from_numpy as frnp
        seq = self.ff.access(index)
        if self.augment and (self.augment_frac >= 1. or self.rng.uniform() < self.augment_frac):
            seq = augment_seq(seq.decode(), self.augment)
        return frnp(self.tokenizer.batch_tokenize([seq], padlen=self.max_seq_len, batch_first=True, destchar='B')).to(torch.long).squeeze()
    def access(self, slc, stop=None, step=None):
        if isinstance(slc, int):
            slc = slice(slc, stop, step)
        from torch import from_numpy as frnp
        seqs = self.ff.access(slc.start, slc.stop, slc.step)
        toks = self.tokenizer.batch_tokenize(seqs, padlen=self.max_seq_len, batch_first=True, destchar='B')
        toks = frnp(toks).to(torch.long)
    def __len__(self):
        return self.ff.nseqs()
    def cleanup(self):
        pass

class AugmentedSeqDataset(FlatFileDataset):
    def __init__(self, ff, tokenizer, augment=1, augment_frac=.5):
        super().__init__(ff, tokenizer, augment=augment, augment_frac=augment_frac)
