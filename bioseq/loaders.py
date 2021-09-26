import os
import sys

import bioseq
import numpy as np
import torch


def FF2NP(x, tokenizer, destfile, *, batch_size=8192):
    import numpy as np
    assert isinstance(x, bioseq.FlatFile)
    assert isinstance(tokenizer, bioseq.Tokenizer)
    msl = x.maxseqlen()
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
        

class FlatFileDataset(torch.utils.data.DataLoader):
    def __init__(self, ff, tokenizer, *, destfile=None, batch_size=8192):
        super(FlatFileDataset).__init__()
        assert isinstance(ff, bioseq.FlatFile)
        assert isinstance(tokenizer, bioseq.Tokenizer)
        self.ff = ff
        self.tokenizer = tokenizer
        self.max_seq_len = ff.maxseqlen() + tokenizer.includes_bos() + tokenizer.includes_eos()
    def __getitem__(self, index):
        import numpy as np
        from torch import from_numpy as frnp
        return frnp(self.tokenizer.batch_tokenize([self.ff.access(index)], padlen=self.max_seq_len, batch_first=True, destchar='B')).to(torch.long).squeeze()
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
