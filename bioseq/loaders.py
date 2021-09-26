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
        if 1:
            return
        # This cannot happen: old code
        from time import time
        tstart = time()
        if destfile is None:
            eosstr = 'eos' if tokenizer.eos() >= 0 else 'noeos'
            bosstr = 'bos' if tokenizer.bos() >= 0 else 'nobos'
            padstr = 'padchar' if tokenizer.is_padded() else "padmask"
            destfile = f"{tokenizer.key}.{ff.path}.{eosstr}.{bosstr}.{padstr}.padded"
        if os.path.isfile(destfile) and os.path.getsize(destfile) == ff.nseqs() * self.max_seq_len:
            self.mat = np.memmap(destfile, dtype=np.uint8, shape=(ff.nseqs(), self.max_seq_len))
            self.matpath = destfile
        else:
            self.mat, self.matpath = FF2NP(ff, tokenizer, destfile=destfile, batch_size=batch_size)
            self.max_seq_len = ff.maxseqlen()
        tstop = time()
        print("Took %gs to create flat, padded RAM-access file" % (tstop - tstart), file=sys.stderr)
    def __getitem__(self, index):
        import numpy as np
        return frnp(self.tokenizer.batch_tokenize([self.ff.access(index)], padlen=self.max_seq_len, batch_first=True, destchar='B')).to(torch.long).squeeze()
    def __len__(self):
        return self.ff.nseqs()
        #return self.mat.shape[0]
    def __enter__(self):
        return self
    def cleanup(self):
        pass
