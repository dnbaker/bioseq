import os

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
    retmat = np.memmap(destfile, mode='w+', dtype=np.uint8, shape=(nseqs, msl))
    nbatches = (nseqs + batch_size - 1) // batch_size
    for i in range(nbatches):
        start = i * batch_size
        stop = min(start + batch_size, nseqs)
        subset = x.access(start, stop)
        retmat[start:stop] = tokenizer.batch_tokenize(subset, padlen=total_msl, batch_first=True, destchar='B')
    return (retmat, destfile)
        


class FlatFileDataset(torch.utils.data.DataLoader):
    def __init__(self, ff, tokenizer, *, destfile=None, batch_size=8192):
        super(FlatFileDataset).__init__()
        assert isinstance(ff, bioseq.FlatFile)
        assert isinstance(tokenizer, bioseq.Tokenizer)
        from time import time
        tstart = time()
        if destfile is None:
            destfile = ff.path + ".padded"
        self.ff = ff
        self.tokenizer = tokenizer
        self.max_seq_len = ff.maxseqlen()
        if os.path.isfile(destfile) and os.path.getsize(destfile) == ff.nseqs() * self.max_seq_len:
            self.mat = np.memmap(destfile, dtype=np.uint8, shape=(ff.nseqs(), self.max_seq_len))
            self.matpath = destfile
        else:
            self.mat, self.matpath = FF2NP(ff, tokenizer, destfile=destfile, batch_size=batch_size)
            self.max_seq_len = ff.maxseqlen()
        tstop = time()
        print("Took %gs to create flat, padded RAM-access file" % (tstop - tstart))
    def __getitem__(self, index):
        import numpy as np
        from torch import from_numpy as frnp
        from torch.cuda import is_available as has_cuda
        ret = frnp(self.mat[index].astype(np.int32))
        if has_cuda():
            ret = ret.cuda()
        return ret
    def __len__(self):
        return self.mat.shape[0]
    def __enter__(self):
        return self
    def cleanup(self):
        import os
        self.mat = None
        if os.path.isfile(self.matpath):
            os.remove(self.matpath)
