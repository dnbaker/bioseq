import bioseq


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
        


class FlatFileLoader:
    def __init__(self, ff, tokenizer, *, destfile=None, batch_size=8192):
        assert isinstance(ff, bioseq.FlatFile)
        assert isinstance(tokenizer, bioseq.Tokenizer)
        if destfile is None:
            destfile = ff.path + ".padded"
        self.ff = ff
        self.tokenizer = tokenizer
        self.seq_len = ff.maxseqlen()
        self.mat, self.matpath = FF2NP(ff, tokenizer, destfile=destfile, batch_size=batch_size)
    def __getitem__(self, index):
        import numpy as np
        from torch import from_numpy as frnp
        return frnp(self.mat[index].astype(np.int32))
    def __len__(self):
        return self.mat.shape[0]
    def __enter__(self):
        return self
    def __del__(self):
        import os
        del self.mat
        if os.path.isfile(self.matpath):
            os.remove(self.matpath)
