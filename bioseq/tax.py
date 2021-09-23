import sys

# Simple taxonomic utilities

def skipgt(x):
    return x[x.startswith(">"):]


def get_qstr(path):
    import gzip
    with gzip.open(path, "rt") as gfp:
        return skipgt(next(gfp).split(" ")[0])


def get_taxids(fns, gbac2id):
    import numpy as np
    return np.array(list(map(get_taxid, fns)))

def get_taxid(fn, isid=False):
    if not isid:
        fn = get_qstr(fn)
    from subprocess import check_output
    cmd = f"esearch -db nucleotide -query \"{fn}\"|esummary|xtract -pattern TaxId -element TaxId"
    print(cmd, file=sys.stderr, flush=True)
    try:
        return int(check_output(cmd, shell=True).decode().strip())
    except:
        return -1


__all__ = ["get_taxid", "get_taxids"]
