import numpy as np
import sys
from collections import Counter

from numpy.random import default_rng
rng = default_rng(int(10000. / 137)) # Fine-structure constant of the universe


BLOSUM_TEXT = '''A  R  N  D  C  Q  E  G  H  I  L  K  M  F  P  S  T  W  Y  V  B  Z  X  *
A  4 -1 -2 -2  0 -1 -1  0 -2 -1 -1 -1 -1 -2 -1  1  0 -3 -2  0 -2 -1  0 -4
R -1  5  0 -2 -3  1  0 -2  0 -3 -2  2 -1 -3 -2 -1 -1 -3 -2 -3 -1  0 -1 -4
N -2  0  6  1 -3  0  0  0  1 -3 -3  0 -2 -3 -2  1  0 -4 -2 -3  3  0 -1 -4
D -2 -2  1  6 -3  0  2 -1 -1 -3 -4 -1 -3 -3 -1  0 -1 -4 -3 -3  4  1 -1 -4
C  0 -3 -3 -3  9 -3 -4 -3 -3 -1 -1 -3 -1 -2 -3 -1 -1 -2 -2 -1 -3 -3 -2 -4
Q -1  1  0  0 -3  5  2 -2  0 -3 -2  1  0 -3 -1  0 -1 -2 -1 -2  0  3 -1 -4
E -1  0  0  2 -4  2  5 -2  0 -3 -3  1 -2 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
G  0 -2  0 -1 -3 -2 -2  6 -2 -4 -4 -2 -3 -3 -2  0 -2 -2 -3 -3 -1 -2 -1 -4
H -2  0  1 -1 -3  0  0 -2  8 -3 -3 -1 -2 -1 -2 -1 -2 -2  2 -3  0  0 -1 -4
I -1 -3 -3 -3 -1 -3 -3 -4 -3  4  2 -3  1  0 -3 -2 -1 -3 -1  3 -3 -3 -1 -4
L -1 -2 -3 -4 -1 -2 -3 -4 -3  2  4 -2  2  0 -3 -2 -1 -2 -1  1 -4 -3 -1 -4
K -1  2  0 -1 -3  1  1 -2 -1 -3 -2  5 -1 -3 -1  0 -1 -3 -2 -2  0  1 -1 -4
M -1 -1 -2 -3 -1  0 -2 -3 -2  1  2 -1  5  0 -2 -1 -1 -1 -1  1 -3 -1 -1 -4
F -2 -3 -3 -3 -2 -3 -3 -3 -1  0  0 -3  0  6 -4 -2 -2  1  3 -1 -3 -3 -1 -4
P -1 -2 -2 -1 -3 -1 -1 -2 -2 -3 -3 -1 -2 -4  7 -1 -1 -4 -3 -2 -2 -1 -2 -4
S  1 -1  1  0 -1  0  0  0 -1 -2 -2  0 -1 -2 -1  4  1 -3 -2 -2  0  0  0 -4
T  0 -1  0 -1 -1 -1 -1 -2 -2 -1 -1 -1 -1 -2 -1  1  5 -2 -2  0 -1 -1  0 -4
W -3 -3 -4 -4 -2 -2 -3 -2 -2 -3 -2 -3 -1  1 -4 -3 -2 11  2 -3 -4 -3 -2 -4
Y -2 -2 -2 -3 -2 -1 -2 -3  2 -1 -1 -2 -1  3 -3 -2 -2  2  7 -1 -3 -2 -1 -4
V  0 -3 -3 -3 -1 -2 -2 -3 -3  3  1 -2  1 -1 -2 -2  0 -3 -1  4 -3 -2 -1 -4
B -2 -1  3  4 -3  0  1 -1  0 -3 -4  0 -3 -3 -2  0 -1 -4 -3 -3  4  1 -1 -4
Z -1  0  0  1 -3  3  4 -2  0 -3 -3  1 -1 -3 -1  0 -1 -3 -2 -2  1  4 -1 -4
X  0 -1 -1 -1 -2 -1 -1 -1 -1 -1 -1 -1 -1 -1 -2  0  0 -2 -1 -1 -1 -1 -1 -4
* -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4 -4  1'''


true_aas = 'ARNDCQEGHILKMFPSTWYVX'
blosum_data = np.array([list(map(int, x.strip().split()[1:])) for x in BLOSUM_TEXT.split('\n')[1:]])
amine_chrs = "".join(x.split()[0] for x in BLOSUM_TEXT.split('\n')[1:])
true_idx = [i for i, x in enumerate(amine_chrs) if x in true_aas]
blosum_specific = blosum_data[np.ix_(true_idx, true_idx[:-1])]
blosum_odds = np.exp2(blosum_specific)
rowsums = np.sum(blosum_odds, axis=1)
normrows = blosum_odds / rowsums[:,np.newaxis]
ca = np.array(list(true_aas))[:-1]

aa_array = ca

probdict = {k: normrows[idx].copy() for idx, k in enumerate(true_aas)}
default_transitions = probdict['X']
substituters = {}
def substitute(inchar, size=1):
    '''
        Inputs: inchar [str] - input character to be replaced
            KWarg: size=1 - number of samples
        Outputs: samples of length [size]

        Used for generating new sequences for augmentation.
        Transitions are based on BLOSUM62 scores.
    '''
    return rng.choice(ca, p=probdict.get(inchar, default_transitions), size=size, replace=True)


def augment_seq(inseq, chain_len=1):
    '''
    Takes an input sequence, mutates it `chain_len` times, and then returns the final sequence.

    Inputs:
        inseq - Str
            Must be comprised of valid AAs
        chain_len=1 - Int
            Number of mutations to cause
    Returns:
        outseq - Str
        Final sequence after mutations
    '''
    # ba = bytearray(inseq, 'utf-8')
    ls = len(inseq)
    for _ in range(chain_len):
        outchar, inchar = (0, 0)
        while inchar == outchar:
            idx = rng.choice(ls)
            outchar = inseq[idx]
            inchar = substitute(outchar)[0]
        ba = bytearray(inseq, 'utf-8')
        ba[idx] = ord(inchar)
        inseq = ba.decode()
    return inseq

# substituters = {k: lambda size=1: substitute(k, size=size) for k in true_aas}
hc = Counter(aa_array[rng.choice(20, size=10000, p=probdict['H'])])
assert hc.most_common()[0][0] == 'H', str(hc)
kc = Counter(aa_array[rng.choice(20, size=10000, p=probdict['K'])])
assert kc.most_common()[0][0] == 'K', str(hc)
hc = Counter(substitute('H', size=10000))
assert hc.most_common()[0][0] == 'H', str(hc) + ", but through substituters"

__all__ = ["BLOSUM_TEXT", "aa_array", "substitute", "normrows", "probdict", "augment_seq"]
