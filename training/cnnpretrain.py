import os
import sys
from argparse import ArgumentParser as AP
from timeit import default_timer as time
from functools import reduce

import torch
import random
import numpy as np
import einops
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import tqdm


import bioseq
import bioseq.cnnencoder as cnn
from bioseq.cnnencoder import RevConvInfiller
from bioseq.blosum import augment_seq


ap = AP()
aa = ap.add_argument
aa("inputfile", help="Path to input fasta")

aa("--alphabet", default="PROTEIN")
aa("--bos", action="store_true", help="Prepend a BOS tag for sequence")
aa("--eos", action="store_true", help="Append an EOS tag to sequence")
aa("--padchar", action="store_true", help="Treat padding characters are unique identifier. (Default: no embeddings)")
aa("--batchsize", "--batch-size", type=int, default=64)
aa("--emb-dim", "--embedding", type=int, default=64)
aa("--revdepth", default=3, type=int, help="Depth of reversible CNN block (number of squeeze-excite layers)")
aa("--totaldepth", default=3, type=int, help="Number of RevNet + Squeeze/Excite Block Pairs. Total number of reversible layers is revdepth * totaldepth.")
aa("--noactivation", action='store_true', help="Whether or not to perform activation at the start of bottleneck layer")
aa("--nthreads", type=int, default=1, help="Number of threads. Set to < 0 to use all threads")
aa("--learning-rate", "-R", type=float, default=2e-4)
aa("--accumfreq", type=int, default=4)
aa("--augment", type=int, default=0, help="Number of mutations to introduce while augmenting data. Default: 0.")
aa("--augment-frac", type=float, default=.5, help="Fraction of sequences to augment. Default: 0.5, but only used if --augment is set.")
aa("--kernel-size", type=int, default=9)
aa("--nepochs", type=float, default=1)
aa("--maskfrac", type=float, default=0.15)
aa("--seed", type=int, default=0)

usecuda = torch.cuda.is_available()
device = torch.device("cuda:0") if usecuda else torch.device("cpu")

ap = ap.parse_args()
args = ap
nt = ap.nthreads
if nt < 0:
    from multiprocessing import cpu_count as CC
    nt = CC()
torch.set_num_threads(nt)
LEARNING_RATE = ap.learning_rate
torch.manual_seed(ap.seed)
np.random.seed(ap.seed)
random.seed(ap.seed)


ff = None
ffp = ap.inputfile + ".ff"
if os.path.isfile(ffp):
    ff = bioseq.FlatFile(ffp)
else:
    ff = bioseq.FlatFile(ap.inputfile, ffp)

argtup = (ap.bos, ap.eos, ap.padchar, ap.alphabet)
tokd = bioseq.get_tokenizer_dict(ap.bos, ap.eos, ap.padchar)
try:
    tokenizer = tokd[ap.alphabet.upper()]
except KeyError:
    print(tokd.keys())
    raise

# print("tokenizer eos: ", tokenizer.eos(), "bos", tokenizer.bos(), "padchar", tokenizer.pad(), "is padded", tokenizer.is_padded())

pl = padlen = ff.maxseqlen + tokenizer.includes_eos() + tokenizer.includes_bos()
inchannels = tokenizer.alphabet_size()
# print("Alphabet size: ", inchannels)
model = cnn.RevConvNetwork1D(inchannels, channels=ap.emb_dim, kernel_size=ap.kernel_size, revdepth=ap.revdepth, totaldepth=ap.totaldepth, noactivation=ap.noactivation)
model = RevConvInfiller(model, tokenizer, ap.emb_dim).to(device)
if usecuda:
    model = nn.DataParallel(model)
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
ffl = bioseq.loaders.FlatFileDataset(ff, tokenizer, augment=ap.augment, augment_frac=ap.augment_frac, cnn=True, device=device)
NUM_BATCHES  = int((ap.nepochs * len(ffl) + ap.accumfreq * ap.batchsize - 1) / (ap.accumfreq * ap.batchsize))

def load_next(batch):
    oh = tokenizer.batch_onehot_encode(batch, padlen=pl)
    laidout = einops.rearrange(oh, "length batch emb -> batch emb length")
    return torch.from_numpy(laidout).device(device).float()

def cycle(x):
    while 1:
        yield from x

random_name = "".join(random.choice("abcdefghijklmn") for x in range(10)) + hex(reduce(lambda x, y: x ^ hash(y), sys.argv, 0))

# train_loader = cycle(DataLoader(ffl, batch_size=ap.batchsize))

assert 1. > ap.maskfrac > 0.

tstart = time()
num = 0
global bstart
bstart = 0
PL = ff.maxseqlen + tokenizer.includes_eos() + tokenizer.includes_bos()
def getbatch():
    global bstart
    seqs = ff[bstart:bstart + ap.batchsize]
    LS = len(seqs)
    augmented_seq_indexes = np.where(np.random.rand(LS) < ap.augment_frac)[0] if ap.augment else []
    for idx in augmented_seq_indexes:
        seqs[idx] = augment_seq(seqs[idx].decode(), ap.augment)

    mask = torch.rand(LS, ff.maxseqlen, device=device) > ap.maskfrac
    for row, seq in enumerate(seqs):
        mask[row,len(seq):] = 1
    # mask = [np.hstack([np.random.rand(len(seq)) > ap.maskfrac, np.zeros(ff.maxseqlen - len(seq), dtype=np.bool_)]) for seq in seqs]
    ohdata = tokenizer.batch_onehot_encode(seqs, padlen=PL)
    maskeddata = tokenizer.batch_onehot_encode(seqs, padlen=PL, mask=mask)
    ohdata, maskeddata = map(lambda x: einops.rearrange(torch.from_numpy(x), "length batch emb -> batch emb length").to(device).float().contiguous(), (ohdata, maskeddata))
    bstart += ap.batchsize
    if bstart > len(ffl):
        bstart = 0
    return ohdata, maskeddata, seqs, mask

losses = []
finished_seqs = 0
saved_loss_id = 0

startpos = int(tokenizer.includes_bos())
tstop = PL - int(tokenizer.includes_eos())

for bn in range(NUM_BATCHES):
    for __ in range(ap.accumfreq):
        gstart = time()
        oh, moh, seqs, masks = getbatch()
        stackmask = torch.logical_not(masks)
        assert moh.device == device
        emb, bo = model(moh)
        tokens = torch.from_numpy(tokenizer.batch_tokenize(seqs, padlen=PL)).to(device).long()
        # print("tokens shape", tokens.shape, stackmask.shape)
        '''
        if 0:
            where = torch.where(stackmask)
            seltoks = tokens[startpos:tstop,:][stackmask.T].to(device).long()
            bot = bo[:,startpos:tstop,:][stackmask]
            loss = F.cross_entropy(bot, seltoks)
        else:
        '''
        loss = F.cross_entropy(bo.transpose(1, 2), tokens.T)
        losses.append(float(loss.item()))
        # sys.exit(1)
        # Now, loss function for masked items
        # This will simply be the MASS objective.
        # Coming back around, it would benefit from turning it to the autoregressive loss
        loss.backward()
        finished_seqs += len(seqs)

    if not (bn & 127):
        print(f'[Batch {bn}] training loss: {loss.item()} after {time() - tstart}s after {finished_seqs} sequences; mean of last 10 {np.mean(losses[-10:])}', flush=True)
        if finished_seqs >= len(seqs):
            torch.save(model, f"model.{random_name}.{saved_loss_id}.pt")
            saved_loss_id += 1
    optim.step()
    optim.zero_grad()

tend = time()
np.array(losses).astype(np.float32).tofile(f"model.{random_name}.final.losses.f32")
torch.save(model, f"model.{random_name}.final.pt")
print("Training took %gs" % (tend - tstart))
