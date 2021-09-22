import os
import sys
from time import time


import bioseq

from bioseq.encoders import SeqEncoder, HTransformer1D
from bioseq.hattn import AutoregressiveWrapper
from argparse import ArgumentParser as AP
import torch
from torch.utils.data import DataLoader

import tqdm


ap = AP()
aa = ap.add_argument
aa("--bos", action="store_true", help="Prepend a BOS tag for sequence")
aa("--eos", action="store_true", help="Append an EOS tag to sequence")
aa("--padchar", action="store_true", help="Treat padding characters are unique identifier. (Default: no embeddings)")
aa("--alphabet", default="PROTEIN")
aa("sequencefile", help="Input sequences; Must be in Fasta or Fastq format. All quality scores are ignored.")
aa("--nbatches", type=int, default=1)
aa("--batchsize", type=int, default=8)
aa("--embdim", type=int, default=64)
aa("--headdim", type=int, default=64)
aa("--nheads", type=int, default=8)
aa("--depth", "--nlayers", type=int, default=6)
aa("--sparseemb", action='store_true', help="Use sparse embeddings.")
aa("--learning-rate", "-R", type=float, default=2e-4)
aa("--accumfreq", type=int, default=4)
aa("--bidir-loss", action='store_true')
args = ap.parse_args()
LEARNING_RATE = args.learning_rate
GRADIENT_ACCUMULATE_EVERY = args.accumfreq
torch.set_num_threads(1)
if args.sparseemb:
    raise Exception("Cannot use sparse embeddings rn")

def roundup(x):
    x = x + 1
    for shift in (1, 2, 4, 8, 16, 32):
        x |= x >> shift
    return x + 1

NUM_BATCHES = args.nbatches
BATCH_SIZE = args.batchsize
argtup = (args.bos, args.eos, args.padchar, args.alphabet)
tokd = bioseq.get_tokenizer_dict(args.bos, args.eos, args.padchar)
try:
    tokenizer = tokd[args.alphabet.upper()]
except KeyError:
    print(tokd.keys())
    raise


def cycle(x):
    while 1:
        yield from x



embeddings = bioseq.make_embedding(tokenizer, args.embdim, norm_type=2.0, sparse=args.sparseemb)

ffp = args.sequencefile + ".ff"

if os.path.isfile(ffp):
    print("Found existing flatfile")
    ff = bioseq.FlatFile(ffp)
else:
    print("Making flatfile")
    ff = bioseq.FlatFile(args.sequencefile, ffp)
ffl = bioseq.loaders.FlatFileDataset(ff, tokenizer)

train_loader  = cycle(DataLoader(ffl, batch_size=args.batchsize))

msl = ffl.max_seq_len + args.eos + args.bos
print("msl: %d. roundedup: %d\n" % (msl, roundup(msl)))
msl = roundup(msl)

tokl = bioseq.encoders.TokenizerLayer(tokenizer, padlen=msl)

seq_encoder = SeqEncoder(tokl, embeddings, HTransformer1D, num_tokens=tokenizer.alphabet_size(), causal=True, reversible=True, heads=args.nheads, depth=args.depth,
                   dim=args.embdim, max_seq_len=msl)
encoder = seq_encoder.encoder
model = seq_encoder
if torch.cuda.is_available():
    print("Using CUDA")
    model = model.cuda()
else:
    print("Using CPU with %d threads" % torch.get_num_threads())
model = AutoregressiveWrapper(model)

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

tstart = time()
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        nextbatch = next(train_loader)
        loss = model(nextbatch)
        loss.backward()
        if args.bidir_loss:
            bid_loss = model(torch.flip(nextbatch, (1,)))
            loss += bid_loss

    print(f'training loss: {loss.item()} after {time() - tstart}s')
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optim.step()
    optim.zero_grad()

from datetime import datetime
dstr = str(datetime.now()).replace(" ", "_").replace(":", "-")
torch.save(model, f"hmodel.{dstr}.pt")

print(f"Total time: {time() - tstart}")
