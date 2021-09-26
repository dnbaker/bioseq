import os
import sys
from time import time


import bioseq

from bioseq.encoders import SeqEncoder, HTransformer1D, XEncoder, XAutoregressiveWrapper, FastEncoder, FAutoregressiveWrapper
from bioseq.hattn import AutoregressiveWrapper as HAutoregressor
from argparse import ArgumentParser as AP
import numpy as np
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
aa("--nepochs", type=int, default=1)
aa("--batchsize", type=int, default=8)
aa("--embdim", type=int, default=64)
aa("--headdim", type=int, default=64)
aa("--nheads", type=int, default=8)
aa("--depth", "--nlayers", type=int, default=6)
aa("--sparseemb", action='store_true', help="Use sparse embeddings.")
aa("--learning-rate", "-R", type=float, default=2e-4)
aa("--accumfreq", type=int, default=4)
aa("--bidir-loss", type=float, const=1., nargs='?')
aa("--clip-grad-norm", "--clip", type=float, default=.25)
aa("--transformer-type", "-T", choices=("Fast", "Hier", "X"), help="Type of transformer to use. Default: HTransformer1D (Hier)", default="X")
aa("--sparse-softmax", action='store_true', help="Whether to use differentiably sparse top-k")
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
    print("Found existing flatfile", file=sys.stderr)
    ff = bioseq.FlatFile(ffp)
else:
    print("Making flatfile", file=sys.stderr)
    ff = bioseq.FlatFile(args.sequencefile, ffp)
ffl = bioseq.loaders.FlatFileDataset(ff, tokenizer)

train_loader  = cycle(DataLoader(ffl, batch_size=args.batchsize))

msl = ffl.max_seq_len
if args.transformer_type == "Hier":
    nmsl = roundup(msl)
    print(f"Padding msl to next power of to: {msl}->{nmsl}",file=sys.stderr)
    msl = nmsl
    del nmsl
# print("msl: %d. roundedup: %d\n" % (msl, roundup(msl)))
# msl = roundup(msl)

tokl = bioseq.encoders.TokenizerLayer(tokenizer, padlen=msl)

argdict = {}

baseargs = {"num_tokens": tokenizer.alphabet_size(), "heads": args.nheads, "depth": args.depth, "dim": args.embdim, "max_seq_len": msl}
if args.transformer_type == "Fast":
    TxType = FastEncoder
    baseargs.update({"query_sparse_softmax": args.sparse_softmax, "key_sparse_softmax": args.sparse_softmax})
elif args.transformer_type == "Hier":
    TxType = HTransformer1D
    baseargs.update({"causal": True, "reversible": True})
else:
    assert args.transformer_type == "X"
    TxType = XEncoder
    baseargs.update({"gate_residual": True, 'rotary_pos_emb': True})
seq_encoder = SeqEncoder(tokl, embeddings, TxType, **baseargs)
encoder = seq_encoder.encoder
model = seq_encoder
if torch.cuda.is_available():
    print("Using CUDA")
    model = model.cuda()
else:
    print("Using CPU with %d threads" % torch.get_num_threads())
if args.transformer_type == "Hier":
    model = HAutoregressor(model)
elif args.transformer_type == "Fast":
    model = XAutoregressiveWrapper(model)
else:
    model = XAutoregressiveWrapper(model)
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

NUM_BATCHES  = (args.nepochs * len(ffl) + GRADIENT_ACCUMULATE_EVERY * args.batchsize - 1) // (GRADIENT_ACCUMULATE_EVERY * args.batchsize)

tstart = time()
num = 0
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        gstart = time()
        nextbatch = next(train_loader).to(torch.long)
        loss = model(nextbatch)
        if args.bidir_loss:
            loss += args.bidir_loss * model(torch.flip(nextbatch, (1,)))
        loss.backward()

    print(f'training loss: {loss.item()} after {time() - tstart}s')
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
    optim.step()
    optim.zero_grad()
print(f"Average time per item: {(time() - tstart) / (GRADIENT_ACCUMULATE_EVERY * args.batchsize * NUM_BATCHES)}")
model.eval()
costs = np.memmap("costs.f32.bin", mode="w+", shape=(len(ffl),), dtype=np.float32)
for i in range(len(ffl)):
    costs[i] = model(ffl[i].to(torch.long).unsqueeze(0))

from datetime import datetime
dstr = str(datetime.now()).replace(" ", "_").replace(":", "-")
torch.save(model, f"hmodel.{dstr}.pt")

print(f"Total time: {time() - tstart}")
