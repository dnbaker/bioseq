import os
import sys
from time import time
from datetime import datetime

import bioseq
from bioseq.decoders import SeqEncoder, HTransformer1D, XDecoder, XAutoregressiveWrapper, FastEncoder, FAutoregressiveWrapper, RecurrentTransformerWrapper, RecurrentAutoregressiveWrapper
from bioseq.hattn import AutoregressiveWrapper as HAutoregressor
from x_transformers import TransformerWrapper, Decoder
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
aa("--nepochs", type=float, default=1)
aa("--batchsize", type=int, default=8)
aa("--embdim", type=int, default=64)
aa("--headdim", type=int, default=64)
aa("--nheads", type=int, default=8)
aa("--depth", "--nlayers", type=int, default=6)
aa("--sparseemb", action='store_true', help="Use sparse embeddings.")
aa("--learning-rate", "-R", type=float, default=2e-4)
aa("--accumfreq", type=int, default=4)
aa("--bidir-loss", type=float, const=1., nargs='?')
aa("--clip-grad-norm", "--clip", type=float, default=.5)
aa("--sparse-softmax", action='store_true', help="Whether to use differentiably sparse top-k")
aa("--nthreads", "-p", type=int, default=1)
aa("--gate-residual", action='store_true')
aa("--window-length", "--window_length", type=int, default=128)
aa("--max-mem-len", "--max-mem-length", type=int, default=-1)
aa("--shift-mem-down", default=0, type=int)
aa("--augment", type=int, default=0, help="Number of mutations to introduce while augmenting data. Default: 0.")
aa("--augment-frac", type=float, default=.5, help="Fraction of sequences to augment. Default: 0.5, but only used if --augment is set.")
args = ap.parse_args()
print("#Parameters: %s" % args, file=sys.stderr)
LEARNING_RATE = args.learning_rate
GRADIENT_ACCUMULATE_EVERY = args.accumfreq
torch.set_num_threads(args.nthreads)
if args.sparseemb:
    raise Exception("Cannot use sparse embeddings rn")

def roundup(x):
    x = x + 1
    for shift in (1, 2, 4, 8, 16, 32):
        x |= x >> shift
    return x + 1

if args.max_mem_len <= 0:
    args.max_mem_len = args.window_length
    print(f"max_mem_len unset; defaulting to window-length {args.window_length}", file=sys.stderr)

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


dstr = str(datetime.now()).replace(" ", "_").replace(":", "-")
ebpos = f"{'eos'if args.eos else 'noeos'}" + f".{'bos'if args.bos else 'nobos'}"
if args.padchar:
    ebpos += ".padded"
sequencefile = args.sequencefile

ffp = args.sequencefile + ".ff"

if os.path.isfile(ffp):
    print("Found existing flatfile", file=sys.stderr)
    ff = bioseq.FlatFile(ffp)
else:
    print("Making flatfile", file=sys.stderr)
    ff = bioseq.FlatFile(args.sequencefile, ffp)
ffl = bioseq.loaders.FlatFileDataset(ff, tokenizer, augment=args.augment, augment_frac=args.augment_frac)

train_loader  = cycle(DataLoader(ffl, batch_size=args.batchsize))

msl = ffl.max_seq_len
nchunks = (args.window_length - 1 + msl) // args.window_length
unique_name = f"{sequencefile}.{dstr}.{args.window_length}.{args.alphabet}.heads{args.nheads}.depth{args.depth}.dim{args.embdim}.maxseqlen{msl}.{ebpos}"
# print("msl: %d. roundedup: %d\n" % (msl, roundup(msl)))
# msl = roundup(msl)

argdict = {}

# First, make transformerwrapper, which tokenizes, trims, pads, etc.
model = TransformerWrapper(num_tokens=tokenizer.alphabet_size(), max_seq_len=args.window_length, max_mem_len=args.max_mem_len, shift_mem_down=args.shift_mem_down,
                           attn_layers = Decoder(dim=args.embdim, depth=args.depth, heads=args.nheads, rotary_pos_emb=True, rel_pos_bias=True, reversible=True, gate_residual=args.gate_residual))
# Then, make recurrenttransformerwrapper which makes this run as if it were a giant transformer model
model = RecurrentTransformerWrapper(model, max_seq_len=msl)
# Finally, apply the autoregressivewrapper
model = RecurrentAutoregressiveWrapper(model)
if torch.cuda.is_available():
    print("Using CUDA")
    model = model.cuda()
else:
    print("Using CPU with %d threads" % torch.get_num_threads())

optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

NUM_BATCHES  = int((args.nepochs * len(ffl) + GRADIENT_ACCUMULATE_EVERY * args.batchsize - 1) / (GRADIENT_ACCUMULATE_EVERY * args.batchsize))
print("Num batches: ", NUM_BATCHES)
print(f"Using seqfile {args.sequencefile}")

tstart = time()
num = 0
for i in tqdm.tqdm(range(NUM_BATCHES), mininterval=10., desc='training'):
    model.train()

    for __ in range(GRADIENT_ACCUMULATE_EVERY):
        gstart = time()
        nextbatch = next(train_loader).to(torch.long)
        loss = model(nextbatch)
        loss.backward()

    print(f'training loss: {loss.item()} after {time() - tstart}s')
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
    optim.step()
    optim.zero_grad()
from time import time
print(f"Average time per item: {(time() - tstart) / (GRADIENT_ACCUMULATE_EVERY * args.batchsize * NUM_BATCHES)}")
model.eval()
costs = np.memmap(f"costs.{unique_name}.{time()}.f32.bin", mode="w+", shape=(len(ffl),), dtype=np.float32)
for i in range(len(ffl)):
    costs[i] = model(ffl[i].to(torch.long).unsqueeze(0))
print(f"Total cost of dataset: {np.sum(costs)}")

torch.save(model, f"hmodel.{unique_name}.pt")

print(f"Total time: {time() - tstart}")
