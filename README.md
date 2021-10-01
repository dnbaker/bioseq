# bioseq

A C++/Python package performing fast one-hot encoding for DNA or Protein sequences with C++ code, optionally converting to pytorch and moving to device.

Offers 4-letter DNA, 20-letter amino acid, and a variety of other compressed protein and DNA alphabets, and optionally is parallelized.

## Tokenizing

bioseq.Tokenizer does the tokenizing, and there are pre-made tokenizers for all alphabets, as well as combinations of EOS, BOS, and whether padding gets a unique character, or is simply masked.

`bos_tokenizers` is a dictionary from alphabets to Tokenizers with a BOS tag prepended.
`eos_tokenizers` is a dictionary from alphabets to Tokenizers with an EOS tag appended.
`pos_tokenizers` is adictionary from alphabets to Tokenizers with a padding character used.
`beos_tokenizers` adds both BOS and EOS
`pbeos_tokenizers` adds BOS, EOS, and padding characters.

Tokenizer can tokenize (`batch_tokenize`), which creates array of tokens, (uint8 by default),
or it can one-hot encode (`batch_onehot_encode`), which takes the tokens one-step further into one-hot encoding.
Both of these `Tokenizer::batch_*` functions can be parallelized by providing `nthreads={int}`.

tokenizing uses seq-first ordering by default as well, but this can be changed with `batch_first=True`.
one-hot encoding uses seq-first ordering (not batch-first). It does not support `batch_first`.

Both of these are ~30x as fast as using bytes.translate + np.frombuffer + np.vstack + `torch.from_numpy`,
and ~500x as fast as transformers.tokenizer.batch\_encode\_plus.

1. To train Transformers, you need to use `batch_first=True`, followed by torch.nn.Embedding.
2. To train CNNs, tokenize with `batch_first=True`, embed with torch.nn.Embedding, and then apply `lambda x: einops.rearrange(x, 'batch seq emb -> batch emb seq')`.
   This is because CNNs expect (Batch, C, L)
3. To train LSTMs, use `batch_first=False` to tokenize, and embed with torch.nn.Embedding

Basically, you only want `batch_first=False` for LSTM training, and using CNNs will require a rearrange call due to the varying expectation of dimension ordering.

## DataLoading
We use a bioseq.FlatFile method, which provides random access to the sequences in a FAST{Q,A} file.
This is then used by bioseq.FlatFileDataset for use with torch.utils.data.DataLoader.

For an example, see training/trainh.py and training/compute.py.

## Sequence augmentation

We also support augmentation by random mutations sampled according to BLOSUM62 transition probabilities.
This is only valid for tokenizers using the full 20-character amino acid alphabet ("PROTEIN" or "AMINO20"). We may modify this in the future to support other alphabets.

bioseq.AmineTokenizer is a pre-build tokenizer without BOS, EOS, or padding which is valid for this.


## Dependencies

pybind11 v2.7 is required, in order to support bytearray
numpy is required
pytorch (as torch) is also required

Besides these, there are some python-only dependencies which setup.py should download for you.

All of these can be manually installed via `python3 -m pip install -r requirements.txt`.
