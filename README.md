# bioseq

Performs one-hot encoding for DNA or Protein sequences with C++ code, optionally converting to pytorch and moving to device.

Offers DNA, 20-letter amino acid, and a variety of compressed protein and DNA alphabets, and optionally is parallelized for conversion.

There are 3 items exposed:
    • Tokenizer, which can then tokenize
    • onehot_encode, which takes a Tokenizer, some options, and a sequence or batch of sequences
    • f_encode, which takes more options. This results in a temporary Tokenizer, so it might be faster to use the prior.
        • Easy to use, functional
