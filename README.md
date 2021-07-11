# bioseq

A C++/Python package performing fast one-hot encoding for DNA or Protein sequences with C++ code, optionally converting to pytorch and moving to device.

Offers 4-letter DNA, 20-letter amino acid, and a variety of other compressed protein and DNA alphabets, and optionally is parallelized.

There are 3 items exposed:
• Tokenizer, which can then tokenize

• onehot_encode, which takes a Tokenizer, some options, and a sequence or batch of sequences

• f_encode, which takes more options. This results in a temporary Tokenizer, so it might be faster to use the prior.
