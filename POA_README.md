### Embedding POA graph

So the current method that extracts the poa graph and embeds it for pytorch geometric is in branch poa-embed.

You give the input sequences as a list of strings:

```
graph = bioseq.SequenceGraph(seqs)
graph.build()
mat = graph.matrix()
# ext = ExtractedPOAGraph(mat) - optional but easier to think around.
```


Then to make an input for pytorch geometric/gat, you use bioseq.POAEmbedder.

```
tok = bioseq.DNATokenizer
embedder = bioseq.POAEmbedder(tok, embed_dim=64)
```

```
x, data = embedder.to_x_data(mat)
# Or for pytorch_geometric
x_data = embedder.embed_graph(mat)

gat = # ... (create gat)
gat_output = gat(x, data)
```


### Project direction

1. Choose input sequences.
RNAFam is good, we can use sets of sequences from RNA-MSM or rinalmo, optionally other databases.
RNAFam is small.
2. Gather downstream tasks.
(2) - RNA-MSM https://drive.google.com/drive/folders/1jYqk7rAp9ysJCBXOa5Yx4Z9es89h-f2h, RINALMO have tasks.

To evaluate after pretraininig, we need to map the graph representation to outputs.


2a. How to predict
The simplest thing I can think of for downstream tasks is to take the graph embeddings for each position and predict the task (e.g., structure) directly.

The ExtractedPOAGraph `seq_node_support` field tells us which nodes each sequence aligned to. We can just use a small network from that embedding to the task.

I think we can start with this. If the graph approach works well, I think this should do decently.

2b. Instead, we can concatenate the embeddings at those positions and use an LSTM layer to predict the secondary task from the graph-informed embeddings.

This can hopefully improve our results if 2a goes well.
