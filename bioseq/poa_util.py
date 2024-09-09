import networkx as nx
import torch
import pytorch_geometric as pyg

class FastxSeq:
    trans = str.maketrans("U", "T")
    def __init__(self, x, standardize_nuc=False):
        self.seq = x.sequence
        self.name = x.name
        self.comment = x.comment
        self.qual = x.quality
        if standardize_nuc:
            self.standardize()

    def __str__(self):
        comment = "" if not self.comment else " " + self.comment
        if self.qual is not None:
            return f"@{self.name}{comment}\n{self.seq}\n+\n{seq.qual}"
        else:
            return f">{self.name}{comment}\n{self.seq}"

    def standardize(self):
        self.seq = str.translate(self.seq, self.trans)


class ExtractedPOAGraph:
    def __init__(self, mat):
        self.node_feats = list(mat['bases'])

        edge_ip = mat['edge_indptr']
        edge_supporting_seqs = mat['edge_nodes']
        self.edge_seq_support = [edge_supporting_seqs[edge_ip[idx]:edge_ip[idx + 1]] for idx in range(len(edge_ip) - 1)]

        seq_ip = mat['seq_indptr']
        seq_supporting_nodes = mat['seq_nodes']
        self.seq_node_support = [seq_supporting_nodes[seq_ip[idx]:seq_ip[idx + 1]] for idx in range(len(seq_ip) - 1)]

        self.edge_coo = mat['matrix_coo'][:,:2]
        self.mat = mat
        self.ranks = mat['ranks']
        graph = nx.DiGraph()
        node_names = [f"{self.node_feats[x]}-{x}"  for x in range(len(self.node_feats))]
        node_handles = list(map(graph.add_node, node_names))
        for (x, y) in self.edge_coo:
            x = node_names[x]
            y = node_names[y]
            graph.add_edge(y, x)
        self.graph = graph


    def __str__(self):
        return f"feats: {self.node_feats}. Ranks: {self.ranks}. Edges: {self.edge_coo}. Graph:{self.graph}"


class POAEmbedder:
    # TODO: add EOS/BOS support, future enhancement.
    # Requires adding edges/nodes to nodes with no incident or excident edges, respectively.
    def __init__(self, tok, emb_dim=128):
        self.tok = tok
        self.emb = torch.nn.Embedding(tok.alphabet_size(), emb_dim)

    # Takes the output of bioseq.SequenceGraph().matrix() and creates the data for GAT
    def embed_graph(self, mat):
        x, data = self.to_x_data(mat)
        return pyg.Data(x, data)

    # Takes the output of bioseq.SequenceGraph().matrix() and creates the data for GAT
    def to_x_data(self, mat):
        embedded = self.emb(torch.from_numpy(tok.batch_tokenize([mat['bases']], padlen=len(mat['bases'])).astype(np.int32)))
        x = embedded.view(-1, embedded.size(2))
        data = torch.from_numpy(mat['matrix_coo'][:,:2].astype(np.int32)) # COO
        return (x, data)


__all__ = ["ExtractedPOAGraph", "FastxSeq"]
