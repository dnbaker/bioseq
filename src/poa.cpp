#include <span>
#include <set>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>
#include <zlib.h>
#include "spoa/spoa.hpp"
#include "spoa/graph.hpp"
namespace py = pybind11;

using namespace spoa;
using Edge = spoa::Graph::Edge;
using Node = spoa::Graph::Node;


static constexpr int32_t GLOBAL = 1;

std::unique_ptr<spoa::AlignmentEngine> make_engine() {
    return spoa::AlignmentEngine::Create( static_cast<spoa::AlignmentType>(GLOBAL), 5, -4, -8, -6, -10, -4);
}

struct GraphRepr {
    std::vector<char> bases;
    std::vector<std::string> strings;
    std::vector<std::vector<int32_t>> outEdges;
    std::vector<std::vector<int32_t>> inEdges;
    std::vector<std::vector<int32_t>> alignedNodes;
    std::vector<spoa::Alignment> alignments;
    std::string consensus;
};

struct SequenceGroup {
    py::list sequences;
    std::vector<int32_t> scores;
    std::string consensus;
    std::vector<std::string> inputs;
    std::vector<spoa::Alignment> alignments;
    std::unique_ptr<spoa::Graph> graph;
    SequenceGroup(py::list sequences): sequences{sequences} {}
    void build(int min_coverage=-1, spoa::AlignmentEngine* engine=nullptr) {
        if(min_coverage <= 0) {
            min_coverage = std::max(py::size_t(0), (sequences.size() - 1) / 2);
        }
        std::unique_ptr<spoa::AlignmentEngine> localEngine(engine ? std::unique_ptr<spoa::AlignmentEngine>(): make_engine());
        if(localEngine) engine = localEngine.get();
        graph = std::make_unique<Graph>();
        /*
        auto getQual = [&] () -> std::optional<std::span<uint8_t>> {
            auto it = py::cast<py::iterator>(qualities);
            if(!it.is_none()) {
            }
            return std::nullopt;
        };
        */
        for(auto seq: sequences) {
            int32_t score{0};
            py::ssize_t size;
            py::str str = py::cast<py::str>(seq);
            const char *ptr = PyUnicode_AsUTF8AndSize(str.ptr(), &size);
            scores.push_back(score);
            inputs.emplace_back(ptr);
            const auto alignment = engine->Align(inputs.back(), *graph, &score);
            graph->AddAlignment(alignment, ptr);
            alignments.push_back(alignment);
        }
        consensus = graph->GenerateConsensus(min_coverage);
    }
    py::dict GraphToPython() const {
        using namespace pybind11::literals;
        std::vector<char> bases;
        std::unordered_map<Edge*, int32_t> edgeIdMap;
        std::unordered_map<Node*, int32_t> nodeIdMap;
        std::unordered_map<Node*, int32_t> nodeRankMap;
        std::unordered_map<int32_t, std::set<int32_t>> seqIdToNodes;
        std::unordered_map<int32_t, std::set<int32_t>> seqIdToEdges;
        std::vector<int32_t> edgeLabels;
        std::vector<int64_t> edgeIndptr{{0}};
        std::unordered_map<uint64_t, int32_t> edges; // Map from (from, to): edge_id
        for(const auto& edge: graph->edges()) {
            const int32_t id = edgeIdMap.size();
            edgeIdMap.emplace(edge.get(), id);
        }
        const auto& rankToNode = graph->rank_to_node();
        int32_t nodeId{0};
        //const auto edgeToId = [&edgeIdMap](Edge* const edge) {return edgeIdMap.at(edge);};
        for(const auto& node: rankToNode) {
            bases.push_back(node->code);
            nodeRankMap.emplace(node, nodeId);
        }
        for(const auto& node: graph->nodes()) {
            const int32_t id = nodeIdMap.size();
            nodeIdMap.emplace(node.get(), id);
        }
        const auto nodeToId = [&nodeIdMap](Node* const node) {return nodeIdMap.at(node);};
        auto updateEdges = [&edges](const int32_t from, const int32_t to) {
            const int32_t edgeId = edges.size();
            edges.emplace((uint64_t(from) << 32) | to, edgeId);
        };
        for(const auto& edge: graph->edges()) {
            updateEdges(nodeToId(edge->head), nodeToId(edge->tail));
            edgeIndptr.push_back(edgeIndptr.back() + edge->labels.size());
            edgeLabels.insert(edgeLabels.end(), edge->labels.begin(), edge->labels.end());
        }
        // 1. Get the nodes out in topological order.
        // 2. Get all edges out.
        // 3. Annotate all edges with input support.
        // 4. Annotate all nodes with input support.
        // 5. Generate all supported paths through the graph.
        // 6. Outside of this - bring in other data from the reads.
        for(const auto& edge: graph->edges()) {
            Node* const source = edge->head;
            Node* const sink = edge->tail;
            for(const int32_t id: edge->labels) {
                seqIdToNodes[id].insert(nodeIdMap.at(source));
                seqIdToNodes[id].insert(nodeIdMap.at(sink));
                seqIdToEdges[id].insert(edgeIdMap.at(edge.get()));
            }
        }
        std::vector<int32_t> nodeRanks(nodeIdMap.size());
        for(const auto& [node, rank]: nodeRankMap) {
            nodeRanks[nodeIdMap.at(node)] = rank;
        }
        py::array_t<int32_t> nodeRanksPy({std::ssize(nodeRanks)});
        int32_t* data = reinterpret_cast<int32_t *>(nodeRanksPy.request().ptr);
        std::copy(nodeRanks.begin(), nodeRanks.end(), data);

        std::vector<int32_t> seqAlignments; // Packed sparse matrix
        std::vector<int64_t> seqIndptr{{0}};
        for(const auto& [seqId, nodes]: seqIdToNodes) {
            const int64_t numNodes = nodes.size();
            seqIndptr.push_back(numNodes + seqIndptr.back());
            std::copy(nodes.begin(), nodes.end(), std::back_inserter(seqAlignments));
        }

        py::array_t<int32_t> seqAlignmentsPy({std::ssize(seqAlignments)});
        std::copy(seqAlignments.begin(), seqAlignments.end(), reinterpret_cast<int32_t *>(seqAlignmentsPy.request().ptr));

        py::array_t<int64_t> seqIndptrPy({std::ssize(seqIndptr)});
        std::copy(seqIndptr.begin(), seqIndptr.end(), reinterpret_cast<int64_t *>(seqAlignmentsPy.request().ptr));

        py::array_t<int32_t> edgeLabelsPy({std::ssize(edgeLabels)});
        std::copy(edgeLabels.begin(), edgeLabels.end(), reinterpret_cast<int32_t *>(edgeLabelsPy.request().ptr));

        py::array_t<int64_t> edgeIndptrPy({std::ssize(edgeIndptr)});
        std::copy(edgeIndptr.begin(), edgeIndptr.end(), reinterpret_cast<int64_t *>(edgeLabelsPy.request().ptr));

        py::array_t<int32_t> matrixCOOPy({std::ssize(edges) * 3});
        int32_t* destPtr = reinterpret_cast<int32_t *>(matrixCOOPy.request().ptr);
        for(const auto& edge: edges) {
            *destPtr++ = edge.first >> 32;
            *destPtr++ = (edge.first << 32) >> 32;
            *destPtr++ = edge.second;
        }

        return py::dict("bases"_a=bases, "ranks"_a=nodeRanksPy, "seq_nodes"_a=seqAlignments, "seq_indptr"_a=seqIndptrPy,
                        "edge_nodes"_a=edgeLabelsPy, "edge_indptr"_a=edgeIndptrPy, "matrix_coo"_a=matrixCOOPy, "consensus"_a=consensus, "input_sequences"_a=sequences);
    }
    GraphRepr GenerateGraph() {
        GraphRepr ret;
        std::vector<char> bases;
        std::unordered_map<Edge*, int32_t> edgeIdMap;
        std::unordered_map<Node*, int32_t> nodeIdMap;
        for(const auto& edge: graph->edges()) {
            const int32_t id = edgeIdMap.size();
            edgeIdMap.emplace(edge.get(), id);
        }
        const auto& rankToNode = graph->rank_to_node();
        int32_t nodeId{0};
        const auto edgeToId = [&edgeIdMap](Edge* const edge) {return edgeIdMap.at(edge);};
        for(const auto& node: rankToNode) {
            bases.push_back(node->code);
            nodeIdMap.emplace(node, nodeId);
            std::vector<int32_t> outEdges;
            std::transform(std::cbegin(node->outedges), std::cend(node->outedges), std::back_inserter(outEdges), edgeToId);
            ret.outEdges.emplace_back(outEdges);

            std::vector<int32_t> inEdges;
            std::transform(std::cbegin(node->inedges), std::cend(node->inedges), std::back_inserter(inEdges), edgeToId);
            ret.inEdges.emplace_back(inEdges);

            std::vector<int32_t> alignedNodes;
            std::transform(std::cbegin(node->aligned_nodes), std::cend(node->aligned_nodes), std::back_inserter(alignedNodes), [&nodeIdMap](const auto node) {return nodeIdMap.at(node);});
            ret.alignedNodes.emplace_back(alignedNodes);
            ++nodeId;
        }
        ret.bases = std::move(bases);
        // copy the rest
        ret.consensus = consensus;
        ret.strings = inputs;
        ret.alignments = alignments;
        return ret;
    }
};

void init_poa(py::module &m) {
    py::class_<SequenceGroup>(m, "SequenceGraph")
    .def(py::init<py::list>())
    .def("build", [](SequenceGroup& group, int minCov) {group.build(minCov);})
    .def("matrix", &SequenceGroup::GraphToPython)
    .def_property_readonly("sequence", [] (const SequenceGroup& group) -> std::string {return group.consensus;});
    
#if 0
    py::class_<FlatFileIterator>(m, "FlatFileIterator")
    .def(py::init<FlatFileIterator>())
    .def("__iter__", [](const FlatFileIterator &x) {return x;})
    .def("__next__", [](FlatFileIterator &x) {return x.next();})
    .def_property_readonly("sequence", &FlatFileIterator::sequence)
    .def_property_readonly("seq", &FlatFileIterator::sequence);

    py::class_<FlatFile>(m, "FlatFile")
    .def(py::init<std::string, py::ssize_t>(), py::arg("inputfile"), py::arg("maxseqlen") = -1)
    .def(py::init<std::string, std::string>())
    .def_readonly("path", &FlatFile::path_)
    .def("access", &FlatFile::access)
    .def("access", [](const FlatFile &x, py::slice slc) {return x.range_access(slc);})
    .def("access", [](const FlatFile &x, size_t i, size_t j, size_t step) {return x.range_access(i, j, step);},
        py::arg("start"), py::arg("stop"), py::arg("step") = 1)
    .def("__len__", &FlatFile::nseqs)
    .def("nseqs", &FlatFile::nseqs)
    .def("size", &FlatFile::nseqs)
    .def("seq_offset", &FlatFile::seq_offset)
    .def("indptr", &FlatFile::indptr)
    .def_property_readonly("maxseqlen", &FlatFile::max_seq_len)
    .def_property_readonly("max_seq_len", &FlatFile::max_seq_len)
    .def("__iter__", [](const FlatFile &x) {return FlatFileIterator(x);}, py::keep_alive<0, 1>())
    .def("__getitem__", [](const FlatFile &x, py::ssize_t idx) -> py::bytearray {
        py::ssize_t ai;
        if(idx >= 0) {
            ai = idx;
        } else {
            if(idx < -py::ssize_t(x.nseqs()))
                throw std::out_of_range("For a negative index, idx must be >= -len(x)");
            ai = x.nseqs() + idx;
        }
        return x.access(ai);
    })
    .def("__getitem__", [](const FlatFile &x, py::slice slc) -> py::list {
        return x.range_access(slc);
    })
    .def("__getitem__", [](const FlatFile &x, py::array arr) -> py::list {
        return x.range_access(arr);
    });


#if 0
    m.def("makeflat", [](std::string inpath, std::string outpath) {
        return FlatFile::make(inpath, outpath);
    }, py::arg("input"), py::arg("output") = "", py::return_value_policy::move);
#endif
    m.def("getstats", [](py::sequence items) {
        std::vector<std::string> paths;
        for(const auto item: items)
            paths.emplace_back(item.cast<std::string>());
        py::list alist = py::list();
        while(py::len(alist) < paths.size()) {
            alist.append(py::none());
        }
        for(size_t i = 0; i < paths.size(); ++i) {
            const auto vals = getlens(paths[i]);
            const py::ssize_t sz = vals.size();
            py::array_t<size_t> ret(std::vector<py::ssize_t>{sz});
            py::buffer_info bi = ret.request();
            std::copy(vals.data(), vals.data() + sz, (size_t *)bi.ptr);
            alist[i] = ret;
        }
        return alist;
    });
#endif
}
