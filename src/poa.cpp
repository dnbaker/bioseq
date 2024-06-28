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
    bool isBuilt{false};
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
        isBuilt = true;
    }
    py::dict GraphToPython() {
        if(!isBuilt) {
            build();
        }
        using namespace pybind11::literals;
        std::string bases;
        std::unordered_map<Edge*, int32_t> edgeIdMap;
        std::unordered_map<Node*, int32_t> nodeIdMap;
        std::unordered_map<Node*, int32_t> nodeRankMap;
        std::unordered_map<int32_t, std::set<int32_t>> seqIdToNodes;
        std::unordered_map<int32_t, std::set<int32_t>> seqIdToEdges;
        std::vector<int32_t> edgeLabels;
        std::vector<int64_t> edgeIndptr;
        edgeIndptr.push_back(0);

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
            nodeRankMap.emplace(node, nodeId++);
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
        std::vector<int32_t> nodeRanks(nodeRankMap.size());
        for(const auto& [node, rank]: nodeRankMap) {
            nodeRanks[nodeIdMap.at(node)] = rank;
        }
        py::array_t<int32_t> nodeRanksPy({std::ssize(nodeRanks)});
        int32_t* data = reinterpret_cast<int32_t *>(nodeRanksPy.request().ptr);
        std::copy(nodeRanks.begin(), nodeRanks.end(), data);

        std::vector<int32_t> seqAlignments; // Packed sparse matrix

        std::vector<int64_t> seqIndptr;
        seqIndptr.push_back(0);

        for(const auto& [seqId, nodes]: seqIdToNodes) {
            const int64_t numNodes = nodes.size();
            seqIndptr.push_back(numNodes + seqIndptr.back());
            std::copy(nodes.begin(), nodes.end(), std::back_inserter(seqAlignments));
        }

        py::array_t<int32_t> seqAlignmentsPy({std::ssize(seqAlignments)});
        std::copy(seqAlignments.begin(), seqAlignments.end(), reinterpret_cast<int32_t *>(seqAlignmentsPy.request().ptr));

        py::array_t<int64_t> seqIndptrPy({std::ssize(seqIndptr)});
        std::copy(seqIndptr.begin(), seqIndptr.end(), reinterpret_cast<int64_t *>(seqIndptrPy.request().ptr));

        py::array_t<int32_t> edgeLabelsPy({std::ssize(edgeLabels)});
        std::copy(edgeLabels.begin(), edgeLabels.end(), reinterpret_cast<int32_t *>(edgeLabelsPy.request().ptr));

        py::array_t<int64_t> edgeIndptrPy({std::ssize(edgeIndptr)});
        std::copy(edgeIndptr.begin(), edgeIndptr.end(), reinterpret_cast<int64_t *>(edgeIndptrPy.request().ptr));

        py::array_t<int32_t> matrixCOOPy({std::ssize(edges) * 3});
        int32_t* destPtr = reinterpret_cast<int32_t *>(matrixCOOPy.request().ptr);
        for(const auto& edge: edges) {
            *destPtr++ = edge.first >> 32;
            *destPtr++ = (edge.first << 32) >> 32;
            *destPtr++ = edge.second;
        }
        matrixCOOPy = matrixCOOPy.reshape(std::vector<int64_t>{{static_cast<int64_t>(edges.size()), 3}});

        std::transform(std::cbegin(bases), std::cend(bases), std::begin(bases), [&](const char base) -> char {return graph->decoder(base);});

        return py::dict("bases"_a=bases, "ranks"_a=nodeRanksPy, "seq_nodes"_a=seqAlignmentsPy, "seq_indptr"_a=seqIndptrPy,
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
    .def("build", [](SequenceGroup& group, int minCov) {group.build(minCov);}, py::arg("mincov") = -1)
    .def("matrix", &SequenceGroup::GraphToPython)
    .def_property_readonly("sequence", [] (const SequenceGroup& group) -> std::string {return group.consensus;});
}
