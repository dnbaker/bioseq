#include <span>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>
#include <zlib.h>
#include "spoa/spoa.hpp"
namespace py = pybind11;

using namespace spoa;

static constexpr int32_t GLOBAL = 1;

std::unique_ptr<spoa::AlignmentEngine> make_engine() {
    return spoa::AlignmentEngine::Create( static_cast<spoa::AlignmentType>(GLOBAL), 5, -4, -8, -6, -10, -4);
}

struct GraphRepr {
    std::vector<char> seqs;
    std::vector<std::vector<int32_t>> outEdges;
    std::vector<std::vector<int32_t>> inEdges;
};

struct SequenceGroup {
    py::list sequences;
    std::vector<int32_t> scores;
    std::string consensus;
    std::unique_ptr<spoa::Graph> graph;
    SequenceGroup(py::list sequences): sequences{sequences} {}
    void build(int min_coverage=-1, spoa::AlignmentEngine* engine=nullptr) {
        if(min_coverage <= 0) {
            min_coverage = std::max(py::size_t(0), (sequences.size() + 1) / 2);
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
            const auto alignment = engine->Align(ptr, *graph, &score);
            graph->AddAlignment(alignment, ptr);
        }
        consensus = graph->GenerateConsensus(min_coverage);
    }
    GraphRepr GenerateGraph() {
        GraphRepr ret;
        std::vector<char> seqs;
        std::unordered_map<Edge*, int32_t> edgeIdMap;
        for(const auto& edge: graph->edges_) {
            const int32_t id = edgeIdMap.size();
            edgeIdMap.emplace(edge.get(), id);
        }
        for(auto& node: graph->nodes_) {
            seqs.push_back(node->code);
        }
        ret.seqs = std::move(seqs);
        return ret;
    }
};

void init_poa(py::module &m) {
    py::class_<SequenceGroup>(m, "SequenceGraph")
    .def(py::init<py::list>())
    .def("build", [](SequenceGroup& group, int minCov) {group.build(minCov);})
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
