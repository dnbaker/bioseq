#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <zlib.h>
#include "mio.hpp"
#include "span.hpp"
#include "kseq.h"
namespace py = pybind11;

KSEQ_INIT(gzFile, gzread)

std::vector<size_t> getlens(const std::string &path) {
    gzFile fp = gzopen(path.data(), "r");
    if(fp == nullptr) throw std::runtime_error(path + " failed to open");
    kseq_t *kseq = kseq_init(fp);
    std::vector<size_t> lens;
    while(kseq_read(kseq) >= 0)
        lens.push_back(kseq->seq.l);
    kseq_destroy(kseq);
    gzclose(fp);
    return lens;
}

struct FlatFile {
    const std::string path_;
    const mio::mmap_source data_;
    const size_t nseqs_;
    nonstd::span<size_t> offsets_;
    const size_t seq_offset_;
    FlatFile(std::string path): path_(path), data_(path_), nseqs_(*(uint64_t *)(data_.data())),
                                offsets_((size_t *)(data_.data()) + 1, nseqs_ + 1), seq_offset_((nseqs_ + 2) * 8)
    {
        //std::fprintf(stderr, "flatfile has %zu seqs\n", nseqs_);
    }
    FlatFile(FlatFile &&o) = delete;
    FlatFile(const FlatFile &o) = default;
    auto &offsets() {return offsets_;}
    const auto &offsets() const {return offsets_;}
    size_t nseqs() const {
        return nseqs_;
    }
    size_t seq_offset() const {
        return seq_offset_;
    }
    py::bytes access(size_t i) const {
        if(i >= nseqs_) throw std::out_of_range("Accessing sequence out of range");
        auto start = offsets_[i] + seq_offset_;
        auto len = offsets_[i + 1] - offsets_[i];
        return py::bytes(&data_[start], len);
    }
};

void init_fxstats(py::module &m) {
    py::class_<FlatFile>(m, "FlatFile")
    .def(py::init<std::string>())
    .def("access", &FlatFile::access)
    .def("nseqs", &FlatFile::nseqs)
    .def("seq_offset", &FlatFile::seq_offset);
    m.def("makeflat", [](std::string inpath, std::string outpath) {
        if(outpath.empty()) throw std::invalid_argument("outpath must be provided");
        std::vector<size_t> offsets{0};
        std::vector<uint32_t> seqlens;
        gzFile fp = gzopen(inpath.data(), "r");
        if(fp == nullptr) throw std::runtime_error(inpath + " failed to open");
        kseq_t *ks = kseq_init(fp);
        std::string cseq;
        std::vector<std::string> seqs;
        while(kseq_read(ks) >= 0) {
            seqlens.push_back(ks->seq.l);
            offsets.push_back(offsets.back() + ks->seq.l);
            seqs.emplace_back(ks->seq.s, ks->seq.l);
        }
        std::FILE *ofp = std::fopen(outpath.data(), "w");
        if(!ofp) throw std::runtime_error(outpath + " could not be opened for writing");
        uint64_t nseqs = seqs.size();
        // 8 bytes: number of sequences
        // 8 * nseqs: offsets to start of sequences
        std::fwrite(&nseqs, sizeof(nseqs), 1, ofp);
        std::fwrite(offsets.data(), sizeof(uint64_t), seqlens.size(), ofp);
        for(const auto &s: seqs) {
            std::fwrite(s.data(), 1, s.size(), ofp);
        }
        kseq_destroy(ks);
        gzclose(fp);
        std::fclose(ofp);
        return outpath;
    }, py::arg("input"), py::arg("output") = "");
    m.def("getstats", [](py::sequence items) {
        std::vector<std::string> paths;
        for(const auto item: items)
            paths.emplace_back(item.cast<std::string>());
        py::list alist = py::list();
        while(py::len(alist) < paths.size()) {
            alist.append(py::none());
        }
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
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
}

