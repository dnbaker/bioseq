#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
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


struct FlatFileIterator;
struct FlatFile {
    const std::string path_;
    mio::mmap_source data_;
    const size_t nseqs_;
    nonstd::span<size_t> offsets_;
    const size_t seq_offset_;
    uint32_t max_seq_len_;
    static FlatFile make(std::string inpath, std::string outpath) {
        if(outpath.empty()) {
            outpath = inpath + ".ff";
        }
        std::vector<size_t> offsets{0};
        uint32_t max_seq_len = 0;
        gzFile fp = gzopen(inpath.data(), "r");
        if(fp == nullptr) throw std::runtime_error(inpath + " failed to open");
        kseq_t *ks = kseq_init(fp);
        std::string cseq;
        std::vector<std::string> seqs;
        while(kseq_read(ks) >= 0) {
            if(ks->seq.l > 0xFFFFFFFFu) throw std::invalid_argument("Cannot handle sequences longer than 2^32 - 1");
            max_seq_len = std::max(max_seq_len, uint32_t(ks->seq.l));
            offsets.push_back(offsets.back() + ks->seq.l);
            seqs.emplace_back(ks->seq.s, ks->seq.l);
        }
        std::FILE *ofp = std::fopen(outpath.data(), "w");
        if(!ofp) throw std::runtime_error(outpath + " could not be opened for writing");
        uint64_t nseqs = seqs.size();
        // 8 bytes: number of sequences
        // 8 * nseqs: offsets to start of sequences
        std::fwrite(&nseqs, sizeof(nseqs), 1, ofp);
        std::fwrite(offsets.data(), sizeof(uint64_t), offsets.size(), ofp);
        for(const auto &s: seqs) {
            std::fwrite(s.data(), 1, s.size(), ofp);
        }
        kseq_destroy(ks);
        gzclose(fp);
        std::fclose(ofp);
        return FlatFile(outpath, max_seq_len);
    }
    FlatFile(std::string inpath, std::string outpath): FlatFile(FlatFile::make(inpath, outpath)) {}
    FlatFile(std::string path, uint32_t mslen=-1): path_(path), data_(path_), nseqs_(*(uint64_t *)(data_.data())),
                                offsets_((size_t *)(data_.data()) + 1, nseqs_ + 1), seq_offset_((nseqs_ + 2) * 8), max_seq_len_(mslen)
    {
        if(mslen == uint32_t(-1)) {
            max_seq_len_ = 0;
            for(size_t i = 0; i < nseqs(); ++i) {
                if(const auto L = length(i);L > max_seq_len_) max_seq_len_ = L;
            }
        }
    }
    FlatFile(FlatFile &&o) = delete;
    FlatFile(const FlatFile &o) = default;
    auto &offsets() {return offsets_;}
    const auto &offsets() const {return offsets_;}
    uint32_t max_seq_len() const {return max_seq_len_;}
    size_t nseqs() const {
        return nseqs_;
    }
    size_t seq_offset() const {
        return seq_offset_;
    }
    size_t length(size_t idx) const {
        return offsets_[idx + 1] - offsets_[idx];
    }
    const char*offset(size_t idx) const {
        return data_.data() + offsets_[idx] + seq_offset_;
    }
    py::list range_access(py::slice slc) const {
        return range_access(slc.attr("start").cast<py::ssize_t>(), slc.attr("stop").cast<py::ssize_t>(), slc.attr("step").cast<py::ssize_t>());
    }
    py::list range_access(py::ssize_t i, py::ssize_t j, py::ssize_t step) const {
        py::list ret;
        if(step == 0) throw std::invalid_argument("step must be nonzero");
        if(step > 0) {
            for(py::ssize_t idx = i; idx < j; idx += step)
                ret.append(py::bytearray(offset(idx), length(idx)));
        } else {
            for(py::ssize_t idx = i; idx > j; idx += step)
                ret.append(py::bytearray(offset(idx), length(idx)));
        }
        return ret;
    }
    py::bytearray access(size_t i) const {
        if(i >= nseqs_) throw std::out_of_range("Accessing sequence out of range");
        auto start = offsets_[i] + seq_offset_;
        auto len = offsets_[i + 1] - offsets_[i];
        return py::bytearray(&data_[start], len);
    }
};

struct FlatFileIterator {
    const FlatFile *ptr_;
    size_t start_;
    size_t stop_;
    FlatFileIterator(const FlatFile &src): ptr_(&src), start_(0), stop_(src.nseqs()) {}
    //FlatFileIterator(const FlatFileIterator &) = default;
    FlatFileIterator(const FlatFile *ptr, size_t start, size_t stop): ptr_(ptr), start_(start), stop_(stop) {}
    FlatFileIterator &next() {
        if(__builtin_expect(++start_== stop_, 0))
            throw py::stop_iteration("End of iterator");
        return *this;
    }
    py::bytearray sequence() const {
        return ptr_->access(start_);
    }
};

void init_fxstats(py::module &m) {
    py::class_<FlatFileIterator>(m, "FlatFileIterator")
    .def(py::init<FlatFileIterator>())
    .def("__iter__", [](const FlatFileIterator &x) {return x;})
    .def("__next__", [](FlatFileIterator &x) {return x.next();})
    .def_property_readonly("sequence", &FlatFileIterator::sequence)
    .def_property_readonly("seq", &FlatFileIterator::sequence);

    py::class_<FlatFile>(m, "FlatFile")
    .def(py::init<std::string>())
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
    .def("maxseqlen", &FlatFile::max_seq_len)
    .def("__iter__", [](const FlatFile &x) {return FlatFileIterator(x);}, py::keep_alive<0, 1>());


    m.def("makeflat", [](std::string inpath, std::string outpath) {
        return FlatFile::make(inpath, outpath);
    }, py::arg("input"), py::arg("output") = "");
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
}
