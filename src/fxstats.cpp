#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <zlib.h>
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

void init_fxstats(py::module &m) {
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

