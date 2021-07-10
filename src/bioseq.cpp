#include "bioseq.h"

void init_omp_helpers(py::module &m);
PYBIND11_MODULE(cbioseq, m) {
    init_tokenize(m);
    init_omp_helpers(m);
}
