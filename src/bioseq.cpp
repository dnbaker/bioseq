#include "bioseq.h"

void init_omp_helpers(py::module &m);
void init_fxstats(py::module &m);
void init_poa(py::module &m);
PYBIND11_MODULE(cbioseq, m) {
    init_tokenize(m);
    init_omp_helpers(m);
    init_fxstats(m);
    init_poa(m);
}
