#include "bioseq.h"
#include "tokenize.h"

void init_tokenize(py::module &m) {
    py::class_<Tokenizer>(m, "Tokenizer")
    .def(py::init<std::string, bool, bool, bool>(), py::arg("key"), py::arg("eos") = false, py::arg("bos") = false, py::arg("padchar") = false)
    .def("onehot_encode", [](const Tokenizer &tok, py::str s, py::ssize_t padlen, std::string dt) -> py::object {
        py::ssize_t size;
        const char *ptr = PyUnicode_AsUTF8AndSize(s.ptr(), &size);
        if(dt[0] == 'B' || dt[0] == 'b')
            return tok.tokenize<uint8_t>(ptr, size, padlen);
        if(dt[0] == 'h' || dt[0] == 'H')
            return tok.tokenize<uint16_t>(ptr, size, padlen);
        if(dt[0] == 'i' || dt[0] == 'I')
            return tok.tokenize<uint32_t>(ptr, size, padlen);
        if(dt[0] == 'f')
            return tok.tokenize<float>(ptr, size, padlen);
        if(dt[0] == 'd')
            return tok.tokenize<double>(ptr, size, padlen);
        throw std::invalid_argument(std::string("Unsupported dtype: ") + dt);
    }, py::arg("str"), py::arg("padlen") = 0, py::arg("destchar") = "f")
    .def("onehot_encode", [](const Tokenizer &tok, py::bytes bs, py::ssize_t padlen, std::string dt) -> py::object {
        py::ssize_t size;
        char *ptr;
        PyBytes_AsStringAndSize(bs.ptr(), &ptr, &size);
        if(dt[0] == 'B' || dt[0] == 'b')
            return tok.tokenize<uint8_t>(ptr, size, padlen);
        if(dt[0] == 'h' || dt[0] == 'H')
            return tok.tokenize<uint16_t>(ptr, size, padlen);
        if(dt[0] == 'i' || dt[0] == 'I')
            return tok.tokenize<uint32_t>(ptr, size, padlen);
        if(dt[0] == 'f')
            return tok.tokenize<float>(ptr, size, padlen);
        if(dt[0] == 'd')
            return tok.tokenize<double>(ptr, size, padlen);
        throw std::invalid_argument(std::string("Unsupported dtype: ") + dt);
    }, py::arg("str"), py::arg("padlen") = 0, py::arg("destchar") = "B")
    // batched one-hot encoding
    .def("batch_onehot_encode", [](const Tokenizer &tok, py::sequence seq, py::ssize_t padlen, std::string dt, int nthreads) -> py::object {
        switch(std::tolower(dt[0])) {
#define C(x, t) case x: return tok.tokenize<t>(seq, padlen, false, nthreads)
            default:
                      C('b', int8_t);
                      C('B', uint8_t);
                      C('h', int16_t);
                      C('H', uint16_t);
                      C('I', uint32_t);
                      C('i', int32_t);
            case 'l': C('q', uint64_t);
            case 'L': C('Q', int64_t);
            C('f', float);
            C('d', double);
#undef C
        }
        throw std::invalid_argument(std::string("Unsupported dtype: ") + dt);
    }, py::arg("batch"), py::arg("padlen") = -1, py::arg("destchar") = "B",  py::arg("nthreads") = 1)
    .def("batch_tokenize", [](const Tokenizer &tok, py::sequence seq, py::ssize_t padlen, std::string dt, bool batch_first, int nthreads) -> py::object {
        switch(std::tolower(dt[0])) {
#define C(x, t) case x: return tok.transencode<t>(seq, padlen, batch_first, nthreads)
            default:
                      C('b', int8_t);
                      C('B', uint8_t);
                      C('h', int16_t);
                      C('H', uint16_t);
                      C('I', uint32_t);
                      C('i', int32_t);
            case 'l': C('q', uint64_t);
            case 'L': C('Q', int64_t);
            C('f', float);
            C('d', double);
#undef C
        }
        throw std::invalid_argument(std::string("Unsupported dtype: ") + dt);
    }, py::arg("batch"), py::arg("padlen") = -1, py::arg("destchar") = "B", py::arg("batch_first")=false, py::arg("nthreads") = 1)
    .def("alphabet_size", &Tokenizer::full_alphabet_size)
    .def("bos", &Tokenizer::bos)
    .def("eos", &Tokenizer::eos)
    .def("pad", &Tokenizer::pad)
    .def("is_padded", &Tokenizer::is_padded)
    .def("includes_bos", &Tokenizer::includes_bos)
    .def("includes_eos", &Tokenizer::includes_eos)
    .def(py::pickle(
        [](const Tokenizer &tok) -> py::tuple {return py::make_tuple(tok.key, tok.include_eos_, tok.include_bos_, tok.zero_onehot_pad_);},
        [](py::tuple t) {
            return Tokenizer(t[0].cast<std::string>(), t[1].cast<bool>(), t[2].cast<bool>(), t[3].cast<bool>());
        }
    ));
}
