#include "bioseq.h"
#include "tokenize.h"

inline __attribute__((always_inline))
py::object tokenize(const Tokenizer &tok, const char *s, const py::ssize_t size, const py::ssize_t padlen, const std::string dt) {
    py::object ret = py::none();
    switch(dt[0] & 223) { // remove case from character by removing bit 0b100000 == 32
        case 'B': ret = tok.tokenize<uint8_t>(s, size, padlen); break;
        case 'H': ret = tok.tokenize<uint16_t>(s, size, padlen); break;
        case 'I': ret = tok.tokenize<uint32_t>(s, size, padlen); break;
        case 'F': ret = tok.tokenize<float>(s, size, padlen); break;
        case 'D': ret = tok.tokenize<double>(s, size, padlen); break;
        default: ; // Else, return None
    }
    return ret;
}

void init_tokenize(py::module &m) {
    py::class_<Tokenizer>(m, "Tokenizer")
    .def(py::init<std::string, bool, bool, bool>(), py::arg("key"), py::arg("eos") = false, py::arg("bos") = false, py::arg("padchar") = false)
    .def("onehot_encode", [](const Tokenizer &tok, py::str s, py::ssize_t padlen, std::string dt) -> py::object {
        py::ssize_t size;
        const char *ptr = PyUnicode_AsUTF8AndSize(s.ptr(), &size);
        py::object ret = tokenize(tok, ptr, size, padlen, dt);
        if(ret.is_none())
            throw std::invalid_argument(std::string("Unsupported dtype: ") + dt);
        return ret;
    }, py::arg("str"), py::arg("padlen") = 0, py::arg("destchar") = "f")
    .def("onehot_encode", [](const Tokenizer &tok, py::bytearray s, py::ssize_t padlen, std::string dt) -> py::object {
        const py::ssize_t size = PyByteArray_GET_SIZE(s.ptr());
        const char *ptr = PyByteArray_AS_STRING(s.ptr());
        py::object ret = tokenize(tok, ptr, size, padlen, dt);
        if(ret.is_none())
            throw std::invalid_argument(std::string("Unsupported dtype: ") + dt);
        return ret;
    }, py::arg("bytearray"), py::arg("padlen") = 0, py::arg("destchar") = "f")
    .def("onehot_encode", [](const Tokenizer &tok, py::bytes bs, py::ssize_t padlen, std::string dt) -> py::object {
        py::ssize_t size;
        char *ptr;
        PyBytes_AsStringAndSize(bs.ptr(), &ptr, &size);
        py::object ret = tokenize(tok, ptr, size, padlen, dt);
        if(ret.is_none())
            throw std::invalid_argument(std::string("Unsupported dtype: ") + dt);
        return ret;
    }, py::arg("str"), py::arg("padlen") = 0, py::arg("destchar") = "B")
    .def("decode_tokens", [](const Tokenizer& tok, py::array array) {
        return tok.decode_tokens_to_string(array);
    })
    .def("token_map", [](const Tokenizer& tok) {
        return tok.token_map();
    })
    .def("nchars", [](const Tokenizer& tok) {
        return tok.nchars();
    })
    // batched one-hot encoding
    .def("batch_onehot_encode", [](const Tokenizer &tok, py::sequence seq, py::ssize_t padlen, std::string dt, int nthreads, py::object mask) -> py::object {
        switch(std::tolower(dt[0])) {
#define C(x, t) case x: return tok.tokenize<t>(seq, padlen, false, nthreads, mask)
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
    }, py::arg("batch"), py::arg("padlen") = -1, py::arg("destchar") = "B",  py::arg("nthreads") = 1, py::arg("mask") = py::none())
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
    .def_property_readonly("key", [](const Tokenizer &tok) {return tok.key;})
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
