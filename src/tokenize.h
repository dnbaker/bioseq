#include "pybind11/pybind11.h"
#include "alphabet.h"
#include "pybind11/numpy.h"
#include <string>

using namespace alph;

struct Tokenizer {
    const Alphabet *ca_;
    const bool include_eos_;
    const bool include_bos_;
    const bool zero_onehot_pad_;
    // Whether to pad with all 0s (instead of one-hot encoding with a wholly new character for 'padding')
    // If true, then trailing sections are left as all 0s
    // if false, then the pad() character is one-hot encoded for padding sections

    size_t full_alphabet_size() const {return ca_->nchars() + include_eos_ + include_bos_ + zero_onehot_pad_;}
    int bos() const {if(!include_bos_) return -1; return ca_->nchars();}
    int eos() const {if(!include_eos_) return -1; return bos() + 1;}
    int pad() const {return eos() + 1;}
    bool is_padded() const {return zero_onehot_pad_;}
    bool includes_bos() const {return include_bos_;}
    bool includes_eos() const {return include_eos_;}
    // Always included: padding
    Tokenizer(const Alphabet &ca, bool eos=false, bool bos=false, bool zero_onehot_pad=false): ca_(&ca), include_eos_(eos), include_bos_(bos), zero_onehot_pad_(zero_onehot_pad) {
    }
    Tokenizer(std::string key, bool include_eos, bool include_bos, bool zohpad): include_eos_(include_eos), include_bos_(include_bos), zero_onehot_pad_(zohpad) {
        std::transform(key.begin(), key.end(), key.begin(),[](auto x){return std::toupper(x);});
        auto it = CAMAP.find(key);
        if(it == CAMAP.end()) {
            std::string options;
            for(const auto &pair: CAMAP) options += pair.first, options += ';';
            throw std::runtime_error(std::string("Invalid tokenizer type; select one from") + options);
        }
        ca_ = it->second;
    }
    template<typename T>
    py::array_t<T> tokenize(const std::string &seq, py::ssize_t padlen=0) const {
        return tokenize<T>(seq.data(), seq.size(), padlen);
    }
    template<typename T>
    py::array_t<T> tokenize(const char *seq, const py::ssize_t seqsz, py::ssize_t padlen=0) const {
        if(padlen > 0) {
            if(seqsz > padlen) throw std::runtime_error("padlen is too short to accommodate sequence\n");
        }
        py::array_t<T> ret;
        const py::ssize_t nc = full_alphabet_size();
        ret.resize({std::max(py::ssize_t(seqsz), padlen) + include_bos_ + include_eos_, nc});
        py::buffer_info bi = ret.request();
        T *ptr = (T *)bi.ptr, *offp = ptr;
        if(include_bos_)
            ptr[bos()]  = 1, offp += nc;
        for(py::ssize_t i = 0; i < seqsz; ++i) {
            //, translated to %d\n", i, seq[i], seq[i], ca_->translate(seq[i]));
            assert(std::strlen(seq) > i);
            auto offset = ca_->translate(seq[i]);
            assert(offset < full_alphabet_size());
            assert(offset >= 0u);
            offp[offset] = 1, offp += nc;
        }
        if(include_eos_)
            offp[eos()] = 1, offp += nc;
        if(zero_onehot_pad_) {
            for(auto pos = (offp - ptr) / nc;pos < padlen; ++pos) {
                ptr[pos * nc + pad()] = 1;
            }
        }
        return ret;
    }
    template<typename T>
    py::array_t<T> tokenize(const std::vector<std::string> &seqs, py::ssize_t padlen=0, bool batch_first = false) const {
        if(seqs.empty()) {
            throw std::invalid_argument(std::string("Cannot tokenize an empty set of sequences; len: ") + std::to_string(seqs.size()));
        }
        const size_t mseqlen = std::accumulate(seqs.begin(), seqs.end(), size_t(0), [](auto x, const auto &y) {return std::max(x, y.size());});
        if(padlen > 0) {
            if(mseqlen > padlen) throw std::runtime_error("padlen is too short to accommodate sequence batch\n");
        } else {
            padlen = mseqlen;
        }
        py::array_t<T> ret;
        const py::ssize_t batchsize = seqs.size();
        const py::ssize_t nc = full_alphabet_size();
        py::ssize_t nr = padlen + include_bos_ + include_eos_;
        const auto mul = nc * nr;
        if(batch_first) {
            ret.resize({batchsize, nr, nc});
        } else {
            ret.resize({nr, batchsize, nc});
        }
        const size_t total_nregs = nr * batchsize * nc;
        py::buffer_info bi = ret.request();
        T *ptr = (T *)bi.ptr, *offp = ptr;
        if(0) {
#if 0
#ifdef _OPENMP
    #pragma omp parallel for
#endif
            for(size_t i = 0; i < seqs.size(); ++i) {
                const auto &seq(seqs[i]);
                auto tp = &offp[i * mul];
                if(include_bos_)
                    tp[bos()] = 1, tp += nc;
                for(size_t j = 0; j < seq.size(); ++j)
                    tp[ca_->translate(seq[j])] = 1, tp += nc;
                if(include_eos_)
                    tp[eos()] = 1, tp += nc;
                if(zero_onehot_pad_) {
                    for(auto ep = &offp[(i + 1) * mul];tp < ep;tp += nc)
                        tp[pad()] = 1;
                }
            }
#endif
        } else {
            for(size_t i = 0; i < seqs.size(); ++i) {
                const auto &seq(seqs[i]);
                if(include_bos_)
                    ptr[i * nc + bos()] = 1;
                for(size_t j = 0; j < seq.size(); ++j) {
                    auto tr = ca_->translate(seq[i]);
                    assert(tr >= 0);
                    assert(tr < full_alphabet_size());
                    assert(ptr + (include_bos_ + j) * nr * nc + i * nc + ca_->translate(seq[i]) < ptr + total_nregs);
                    ptr[(include_bos_ + j) * nr * nc + i * nc + ca_->translate(seq[i])] = 1;
                }
                if(include_eos_) {
                    ptr[(include_bos_ + seq.size()) * nr * nc + i * nc + eos()] = 1;
                }
                if(zero_onehot_pad_) {
                    for(py::ssize_t myi = seq.size() + include_bos_ + include_eos_; myi < padlen; ++myi)
                        ptr[myi * nr * nc + i * nc + pad()] = 1;
                }
            }
        }
    }
    template<typename T>
    py::array_t<T> tokenize(py::sequence items, py::ssize_t padlen=-1, bool batch_first = false, py::ssize_t nthreads = 1) const {
        if(padlen <= 0) throw std::invalid_argument("batch tokenize requires padlen is provded.");
        if(nthreads <= 0) nthreads = 1;
        const py::ssize_t nc = full_alphabet_size();
        py::ssize_t nr = padlen + include_bos_ + include_eos_;
        std::vector<std::pair<const char *, size_t>> strs;
        py::ssize_t nitems = 0;
        for(auto item: items) {
            py::ssize_t size;
            if(py::isinstance<py::str>(item)) {
                const char *s = PyUnicode_AsUTF8AndSize(item.ptr(), &size);
                strs.push_back({s, size});
            } else if(py::isinstance<py::bytes>(item)) {
                char *s;
                if(PyBytes_AsStringAndSize(item.ptr(), &s, &size))
                    throw std::invalid_argument("item is not a bytes object; this should never happen.");
                strs.push_back({s, size});
            } else if(py::isinstance<py::bytearray>(item)) {
                strs.push_back({PyByteArray_AS_STRING(item.ptr()), PyByteArray_GET_SIZE(item.ptr())});
            } else if(py::isinstance<py::array>(item)) {
                auto inf = py::cast<py::array>(item).request();
                switch(inf.format.front()) {
                    case 'b': case 'B': {
                        strs.push_back({(const char *)inf.ptr, size_t(inf.size)});
                    }
                    default: goto invalid;
                }
            } else {
                invalid:
                throw std::invalid_argument("item was none of string, bytes, or numpy array of 8-bit integers. ");
            }
            ++nitems;
        }
        if(batch_first) {
            throw std::invalid_argument("Batch first is disabled. Instead, using Einops' rearrange to correct the shape.");
        }
        py::array_t<T> ret(std::vector<py::ssize_t>({nr, nitems, nc})); // T B C
        const auto nrc = nitems * nc;
#define __access(seqind, batchind, charind) \
        assert((seqind) * nrc + (batchind) * nc + (charind) < nr * nitems * nc);\
        ptr[(seqind) * nrc + (batchind) * nc + (charind)]
        py::buffer_info bi = ret.request();
        std::memset(bi.ptr, 0, sizeof(T) * nitems * nr * nc);
        T *ptr = (T *)bi.ptr;
#ifdef _OPENMP
    #pragma omp parallel for num_threads(nthreads)
#endif
        for(size_t i = 0; i < strs.size(); ++i) {
            const auto &seq(strs[i]);
            if(include_bos_) {
                __access(0, i, bos()) = 1;
            }
            for(size_t j = 0; j < seq.second; ++j) {
                auto tr = ca_->translate(seq.first[j]);
                __access((include_bos_ + j), i, tr) = 1;
            }
            if(include_eos_) {
                __access((include_bos_ + seq.second), i, eos()) = 1;
            }
            if(static_cast<py::ssize_t>(seq.second + include_bos_ + include_eos_) > padlen) {
                auto tl = seq.second + include_bos_ + include_eos_;
                throw std::invalid_argument(std::string("seq len + bos + eos > padlen: ") + std::to_string(tl) + ", vs padlen " + std::to_string(padlen));
            }
            if(zero_onehot_pad_) {
                for(py::ssize_t k = seq.second + include_bos_ + include_eos_; k < padlen;)
                {
                    __access(k++, i, pad()) = 1;
                }
            }
        }
        return ret;
    }
    template<typename T>
    py::object transencode(py::sequence items, py::ssize_t padlen=-1, bool batch_first = false, py::ssize_t nthreads = 1) const {
        if(padlen <= 0) throw std::invalid_argument("batch tokenize requires padlen is provded.");
        if(nthreads <= 0) nthreads = 1;
        py::ssize_t nr = padlen + include_bos_ + include_eos_;
        std::vector<std::pair<const char *, size_t>> strs;
        py::ssize_t nitems = 0;
        for(auto item: items) {
            py::ssize_t size;
            if(py::isinstance<py::str>(item)) {
                const char *s = PyUnicode_AsUTF8AndSize(item.ptr(), &size);
                strs.push_back({s, size});
            } else if(py::isinstance<py::bytes>(item)) {
                char *s;
                if(PyBytes_AsStringAndSize(item.ptr(), &s, &size))
                    throw std::invalid_argument("item is not a bytes object; this should never happen.");
                strs.push_back({s, size});
            } else if(py::isinstance<py::bytearray>(item)) {
                strs.push_back({PyByteArray_AS_STRING(item.ptr()), PyByteArray_GET_SIZE(item.ptr())});
            } else if(py::isinstance<py::array>(item)) {
                auto inf = py::cast<py::array>(item).request();
                switch(inf.format.front()) {
                    case 'b': case 'B': {
                        strs.push_back({(const char *)inf.ptr, size_t(inf.size)});
                    }
                    default: goto invalid;
                }
            } else {
                invalid:
                throw std::invalid_argument("item was none of string, bytes, or numpy array of 8-bit integers. ");
            }
            ++nitems;
        }
        py::object ret = py::none();
        if(batch_first) {
            ret = py::array_t<T>(std::vector<py::ssize_t>({nitems, nr})); // B T
        } else {
            ret = py::array_t<T>(std::vector<py::ssize_t>({nr, nitems})); // T B
        }
        py::buffer_info bi = ret.cast<py::array_t<T>>().request();
        std::memset(bi.ptr, 0, sizeof(T) * nitems * nr);
        const size_t padl = nr;
        T *ptr = (T *)bi.ptr;
#define __assign_bf(seqind, batchind, charind) \
        do {\
            assert(seqind + batchind * padl < nitems * padl);\
            ptr[batchind * padl + seqind] = charind;\
        } while(0)
#define __assign_tf(seqind, batchind, charind) \
        do {\
            assert(seqind * nitems + batchind < nitems * padl);\
            ptr[seqind * nitems + batchind] = charind;\
        } while(0)
#define __assign(seqind, batchind, charind) \
        do {\
            if(charind >= 0) {\
                if(batch_first) {\
                    __assign_bf(seqind, batchind, charind);\
                } else {\
                    __assign_tf(seqind, batchind, charind);\
                }\
            }\
        } while(0)

#ifdef _OPENMP
    #pragma omp parallel for num_threads(nthreads)
#endif
        for(size_t i = 0; i < strs.size(); ++i) {
            const auto &seq(strs[i]);
            if(__builtin_expect(static_cast<py::ssize_t>(seq.second + include_bos_ + include_eos_) > padlen, 0)) {
                auto tl = seq.second + include_bos_ + include_eos_;
                throw std::runtime_error(std::string("seq len + bos + eos > padlen: ") + std::to_string(tl) + ", vs padlen " + std::to_string(padlen));
            }
            if(include_bos_) {
                __assign(0, i, bos());
            }
            for(size_t j = 0; j < seq.second; ++j) {
                auto tr = ca_->translate(seq.first[j]);
                __assign((include_bos_ + j), i, tr);
            }
            if(include_eos_) {
                __assign((include_bos_ + seq.second), i, eos());
            }
            for(py::ssize_t k = seq.second + include_bos_ + include_eos_; k < padlen;)
            {
                __assign(k++, i, pad());
            }
        }
        return ret;
#undef __assign_tf
#undef __assign_bf
#undef __assign
#undef __access
    }
};
