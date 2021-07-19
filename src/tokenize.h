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
#ifndef NDEBUG
        for(size_t i = 0; i < 256; ++i) {
            std::fprintf(stderr, "using %d/%c->%d/%c\n", int(i), char(i), ca_->lut[i], ca_->lut[i]);
        }
        std::fprintf(stderr, "Tokenizer has %zu commas/%zu chars\n", ca_->nc, ca_->nc + 1);
#endif
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
#ifndef NDEBUG
            std::fprintf(stderr, "char %zu is %d/%c", i, seq[i], seq[i]);
#endif
            //, translated to %d\n", i, seq[i], seq[i], ca_->translate(seq[i]));
            offp[ca_->translate(seq[i])] = 1, offp += nc;
        }
        if(include_eos_)
            offp[eos()] = 1, offp += nc;
        if(zero_onehot_pad_) {
            for(auto pos = (offp - ptr) / nc;pos < padlen; ++pos)
                ptr[pos * nc + pad()] = 1;
        }
        return ret;
    }
    template<typename T>
    py::array_t<T> tokenize(const std::vector<std::string> &seqs, py::ssize_t padlen=0, bool batch_first = false) const {
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
        py::buffer_info bi = ret.request();
        T *ptr = (T *)bi.ptr, *offp = ptr;
        if(batch_first) {
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
        } else {
#ifdef _OPENMP
    #pragma omp parallel for
#endif
            for(size_t i = 0; i < seqs.size(); ++i) {
                const auto &seq(seqs[i]);
                if(include_bos_)
                    ptr[i * nc + bos()] = 1;
                for(size_t j = 0; j < seq.size(); ++j)
                    ptr[(include_bos_ + j) * nr * nc + i * nc + ca_->translate(seq[i])] = 1;
                if(include_eos_)
                    ptr[(include_bos_ + seq.size()) * nr * nc + i * nc + eos()] = 1;
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
            std::fprintf(stderr, "Batch first seems to be buggy. Instead, using Einops' rearrange to correct the shape.");
            batch_first = false;
        }
        py::array_t<T> ret(batch_first ? std::vector<py::ssize_t>({nitems, nr, nc}): std::vector<py::ssize_t>({nr, nitems, nc}));
        py::buffer_info bi = ret.request();
        std::memset(bi.ptr, 0, sizeof(T) * nitems * nr * nc);
        T *ptr = (T *)bi.ptr, *offp = ptr;
        const auto nrc = nr * nc;
#ifdef _OPENMP
#pragma omp parallel for num_threads(nthreads)
#endif
        for(size_t i = 0; i < strs.size(); ++i) {
            const auto &seq(strs[i]);
            if(include_bos_)
                ptr[i * nc + bos()] = 1;
            for(size_t j = 0; j < seq.second; ++j)
                ptr[(include_bos_ + j) * nrc + i * nc + ca_->translate(seq.first[i])] = 1;
            if(include_eos_)
                ptr[(include_bos_ + seq.second) * nrc + i * nc + eos()] = 1;
            if(zero_onehot_pad_)
                for(py::ssize_t k = seq.second + include_bos_ + include_eos_;
                    k < padlen;
                    ptr[k++ * nrc + i * nc + pad()] = 1);
        }
        return ret;
    }
};
