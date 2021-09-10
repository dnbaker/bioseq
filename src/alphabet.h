#ifndef BIOSEQ_ALPHBET_H__
#define BIOSEQ_ALPHBET_H__
#include <array>
#include <map>
#include <cstdint>
#include <string>
#include <string_view>
#include <climits>
#include <type_traits>
#include <cassert>
#include <cstring>
#include <vector>

namespace alph {
using std::size_t;

template<typename VT=int8_t, size_t NCHAR=size_t(1) << (sizeof(VT) * CHAR_BIT)>
struct TAlphabet {
    static_assert(NCHAR > 1, "Nchar must be positive");
    static_assert(std::is_integral<VT>::value, "VT must be integral");
    const char *name;
    const char *setstr;
private:
    size_t nc;
public:
    bool padding;
    size_t nchars() const {return nc + 1;} // One for padding
    using LUType = std::array<VT, NCHAR>;
    LUType lut;
    static constexpr LUType make_lut(const char *s, const size_t nc, bool padding=false, const char *aliases=0) {
        LUType arr{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1};
        int id = padding;
        size_t ci = 0;
        for(size_t i = 0; i < nc; ++i, ++id, ++ci) {
            while(s[ci] && s[ci] != ',') {
                const auto v = s[ci++];
                arr[v | 32] = arr[v & static_cast<uint8_t>(0xdf)] = id; // lower-case and upper-case
            }
        }
        while(s[ci]) {
            const auto v = s[ci++];
            arr[v | 32] = arr[v & static_cast<uint8_t>(0xdf)] = id; // lower-case and upper-case
        }
        if(aliases) {
            const char *p = aliases;
            while(*p && *p != ':') ++p;
            if(*p) {
                const size_t offset = p - aliases;
                for(size_t i = 0; i < offset; ++i) {
                    const auto destchar = arr[p[i + 1]];
                    if(arr[aliases[i] & 0xdf] == VT(-1))
                        arr[aliases[i] & 0xdf] = arr[destchar];
                    if(arr[aliases[i] | 32] == VT(-1))
                        arr[aliases[i] | 32] = arr[destchar];
                }
            }
        }
        return arr;
    }
    std::vector<std::pair<VT, VT>> to_sparse() const {
        std::vector<std::pair<VT, VT>> ret;
        for(size_t i = 0; i < NCHAR; ++i) {
            if(lut[i] != VT(-1))
                ret.push_back({VT(i), lut[i]});
        }
        return ret;
    }
    void display() const {
        for(size_t i = 0; i < NCHAR; ++i) {
            if(lut[i] != VT(-1)) {
                std::fprintf(stderr, "Mapping %d/%c to %d/%c\n", int(i), char(i), int(lut[i]), lut[i]);
            }
        }
    }
    using SignedVT = typename std::make_signed<VT>::type;
    constexpr VT translate(VT x) const {return lut[static_cast<SignedVT>(x)];}
    constexpr VT *data() {return lut.data();}
    constexpr const VT *data() const {return lut.data();}
    static constexpr size_t size() {return NCHAR;}
    static constexpr size_t ncommas(const char *s) {
        size_t ret = 0, i = 0;
        while(s[i]) ret += (s[i] == ','), ++i;
        return ret;
    }
    constexpr TAlphabet(const TAlphabet &) = default;
    constexpr TAlphabet(TAlphabet &&) = default;
    constexpr TAlphabet(const char *name, const char *s, bool padding=false, const char *aliases=0): name(name), setstr(s), nc(ncommas(s)), padding(padding), lut(make_lut(s, nc, padding, aliases)) {
    }
    static constexpr LUType emptylut(bool padding) {
        LUType tlut{-1};
        for(size_t i = 0; i < NCHAR; ++i) tlut[i] = i + padding;
        return tlut;
    }
    constexpr TAlphabet(bool padding=false): name("Bytes"), setstr(""), nc(NCHAR - 1 + padding), padding(padding), lut(emptylut(padding)) {
    }
};
struct Alphabet: public TAlphabet<int8_t> {
    template<typename...Args> constexpr Alphabet(Args &&...args): TAlphabet(std::forward<Args>(args)...) {}
};

// Protein Alphabets
// All protein alphabets handle pyrrolysine and selenocysteine if unhandled
// Handle Pyrrolysine (P) by mapping it to Lysine (K) if unhandled
// Handle SelenoCysteine (U) by mapping it to Cysteine if unhandled
// (PU:KC) means P maps to K and U maps to C
static constexpr const Alphabet BYTES;
static constexpr const Alphabet AMINO20("Standard20", "A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y", false, "PU:KC");

static constexpr const Alphabet SEB14("SE-B(14)", "A,C,D,EQ,FY,G,H,IV,KR,LM,N,P,ST,W", false, "PU:KC");

static constexpr const Alphabet SEB10("SE-B(10)", "AST,C,DN,EQ,FY,G,HW,ILMV,KR,P", false, "PU:KC");
static constexpr const Alphabet SEV10("SE-V(10)", "AST,C,DEN,FY,G,H,ILMV,KQR,P,W", false, "PU:KC");
static constexpr const Alphabet SOLISD("Solis-D", "AM,C,DNS,EKQR,F,GP,HT,IV,LY,W", false, "PU:KC");
static constexpr const Alphabet SOLISG("Solis-G", "AEFIKLMQRVW,C,D,G,H,N,P,S,T,Y", false, "PU:KC");
static constexpr const Alphabet MURPHY("Murphy", "A,C,DENQ,FWY,G,H,ILMV,KR,P,ST", false, "PU:KC");
static constexpr const Alphabet LIA10("Li-A(10)", "AC,DE,FWY,G,HN,IV,KQR,LM,P,ST", false, "PU:KC");
static constexpr const Alphabet LIB10("Li-B(10)", "AST,C,DEQ,FWY,G,HN,IV,KR,LM,P", false, "PU:KC");

static constexpr const Alphabet SEB8("SE-B(8)","AST,C,DHN,EKQR,FWY,G,ILMV,P", false, "PU:KC");
static constexpr const Alphabet SEB6("SE-B(6)","AST,CP,DHNEKQR,FWY,G,ILMV", false, "PU:KC");

static constexpr const Alphabet DAYHOFF("Dayhoff","AGPST,C,DENQ,FWY,HKR,ILMV", false, "PU:KC");

namespace amine_traits {
static constexpr const  std::string_view alcoholic_bases("oST");
static constexpr const  std::string_view hydrophobic_bases("hACFGHIKLMRTVWY");
static constexpr const  std::string_view polar_bases("pCDEHKNQRST");
static constexpr const  std::string_view charged_bases("cDEHKR");
static constexpr const  std::string_view positive_bases("+HKR");
static constexpr const  std::string_view negative_bases("-DE");
static constexpr const  std::string_view small_bases("sAGSCDNPTV");
static constexpr const  std::string_view tiny_bases("uAGS");
static constexpr const  std::string_view aromatic_bases("aFHWY");
static constexpr const  std::string_view turnlike_bases("tACDEGHKNQRST");

static constexpr bool is_alcoholic(char c) {
    switch(c) case 'o': case  'S': case 'T': return true;
    return false;
}
static constexpr bool is_hydrophobic(char c) {
    switch(c) {
        case 'h': case 'A': case 'C': case 'F': case 'G': case 'H': case 'I': case 'K': case 'L': case 'M': case 'R': case 'T': case 'V': case 'W': case 'Y':
            return true;
    }
    return false;
}
static constexpr bool is_polar(char c) {
    switch(c) case 'p': case 'C': case 'D': case 'E': case 'H': case 'K': case 'N': case 'Q': case 'R': case 'S': case 'T': return true;
    return false;
}
static constexpr bool is_negative(char c) {
    switch(c) case '-': case 'D': case 'E': return true;
    return false;
}
static constexpr bool is_positive(char c) {
    switch(c) case '+': case 'H': case 'K': case 'R': return true;
    return false;
}
static constexpr bool is_charged(char c) {
    switch(c) case 'c': case 'D': case 'E': case 'H': case 'K': case 'R': return true;
    return false;
}
static constexpr bool is_small(char c) {
    switch(c) case 's': case 'A': case 'G': case 'S': case 'C': case 'D': case 'N': case 'P': case 'T': case 'V': return true;
    return false;
}
static constexpr bool is_tiny(char c) {
    switch(c) case 'u': case 'A': case 'G': case 'S': return true;
    return false;
}
static constexpr bool is_aromatic(char c) {
    switch(c) case 'a': case 'F': case 'H': case 'W': case 'Y': return true;
    return false;
}
static constexpr bool is_turnlike(char c) {
    switch(c) case 't': case 'A':  case 'C': case 'D': case 'E': case 'G': case 'H': case 'K': case 'N': case 'Q': case 'R': case 'S': case 'T':
        return true;
    return false;
}

} // namespace amine_traits


// DNA alphabets
// We also map U to T in order to support RNA sequences

static constexpr const Alphabet DNA4("DNA4", "A,C,G,T", false, "U:T");
static constexpr const Alphabet DNA5("DNA5", "A,C,G,T,NMRWSYKVHDB", false, "U:T");

static constexpr const Alphabet DNA2KETAMINE("DNA2", "ACM,KGT", false, "U:T"); // Amino/Ketones
static constexpr const Alphabet DNA2PYRPUR("DNA2", "AGR,YCT", false, "U:T"); // Purines/Pyrimidines
static constexpr const Alphabet DNA2METHYL("DNAMETH", "C,AGT", false, "U:T"); // Purines/Pyrimidines

// Source: Reference
// Edgar, RC (2004) Local homology recognition and distance measures in linear time using compressed amino acid alphabets, NAR 32(1), 380-385. doi: 10.1093/nar/gkh180
static const std::map<std::string, const Alphabet *> CAMAP {
    {"BYTES", &BYTES},
    {"AMINO20", &AMINO20},
    {"AMINO", &AMINO20},
    {"PROTEIN", &AMINO20},
    {"SEB8", &SEB8},
    {"SEB10", &SEB10},
    {"SEB14", &SEB14},
    {"SEV10", &SEV10},
    {"MURPHY", &MURPHY},
    {"LIA10", &LIA10},
    {"LIB10", &LIB10},
    {"SEB6", &SEB6},
    {"DAYHOFF", &DAYHOFF},

    {"DNAMETH", &DNA2METHYL},
    {"C", &DNA2METHYL},
    {"KETO", &DNA2KETAMINE},
    {"PURPYR", &DNA2PYRPUR},

    {"DNA4", &DNA4},
    {"DNA", &DNA4},

    {"DNA5", &DNA5}
};

} // alph

using alph::Alphabet;
using alph::BYTES;
using alph::AMINO20;
using alph::SEB14;
using alph::SEB10;
using alph::SEB6;
using alph::SEB8;
using alph::SEV10;
using alph::SOLISD;
using alph::SOLISG;
using alph::MURPHY;
using alph::LIA10;
using alph::LIB10;
using alph::DAYHOFF;
using alph::DNA5;
using alph::DNA4;
using alph::DNA2KETAMINE;
using alph::DNA2PYRPUR;


#endif /* BIOSEQ_ALPHBET_H__ */
