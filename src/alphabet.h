#ifndef BIOSEQ_ALPHBET_H__
#define BIOSEQ_ALPHBET_H__
// The alphabet encoder (biological characters -> tokens) is bonsai's, used directly from the bonsai submodule
// (branch maint) so that bioseq inherits its fixes instead of carrying a diverging copy. Fixes this picked up:
// aliases take effect (U reads as T in the DNA alphabets, O as K and U as C in the protein alphabets; bonsai
// 665d78a) and lookups index with unsigned bytes (0x80-0xff are invalid, not out-of-table reads; ad105ec).
// Regression test: src/test_alphabet.cpp.
#include "bonsai/alphabet.h"
#include <string_view>

namespace alph {
using namespace bns::alph;

// bioseq-only additions (amino-acid property sets; not part of bonsai)
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

} // namespace alph

using bns::alph::Alphabet;
using bns::alph::BYTES;
using bns::alph::AMINO20;
using bns::alph::SEB14;
using bns::alph::SEB10;
using bns::alph::SEB6;
using bns::alph::SEB8;
using bns::alph::SEV10;
using bns::alph::SOLISD;
using bns::alph::SOLISG;
using bns::alph::MURPHY;
using bns::alph::LIA10;
using bns::alph::LIB10;
using bns::alph::DAYHOFF;
using bns::alph::DNA5;
using bns::alph::DNA4;
using bns::alph::DNA2KETAMINE;
using bns::alph::DNA2PYRPUR;
using bns::alph::CAMAP;

#endif /* BIOSEQ_ALPHBET_H__ */
