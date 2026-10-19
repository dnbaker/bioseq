// Regression tests for the alphabet encoder (src/alphabet.h), mirroring bonsai's
// test/regress_alphabet_aliases.cpp and test/regress_high_bytes.cpp (fixes 665d78a, ad105ec).
// Build: g++ -std=c++17 -O1 src/test_alphabet.cpp -o test_alphabet && ./test_alphabet
#include "alphabet.h"
#include <cstdio>

static int failures = 0;
static void check(bool ok, const char *what) {
    std::printf("%-60s %s\n", what, ok ? "OK" : "FAIL");
    failures += !ok;
}

int main() {
    // Aliases: U reads as T in the DNA alphabets (RNA parses like DNA); O reads as K and U as C in protein alphabets.
    check(DNA4.translate('U') == DNA4.translate('T') && DNA4.translate('T') != -1, "DNA4: U == T");
    check(DNA4.translate('u') == DNA4.translate('t'), "DNA4: u == t");
    check(DNA5.translate('U') == DNA5.translate('T'), "DNA5: U == T");
    check(alph::DNA2PYRPUR.translate('U') == alph::DNA2PYRPUR.translate('T'), "DNA2 pur/pyr: U == T");
    check(AMINO20.translate('O') == AMINO20.translate('K') && AMINO20.translate('K') != -1, "AMINO20: O == K");
    check(AMINO20.translate('U') == AMINO20.translate('C'), "AMINO20: U == C");
    check(SEB10.translate('O') == SEB10.translate('K') && SEB10.translate('U') == SEB10.translate('C'), "SEB10: O == K, U == C");
    check(DAYHOFF.translate('u') == DAYHOFF.translate('c'), "DAYHOFF: u == c");
    // Aliases never override a letter the alphabet already encodes.
    check(AMINO20.translate('C') != AMINO20.translate('K'), "AMINO20: aliases leave encoded letters alone");
    // High bytes 0x80-0xff are invalid, never a base (signed-char indexing read memory before the table).
    bool hi_ok = true;
    for(int b = 0x80; b <= 0xff; ++b) {
        const char c = static_cast<char>(b);
        hi_ok &= DNA4.translate(c) == -1 && DNA5.translate(c) == -1 && AMINO20.translate(c) == -1;
    }
    check(hi_ok, "bytes 0x80-0xff translate to -1 (DNA4, DNA5, AMINO20)");
    std::printf("%s\n", failures ? "FAILED" : "ALL PASSED");
    return failures != 0;
}
