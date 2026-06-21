#include "lib/Transforms/FoldPlaintextEncoding/Ntt.h"

#include <cassert>
#include <cstdint>

#include "llvm/include/llvm/ADT/ArrayRef.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"  // from @llvm-project

namespace mlir::heir::fold_plaintext_encoding {

namespace {

// Modular exponentiation `base^exp mod q` using the (mul64, bred) pair
// rather than calling mulmod so we don't reload bredConst every step.
uint64_t powmodInternal(uint64_t base, uint64_t exp, uint64_t q,
                        BRedConstants bredConst) {
  uint64_t result = 1 % q;
  uint64_t b = base % q;
  while (exp > 0) {
    if (exp & 1) {
      uint64_t hi, lo;
      mul64(result, b, hi, lo);
      result = bred(hi, lo, q, bredConst);
    }
    exp >>= 1;
    if (exp == 0) break;
    uint64_t hi, lo;
    mul64(b, b, hi, lo);
    b = bred(hi, lo, q, bredConst);
  }
  return result;
}

// Trial-divide `n` into a small list of distinct prime factors. The
// caller passes `n = q - 1`; for the limb primes we care about this is
// at most ~63 bits and has few large factors (it's always 2^k · m
// with small m), so trial division up to √n is fine. Returns the
// distinct prime divisors in ascending order.
void distinctPrimeFactors(uint64_t n, llvm::SmallVectorImpl<uint64_t>& out) {
  out.clear();
  if (n == 0) return;
  if (n % 2 == 0) {
    out.push_back(2);
    while (n % 2 == 0) n /= 2;
  }
  for (uint64_t p = 3; p * p <= n; p += 2) {
    if (n % p == 0) {
      out.push_back(p);
      while (n % p == 0) n /= p;
    }
  }
  if (n > 1) out.push_back(n);
}

// Reverse the low `logN` bits of `i`. Used to permute NTT output into
// Lattigo's bit-reversed natural order.
uint64_t bitReverse(uint64_t i, int logN) {
  uint64_t r = 0;
  for (int b = 0; b < logN; ++b) {
    r = (r << 1) | (i & 1);
    i >>= 1;
  }
  return r;
}

}  // namespace

BRedConstants computeBRed(uint64_t q) {
  assert(q > 1 && "modulus must be > 1");
  // floor(2^128 / q). __uint128_t division is well-supported on the
  // GCC/Clang targets HEIR builds against.
  __uint128_t numerator = ~__uint128_t{0};   // 2^128 - 1
  numerator = numerator / static_cast<__uint128_t>(q);
  // For q a prime well under 2^63 this differs from floor(2^128 / q)
  // by at most 1 in the LSB and only when 2^128 mod q == 0, which can
  // only happen if q divides 2^128 — i.e. q is a power of two. Limb
  // primes are odd primes, so we're safe.
  BRedConstants bc;
  bc.lo = static_cast<uint64_t>(numerator);
  bc.hi = static_cast<uint64_t>(numerator >> 64);
  return bc;
}

void mul64(uint64_t a, uint64_t b, uint64_t& hi, uint64_t& lo) {
  __uint128_t prod = static_cast<__uint128_t>(a) * static_cast<__uint128_t>(b);
  lo = static_cast<uint64_t>(prod);
  hi = static_cast<uint64_t>(prod >> 64);
}

uint64_t bred(uint64_t xHi, uint64_t xLo, uint64_t q, BRedConstants bredConst) {
  // Compute floor(x · m / 2^128) where m = bredConst (≈ 2^128 / q),
  // then subtract that estimate · q from x. The result is in
  // [0, 2q); one conditional subtract refines it to [0, q).
  __uint128_t x = (static_cast<__uint128_t>(xHi) << 64) | xLo;

  // High 128 bits of the full 256-bit product x · m.
  __uint128_t xLo128 = xLo;
  __uint128_t xHi128 = xHi;
  __uint128_t mLo128 = bredConst.lo;
  __uint128_t mHi128 = bredConst.hi;
  __uint128_t ll = xLo128 * mLo128;
  __uint128_t lh = xLo128 * mHi128;
  __uint128_t hl = xHi128 * mLo128;
  __uint128_t hh = xHi128 * mHi128;
  __uint128_t mid = (ll >> 64) + (lh & ~uint64_t{0}) + (hl & ~uint64_t{0});
  __uint128_t qEst = hh + (lh >> 64) + (hl >> 64) + (mid >> 64);

  // r = x - qEst · q (mod 2^128). All upper bits cancel; the low 128
  // bits are what matter.
  __uint128_t r = x - qEst * static_cast<__uint128_t>(q);
  uint64_t r64 = static_cast<uint64_t>(r);
  // Two conditional subtracts: the Barrett estimate can be off by at
  // most 2 when m is the truncated reciprocal. The compiler folds the
  // pair into a couple of cmovs.
  if (r64 >= q) r64 -= q;
  if (r64 >= q) r64 -= q;
  return r64;
}

uint64_t mulmod(uint64_t a, uint64_t b, uint64_t q, BRedConstants bredConst) {
  uint64_t hi, lo;
  mul64(a, b, hi, lo);
  return bred(hi, lo, q, bredConst);
}

uint64_t findPrimitive2NthRoot(uint64_t q, int64_t N) {
  assert(N > 0 && "ring degree must be positive");
  uint64_t twoN = static_cast<uint64_t>(2 * N);
  if (q < 2) return 0;
  if ((q - 1) % twoN != 0) return 0;

  BRedConstants bc = computeBRed(q);
  llvm::SmallVector<uint64_t, 8> primes;
  distinctPrimeFactors(q - 1, primes);

  uint64_t exponent = (q - 1) / twoN;

  // Match Lattigo's `ring.PrimitiveRoot` (`subring.go:174-188`): the
  // search increments `g` BEFORE checking, so it effectively starts at
  // g = 3. For most CKKS limb primes this picks the same generator as
  // a g = 2 start (the standard CKKS primes have 2 as a quadratic
  // residue, so 2 isn't a generator anyway), but starting at 3 is what
  // makes ψ bit-identical to Lattigo on the rare primes where 2 is a
  // primitive root.
  for (uint64_t g = 3; g < q; ++g) {
    bool isGenerator = true;
    for (uint64_t p : primes) {
      uint64_t e = (q - 1) / p;
      if (powmodInternal(g, e, q, bc) == 1) {
        isGenerator = false;
        break;
      }
    }
    if (!isGenerator) continue;
    uint64_t psi = powmodInternal(g, exponent, q, bc);
    // Sanity: ψ^N == q - 1 (i.e. -1) and ψ^(2N) == 1.
    if (powmodInternal(psi, static_cast<uint64_t>(N), q, bc) != q - 1) continue;
    return psi;
  }
  return 0;
}

void nttPerLimb(llvm::ArrayRef<uint64_t> coeffs, int64_t N, uint64_t q,
                BRedConstants bredConst, llvm::ArrayRef<uint64_t> psiPowers,
                llvm::SmallVectorImpl<uint64_t>& evals) {
  assert(static_cast<int64_t>(coeffs.size()) == N);
  assert(static_cast<int64_t>(psiPowers.size()) == 2 * N);
  assert(N > 0 && (N & (N - 1)) == 0 && "N must be a power of two");
  uint64_t twoN = static_cast<uint64_t>(2 * N);
  int logN = llvm::Log2_64(static_cast<uint64_t>(N));

  evals.assign(N, 0);
  for (int64_t k = 0; k < N; ++k) {
    // Eval at X = ψ^(2k+1): p(X) = Σ_j coeffs[j] · X^j
    //                            = Σ_j coeffs[j] · ψ^((2k+1)·j).
    uint64_t sum = 0;
    uint64_t exponentStep = static_cast<uint64_t>(2 * k + 1);
    for (int64_t j = 0; j < N; ++j) {
      uint64_t m = (exponentStep * static_cast<uint64_t>(j)) % twoN;
      uint64_t term = mulmod(coeffs[j], psiPowers[m], q, bredConst);
      sum += term;
      if (sum >= q) sum -= q;
    }
    // Bit-reverse the output index so the result lands in Lattigo's
    // `ring.NTTStandard` layout (`ring.Poly.Coeffs[lvl]`).
    uint64_t out = bitReverse(static_cast<uint64_t>(k), logN);
    evals[out] = sum;
  }
}

}  // namespace mlir::heir::fold_plaintext_encoding
