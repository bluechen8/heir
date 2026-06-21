#ifndef LIB_TRANSFORMS_FOLDPLAINTEXTENCODING_NTT_H_
#define LIB_TRANSFORMS_FOLDPLAINTEXTENCODING_NTT_H_

#include <cstdint>

#include "llvm/include/llvm/ADT/ArrayRef.h"    // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project

// Negacyclic NTT used by --fold-plaintext-encoding. We're in the ring
// ℤ_q[X] / (X^N + 1), so the NTT evaluates at the N odd-power 2N-th
// roots of unity ψ, ψ^3, …, ψ^{2N-1} where ψ^{2N} ≡ 1 mod q and
// ψ^N ≡ -1 mod q. Output ordering and modular representation match
// Lattigo v6.1.0's `ring.NTTStandard` (bit-reversed natural order,
// standard residue form [0, q)) so a downstream Lattigo FFI bridge is
// a flat memcpy.
//
// All entry points are pure C++ — no MLIR deps — so the NTT can be
// unit-tested in isolation and shares its source semantics with a
// numpy reference port (see validation/cross_oracle.py, added in
// Phase 2.2.2).

namespace mlir::heir::fold_plaintext_encoding {

// Precomputed reciprocal `floor(2^128 / q)` split into (hi, lo).
struct BRedConstants {
  uint64_t hi;
  uint64_t lo;
};

// Compute the Barrett reciprocal for `q` using __uint128_t internally.
// `q` must be > 1 and fit in 63 bits (suffices for CKKS limb primes).
BRedConstants computeBRed(uint64_t q);

// 64×64 → 128-bit unsigned multiply. Splits the product into a (hi, lo)
// pair so the public API stays portable; the implementation may use
// `__uint128_t` internally.
void mul64(uint64_t a, uint64_t b, uint64_t& hi, uint64_t& lo);

// Barrett reduction of the 128-bit value (xHi, xLo) mod q using a
// previously computed `bredConst`. Returns a value in [0, q).
uint64_t bred(uint64_t xHi, uint64_t xLo, uint64_t q, BRedConstants bredConst);

// Modular multiply `a · b mod q`. Both operands must already be in
// [0, q).
uint64_t mulmod(uint64_t a, uint64_t b, uint64_t q, BRedConstants bredConst);

// Derive a primitive 2N-th root of unity `ψ` mod `q`. Precondition:
// 2N must divide `q - 1` (i.e. `q` is NTT-friendly for ring degree N).
// Returns 0 if no such root exists (callers should check this with
// an explicit divisibility test before invoking).
//
// Implementation: trial-divides q-1 to factor it, then scans candidate
// generators g = 2, 3, 5, … until one satisfies `g^((q-1)/p) ≢ 1` for
// every prime divisor p of q-1 (i.e. g is a generator of (ℤ/q)*).
// Then ψ = g^((q-1) / (2N)) mod q.
uint64_t findPrimitive2NthRoot(uint64_t q, int64_t N);

// Negacyclic NTT of `coeffs` (length N) mod q. Writes the N evaluation
// points to `evals` in bit-reversed natural order: evals[BR(k)] is
// the value at ψ^(2k+1) for k ∈ [0, N). `psiPowers` must hold
// ψ^m mod q for m ∈ [0, 2N).
//
// Implementation is a direct O(N²) DFT — adequate for compile-time
// encoding of small constants (biases, layer scales). Production NTT
// runs in the runtime path and uses a butterfly (Phase 3).
void nttPerLimb(llvm::ArrayRef<uint64_t> coeffs, int64_t N, uint64_t q,
                BRedConstants bredConst, llvm::ArrayRef<uint64_t> psiPowers,
                llvm::SmallVectorImpl<uint64_t>& evals);

}  // namespace mlir::heir::fold_plaintext_encoding

#endif  // LIB_TRANSFORMS_FOLDPLAINTEXTENCODING_NTT_H_
