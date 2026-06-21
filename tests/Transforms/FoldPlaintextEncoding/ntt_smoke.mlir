// RUN: heir-opt --fold-plaintext-encoding %s | FileCheck %s

// Production-shape smoke test for the encoder's slot-permuted
// `SpecialIFFTDouble` + negacyclic-NTT pipeline. N=16 (so the
// bit-reversal pattern inside SpecialIFFTDouble is visibly non-
// trivial — the N=4 fixtures don't distinguish bit-reversed from
// natural order), single 32-bit NTT-friendly prime q = 65537. That's
// the Fermat prime 2^16 + 1, with q-1 = 2^16, so 2N=32 trivially
// divides q-1. Smallest generator of (ℤ/65537)* found by the
// Lattigo-compatible search (start at g=3) is g=3 (g=2 has
// order 32, since 2^16 ≡ -1 (mod 65537) → 2^32 ≡ 1); ψ =
// 3^((q-1)/2N) = 3^2048 ≡ 65529 (mod 65537).
//
// Slot vector z = (1, -2, 3, -4, 5, -6, 7, -8): 8 = N/2 slots,
// asymmetric on purpose so the post-NTT signature isn't a fluke
// of conjugate symmetry.
//
// Pipeline values (Δ = 2^4 = 16; cross-checked against
// validation/cross_oracle.py::ckks_inverse_canonical_encode and
// Lattigo `ckks.SpecialIFFTDouble`):
//
//   SpecialIFFTDouble(z, halfN=8) (8 complex iFFT outputs); the
//   polynomial coefficients are coeffs[i] = real(iFFT[i]),
//   coeffs[i+8] = imag(iFFT[i]). Scaled · 16:
//     [-8,  2, -10, -12, 51, -17,  4,  8,
//       0, -8,  -4,  17, -51, 12, 10, -2]
//
//   per-coefficient residue mod 65537 (pre-NTT, coefficient form):
//     [65529,     2, 65527, 65525,    51, 65520,     4,     8,
//          0, 65529, 65533,    17, 65486,    12,    10, 65535]
//
//   post-NTT bit-reversed natural order
//   (evals[BR(k)] = p(ψ^(2k+1)) for k ∈ [0, 16)):
//     [58618, 49007, 64824,  4469, 15748, 19070, 42100,  8248,
//       8248, 42100, 19070, 15748,  4469, 64824, 49007, 58618]
//
//   The post-NTT palindrome is the real-input signature: real slot
//   values produce real polynomial coefficients, which negacyclic-
//   NTT to a conjugate-symmetric eval tensor (residues paired via
//   ψ^(2k+1) and ψ^{-(2k+1)} after bit-reversal).

!Zq = !mod_arith.int<65537 : i64>
!rns_L0 = !rns.rns<!Zq>
#ring_rns = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**16>>
#ring_f64 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**16>>
#ic = #lwe.inverse_canonical_encoding<scaling_factor = 4>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64, encoding = #ic>>

// CHECK-LABEL: func.func @encode_smoke_n16
// CHECK:         lwe.rlwe_encode
// CHECK-SAME:    lwe.encoded_limbs = dense<{{\[}}[58618], [49007], [64824], [4469], [15748], [19070], [42100], [8248], [8248], [42100], [19070], [15748], [4469], [64824], [49007], [58618]]> : tensor<16x1xi64>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 4, Q = [65537], P = [786433], logDefaultScale = 4>} {
  func.func @encode_smoke_n16() -> !pt {
    %cst = arith.constant dense<[1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]> : tensor<8xf64>
    %pt = lwe.rlwe_encode %cst {encoding = #ic, ring = #ring_f64} : tensor<8xf64> -> !pt
    return %pt : !pt
  }
}
