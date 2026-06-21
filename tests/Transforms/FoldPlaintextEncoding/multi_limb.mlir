// RUN: heir-opt --fold-plaintext-encoding %s | FileCheck %s

// Two-limb RNS, splat (constant) cleartext exercises the splat fast
// path. For ring degree N=4 the canonical embedding has N/2 = 2 slots;
// we feed slots z = (1+0i, 1+0i). The polynomial whose canonical
// embedding is the constant slot vector is the constant polynomial
// p(x) = c, and the negacyclic NTT of a constant polynomial is the
// constant tensor (c, c, c, c) at every evaluation point.
//
// So after scale-round-reduce we get per-limb residue
//   c · Δ mod q_j = 1 · 16 mod q_j = 16  (both limbs, since 16 < 97 < 193).
// Tiled across the degree axis the result is dense<16> : tensor<4x2xi64>.
//
// `--fold-plaintext-encoding` detects the splat slot vector and skips
// both the O(N²) iFFT and the O(N²) NTT for this case.

!Zq0 = !mod_arith.int<97 : i64>
!Zq1 = !mod_arith.int<193 : i64>
!rns_L1 = !rns.rns<!Zq0, !Zq1>
#ring_rns = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**4>>
#ring_f64 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**4>>
#ic = #lwe.inverse_canonical_encoding<scaling_factor = 4>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64, encoding = #ic>>

// CHECK-LABEL: func.func @encode_ones
// CHECK:         lwe.rlwe_encode
// CHECK-SAME:    lwe.encoded_limbs = dense<16> : tensor<4x2xi64>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 2, Q = [97, 193], P = [257], logDefaultScale = 4>} {
  func.func @encode_ones() -> !pt {
    %cst = arith.constant dense<1.000000e+00> : tensor<4xf64>
    %pt = lwe.rlwe_encode %cst {encoding = #ic, ring = #ring_f64} : tensor<4xf64> -> !pt
    return %pt : !pt
  }
}
