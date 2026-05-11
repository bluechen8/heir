// RUN: heir-opt --fold-plaintext-encoding %s | FileCheck %s

// Two-limb RNS, non-zero cleartext exercises the iFFT path. For ring
// degree N=4 the canonical embedding has N/2 = 2 slots; we feed the
// first 2 values (slots z = (1+0i, 1+0i)). The polynomial whose
// canonical embedding is the constant vector (c, c, …) is the constant
// polynomial p(x) = c, so:
//   p = (1, 0, 0, 0)
//   scaled = (16, 0, 0, 0)        with Δ = 2^4 = 16
//   limbs  = ((16,16), (0,0), (0,0), (0,0))  per (q0=97, q1=193)
//
// Both limbs see the same value because 16 < 97 < 193 (no wrap on
// either modulus).

!Zq0 = !mod_arith.int<97 : i64>
!Zq1 = !mod_arith.int<193 : i64>
!rns_L1 = !rns.rns<!Zq0, !Zq1>
#ring_rns = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**4>>
#ring_f64 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**4>>
#ic = #lwe.inverse_canonical_encoding<scaling_factor = 4>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64, encoding = #ic>>

// CHECK-LABEL: func.func @encode_ones
// CHECK:         lwe.rlwe_encode
// CHECK-SAME:    lwe.encoded_limbs = dense<{{\[}}[16, 16], [0, 0], [0, 0], [0, 0]]> : tensor<4x2xi64>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 2, Q = [97, 193], P = [257], logDefaultScale = 4>} {
  func.func @encode_ones() -> !pt {
    %cst = arith.constant dense<1.000000e+00> : tensor<4xf64>
    %pt = lwe.rlwe_encode %cst {encoding = #ic, ring = #ring_f64} : tensor<4xf64> -> !pt
    return %pt : !pt
  }
}
