// RUN: heir-opt --fold-plaintext-encoding %s | FileCheck %s

// Non-constant slot vector exercising the iFFT path with values that
// distinguish coefficient-encoding (which would yield (16, 32, 0, 0))
// from inverse-canonical-embedding (which yields the result below).
//
// Hand-computed reference (Δ = 2^4 = 16, q = 97):
//   slots z = (1+0i, 2+0i)                               (N/2 = 2)
//   conjugate-symmetric extension zFull = (1, 2, 2, 1)
//   p[j] = (1/4) Σ_k zFull[k] · exp(-iπ(2k+1)j/4)
//        =  (1.5, -0.3535…, 0.0, 0.3535…)
//   scaled = round(p · 16)         = (24, -6, 0, 6)
//   mod 97 (centered to [0, 97))   = (24, 91, 0, 6)
//
// Independently cross-checked against a numpy port; see commit log.

!Zq = !mod_arith.int<97 : i64>
!rns_L0 = !rns.rns<!Zq>
#ring_rns = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**4>>
#ring_f64 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**4>>
#ic = #lwe.inverse_canonical_encoding<scaling_factor = 4>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64, encoding = #ic>>

// CHECK-LABEL: func.func @encode_one_two
// CHECK:         lwe.rlwe_encode
// CHECK-SAME:    lwe.encoded_limbs = dense<{{\[}}[24], [91], [0], [6]]> : tensor<4x1xi64>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 2, Q = [97], P = [193], logDefaultScale = 4>} {
  func.func @encode_one_two() -> !pt {
    %cst = arith.constant dense<[1.0, 2.0, 0.0, 0.0]> : tensor<4xf64>
    %pt = lwe.rlwe_encode %cst {encoding = #ic, ring = #ring_f64} : tensor<4xf64> -> !pt
    return %pt : !pt
  }
}
