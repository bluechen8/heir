// RUN: heir-opt --fold-plaintext-encoding %s | FileCheck %s

// AddModel-shaped fixture: bias = zeros, single RNS limb, single
// polynomial. The fold stamps the encode op with an
// `lwe.encoded_limbs` attribute containing the pre-encoded RNS limbs
// (tensor<degree x numLimbs x iK>). SSA shape is unchanged so the
// op's ElementwiseByOperandOpInterface verifier still accepts it.

!Zq_i64 = !mod_arith.int<36028797018652673 : i64>
!rns_L0 = !rns.rns<!Zq_i64>
#ring_rns = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**16>>
#ring_f64 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**16>>
#ic = #lwe.inverse_canonical_encoding<scaling_factor = 45>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64, encoding = #ic>>

// CHECK-LABEL: func.func @encode_zeros
// CHECK:         %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<16xf32>
// CHECK:         lwe.rlwe_encode %[[CST]]
// CHECK-SAME:    lwe.encoded_limbs = dense<0> : tensor<16x1xi64>
// CHECK-SAME:    tensor<16xf32> -> [[PT:.*]]
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673], P = [1152921504606994433], logDefaultScale = 45>} {
  func.func @encode_zeros() -> !pt {
    %cst = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %pt = lwe.rlwe_encode %cst {encoding = #ic, ring = #ring_f64} : tensor<16xf32> -> !pt
    return %pt : !pt
  }
}
