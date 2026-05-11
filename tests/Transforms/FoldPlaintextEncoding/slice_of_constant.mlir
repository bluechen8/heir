// RUN: heir-opt --fold-plaintext-encoding %s | FileCheck %s

// `tensor.extract_slice` of a splat constant collapses to a smaller
// splat of the same value. The fold pass traces through this.

#ring_f64 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**4>>
#ic = #lwe.inverse_canonical_encoding<scaling_factor = 4>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64, encoding = #ic>>

// CHECK-LABEL: func.func @encode_sliced_zero
// CHECK:         lwe.rlwe_encode
// CHECK-SAME:    lwe.encoded_limbs = dense<0> : tensor<4x1xi64>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 2, Q = [97], P = [193], logDefaultScale = 4>} {
  func.func @encode_sliced_zero() -> !pt {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x4xf64>
    %slice = tensor.extract_slice %cst[0, 0] [1, 4] [1, 1] : tensor<1x4xf64> to tensor<4xf64>
    %pt = lwe.rlwe_encode %slice {encoding = #ic, ring = #ring_f64} : tensor<4xf64> -> !pt
    return %pt : !pt
  }
}
