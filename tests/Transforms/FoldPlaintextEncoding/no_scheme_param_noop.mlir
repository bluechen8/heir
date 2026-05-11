// RUN: heir-opt --fold-plaintext-encoding %s | FileCheck %s

// Without a module-level ckks.schemeParam (BGV path, plaintext-only
// path, hand-crafted IR), --fold-plaintext-encoding must be a no-op:
// it cannot derive Δ or the Q array. Encode ops survive unchanged.

#ring_f64 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**4>>
#ic = #lwe.inverse_canonical_encoding<scaling_factor = 4>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64, encoding = #ic>>

// CHECK-LABEL: func.func @no_scheme
// CHECK:         lwe.rlwe_encode
// CHECK-NOT:     lwe.encoded_limbs
module {
  func.func @no_scheme() -> !pt {
    %cst = arith.constant dense<0.000000e+00> : tensor<4xf64>
    %pt = lwe.rlwe_encode %cst {encoding = #ic, ring = #ring_f64} : tensor<4xf64> -> !pt
    return %pt : !pt
  }
}
