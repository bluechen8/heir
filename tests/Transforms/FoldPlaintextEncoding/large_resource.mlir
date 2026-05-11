// RUN: heir-opt --fold-plaintext-encoding %s | FileCheck %s

// Large heterogeneous plaintext: degree N=512 > the 256-element
// inline threshold, and the cleartext is non-constant (a
// `dense_resource<...>` blob with mixed bytes) so iFFT output is
// heterogeneous. The fold pass switches to `dense_resource<...>`
// storage for the encoded limbs to keep the IR text compact.
//
// We don't pin the encoded numeric values here — they're 512 i64s,
// not human-checkable. What matters is the *form* of the stamped
// attribute: `dense_resource<lwe_encoded_limbs_N>`. The
// `large_resource.mlir` fixture in
// tests/Dialect/LWE/Conversions/LWEToLinalg confirms the consumer
// side (radd_plain pattern) accepts resource-backed attributes.

!Zq_i64 = !mod_arith.int<36028797018652673 : i64>
!rns_L0 = !rns.rns<!Zq_i64>
#ring_f64 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**512>>
#ic = #lwe.inverse_canonical_encoding<scaling_factor = 30>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64, encoding = #ic>>

// CHECK-LABEL: func.func @encode_large
// CHECK:         lwe.rlwe_encode
// CHECK-SAME:    lwe.encoded_limbs = dense_resource<lwe_encoded_limbs_{{[0-9]+}}> : tensor<512x1xi64>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 9, Q = [36028797018652673], P = [1152921504606994433], logDefaultScale = 30>} {
  func.func @encode_large() -> !pt {
    %cst = arith.constant dense_resource<input_cleartext> : tensor<512xf32>
    %pt = lwe.rlwe_encode %cst {encoding = #ic, ring = #ring_f64} : tensor<512xf32> -> !pt
    return %pt : !pt
  }
}

{-#
  dialect_resources: {
    builtin: {
      // 512 f32 values: 1.0, 2.0, 3.0, ... encoded as little-endian bytes.
      // For the fold pass this just needs to be a non-trivial (non-zero
      // non-splat) blob; we use a simple "fill with 1.0 in first slot, 0 elsewhere"
      // pattern so the iFFT produces a non-splat output without needing
      // a 2KB byte literal.
      input_cleartext: "0x040000000000803f"
    }
  }
#-}
