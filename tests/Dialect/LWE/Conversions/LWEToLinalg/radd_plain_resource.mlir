// RUN: heir-opt --lwe-to-linalg --canonicalize %s | FileCheck %s

// Companion to radd_plain.mlir: the encode op's `lwe.encoded_limbs`
// attr is a `dense_resource<...>` instead of an inline `dense<...>`.
// The LWEToLinalg pattern accepts both (looks up via `ElementsAttr`,
// not `DenseElementsAttr`) and materializes an `arith.constant` of
// the same resource-backed attr — IREE then maps the blob like any
// other weight tensor without expanding into the IR text.
//
// The encoded_limbs are hand-stamped here (not the output of
// --fold-plaintext-encoding) so the fixture remains stable independent
// of any encoder changes. We use a 4×1 i64 tensor with heterogeneous
// values to keep the test compact.

!Zq_i64 = !mod_arith.int<97 : i64>
!rns_L0 = !rns.rns<!Zq_i64>
#ring_rns = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**4>>
#ring_f64 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**4>>
#ic = #lwe.inverse_canonical_encoding<scaling_factor = 4>
#cs = #lwe.ciphertext_space<ring = #ring_rns, encryption_type = mix>
#ps = #lwe.plaintext_space<ring = #ring_f64, encoding = #ic>
#key = #lwe.key<>
#mc = #lwe.modulus_chain<elements = <97 : i64>, current = 0>
!ct = !lwe.lwe_ciphertext<plaintext_space = #ps, ciphertext_space = #cs, key = #key, modulus_chain = #mc>
!pt = !lwe.lwe_plaintext<plaintext_space = #ps>

// CHECK-LABEL: func.func @radd_plain_resource
// CHECK:         arith.constant dense_resource<radd_plain_pt> : tensor<4x1xi64>
// CHECK:         tensor.extract_slice
// CHECK:         linalg.generic
// CHECK:         tensor.insert_slice
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 2, Q = [97], P = [193], logDefaultScale = 4>} {
  func.func @radd_plain_resource(%ct: !ct) -> !ct {
    %cst = arith.constant dense<1.0> : tensor<4xf64>
    %pt = lwe.rlwe_encode %cst {
        encoding = #ic,
        ring = #ring_f64,
        lwe.encoded_limbs = dense_resource<radd_plain_pt> : tensor<4x1xi64>
      } : tensor<4xf64> -> !pt
    %r = lwe.radd_plain %ct, %pt : (!ct, !pt) -> !ct
    return %r : !ct
  }
}

{-#
  dialect_resources: {
    builtin: {
      // 4 × i64 limbs: 16, 0, 0, 0 (matches the iFFT output for the
      // splat-1.0 input). 8-byte alignment header + 32 bytes of data.
      radd_plain_pt: "0x080000001000000000000000000000000000000000000000000000000000000000000000"
    }
  }
#-}
