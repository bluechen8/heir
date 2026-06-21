// RUN: heir-opt --fold-plaintext-encoding --lwe-to-linalg --canonicalize %s | FileCheck %s

// Single-limb radd_plain. The fold pass stamps the encode op with
// `lwe.encoded_limbs`; the LWEToLinalg pattern reads the limbs as an
// `arith.constant`, takes ct[0,:,:], adds the plaintext per limb with
// lazy mod-reduce, and writes back to ct[0,:,:].

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

// CHECK-LABEL: func.func @radd_plain
// CHECK-SAME:    %[[CT:.*]]: tensor<2x4x1xi64>
// Splat slot vector (1.0, 1.0, 1.0, 1.0) → constant polynomial p(X)=1,
// whose negacyclic NTT yields the splat evaluation tensor
// (1, 1, 1, 1). With Δ = 16 the per-(i,j) limb residue is 16, tiled
// across the degree axis — collapses to `dense<16>`.
// CHECK-DAG:     %[[PT:.*]] = arith.constant dense<16> : tensor<4x1xi64>
// CHECK-DAG:     %[[Q:.*]] = arith.constant dense<97> : tensor<1xi64>
// CHECK:         %[[CT0:.*]] = tensor.extract_slice %[[CT]][0, 0, 0] [1, 4, 1] [1, 1, 1]
// CHECK:         %[[SUM:.*]] = linalg.generic
// CHECK-SAME:      ins(%[[CT0]], %[[PT]], %[[Q]]
// CHECK:         %[[OUT:.*]] = tensor.insert_slice %[[SUM]] into %[[CT]][0, 0, 0]
// CHECK:         return %[[OUT]]
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 2, Q = [97], P = [193], logDefaultScale = 4>} {
  func.func @radd_plain(%ct: !ct) -> !ct {
    %cst = arith.constant dense<1.000000e+00> : tensor<4xf64>
    %pt = lwe.rlwe_encode %cst {encoding = #ic, ring = #ring_f64} : tensor<4xf64> -> !pt
    %r = lwe.radd_plain %ct, %pt : (!ct, !pt) -> !ct
    return %r : !ct
  }
}
