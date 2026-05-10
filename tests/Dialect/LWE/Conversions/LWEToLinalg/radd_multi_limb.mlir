// RUN: heir-opt --lwe-to-linalg %s | FileCheck %s

// Two-limb RNS: verifies the moduli constant tensor is shape <2 x i64>
// and the limb-axis indexing map (d0, d1, d2) -> (d2) broadcasts the
// per-limb moduli into the linalg body.

!Zq1 = !mod_arith.int<36028797018652673 : i64>
!Zq2 = !mod_arith.int<288230376151760897 : i64>
!rns_L1 = !rns.rns<!Zq1, !Zq2>
#ring = #polynomial.ring<coefficientType = !rns_L1, polynomialModulus = <1 + x**16>>
#ciphertext_space = #lwe.ciphertext_space<ring = #ring, encryption_type = mix>
#plaintext_space = #lwe.plaintext_space<ring = #ring, encoding = #lwe.full_crt_packing_encoding<scaling_factor = 1>>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <36028797018652673 : i64, 288230376151760897 : i64>, current = 1>
!ct = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space, key = #key, modulus_chain = #modulus_chain>

// CHECK-LABEL: func @radd_two_limbs
// CHECK-SAME:    (%[[A:.*]]: tensor<2x16x2xi64>, %[[B:.*]]: tensor<2x16x2xi64>) -> tensor<2x16x2xi64>
// CHECK:         %[[Q:.*]] = arith.constant dense<[36028797018652673, 288230376151760897]> : tensor<2xi64>
// CHECK:         linalg.generic
// CHECK-SAME:      ins(%[[A]], %[[B]], %[[Q]]
func.func @radd_two_limbs(%a: !ct, %b: !ct) -> !ct {
  %r = lwe.radd %a, %b : (!ct, !ct) -> !ct
  return %r : !ct
}
