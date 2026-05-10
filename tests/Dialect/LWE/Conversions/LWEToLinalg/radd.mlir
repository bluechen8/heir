// RUN: heir-opt --lwe-to-linalg %s | FileCheck %s

!Zq_i64 = !mod_arith.int<36028797018652673 : i64>
!rns_L0 = !rns.rns<!Zq_i64>
#ring = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**16>>
#ciphertext_space = #lwe.ciphertext_space<ring = #ring, encryption_type = mix>
#plaintext_space = #lwe.plaintext_space<ring = #ring, encoding = #lwe.full_crt_packing_encoding<scaling_factor = 1>>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <36028797018652673 : i64>, current = 0>
!ct = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space, key = #key, modulus_chain = #modulus_chain>

// CHECK-LABEL: func @radd_single_limb
// CHECK-SAME:    (%[[A:.*]]: tensor<2x16x1xi64>, %[[B:.*]]: tensor<2x16x1xi64>) -> tensor<2x16x1xi64>
// CHECK:         %[[Q:.*]] = arith.constant dense<36028797018652673> : tensor<1xi64>
// CHECK:         %[[EMPTY:.*]] = tensor.empty() : tensor<2x16x1xi64>
// CHECK:         %[[R:.*]] = linalg.generic
// CHECK-SAME:      ins(%[[A]], %[[B]], %[[Q]]
// CHECK-SAME:      outs(%[[EMPTY]]
// CHECK:         ^bb0(%[[AE:.*]]: i64, %[[BE:.*]]: i64, %[[QE:.*]]: i64, %{{.*}}: i64):
// CHECK:           %[[SUM:.*]] = arith.addi %[[AE]], %[[BE]] : i64
// CHECK:           %[[CMP:.*]] = arith.cmpi uge, %[[SUM]], %[[QE]] : i64
// CHECK:           %[[SUB:.*]] = arith.subi %[[SUM]], %[[QE]] : i64
// CHECK:           %[[SEL:.*]] = arith.select %[[CMP]], %[[SUB]], %[[SUM]] : i64
// CHECK:           linalg.yield %[[SEL]] : i64
// CHECK:         return %[[R]] : tensor<2x16x1xi64>
func.func @radd_single_limb(%a: !ct, %b: !ct) -> !ct {
  %r = lwe.radd %a, %b : (!ct, !ct) -> !ct
  return %r : !ct
}
