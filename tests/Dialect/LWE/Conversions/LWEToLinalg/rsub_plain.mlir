// RUN: heir-opt --fold-plaintext-encoding --lwe-to-linalg --canonicalize %s | FileCheck %s

// rsub_plain: ct[0] gets (ct[0] - pt) computed as (ct[0] + (q - pt)) mod q
// via lazy-reduce. ct[1] passes through.

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

// CHECK-LABEL: func.func @rsub_plain
// CHECK:         %[[CT0:.*]] = tensor.extract_slice
// CHECK:         linalg.generic
// CHECK:           %[[QMP:.*]] = arith.subi %{{.*}}, %{{.*}} : i64
// CHECK:           %[[S:.*]] = arith.addi %{{.*}}, %[[QMP]] : i64
// CHECK:         tensor.insert_slice
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 2, Q = [97], P = [193], logDefaultScale = 4>} {
  func.func @rsub_plain(%ct: !ct) -> !ct {
    %cst = arith.constant dense<1.000000e+00> : tensor<4xf64>
    %pt = lwe.rlwe_encode %cst {encoding = #ic, ring = #ring_f64} : tensor<4xf64> -> !pt
    %r = lwe.rsub_plain %ct, %pt : (!ct, !pt) -> !ct
    return %r : !ct
  }
}
