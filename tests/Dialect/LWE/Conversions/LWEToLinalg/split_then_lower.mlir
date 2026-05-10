// RUN: heir-opt --split-client-interface --lwe-to-linalg %s | FileCheck %s

// Composition smoke: exercises the A1 server-side path
// (split → lwe-to-linalg) on a simple ciphertext-only `radd` server
// with peeled encrypt/decrypt helpers. Validates that lwe-to-linalg
// only touches the server module and that the client module is
// preserved with its `lwe.rlwe_*` ops intact for downstream
// host codegen via `--lwe-to-{openfhe,lattigo}`.

!Zq_i64 = !mod_arith.int<36028797018652673 : i64>
!rns_L0 = !rns.rns<!Zq_i64>
#ring = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**16>>
#ciphertext_space = #lwe.ciphertext_space<ring = #ring, encryption_type = mix>
#plaintext_space = #lwe.plaintext_space<ring = #ring, encoding = #lwe.full_crt_packing_encoding<scaling_factor = 1>>
#key = #lwe.key<>
#modulus_chain = #lwe.modulus_chain<elements = <36028797018652673 : i64>, current = 0>
!ct = !lwe.lwe_ciphertext<plaintext_space = #plaintext_space, ciphertext_space = #ciphertext_space, key = #key, modulus_chain = #modulus_chain>
!pt = !lwe.lwe_plaintext<plaintext_space = #plaintext_space>
!sk = !lwe.lwe_secret_key<key = #key, ring = #ring>
!pk = !lwe.lwe_public_key<key = #key, ring = #ring>

// CHECK: module attributes {heir.client_module}
// Client side untouched — `lwe.rlwe_*` ops still present:
// CHECK:   func.func @forward__encrypt__arg0
// CHECK:     lwe.rlwe_encrypt
// CHECK:   func.func @forward__decrypt__result0
// CHECK:     lwe.rlwe_decrypt
// CHECK: module attributes {heir.server_module}
// Server `forward` got fully lowered to linalg.generic (no lwe.* ops):
// CHECK:   func.func @forward
// CHECK:     linalg.generic
// CHECK-NOT: lwe.radd

module {
  func.func @forward(%a: !ct, %b: !ct) -> !ct {
    %r = lwe.radd %a, %b : (!ct, !ct) -> !ct
    return %r : !ct
  }
  func.func @forward__encrypt__arg0(%pt: !pt, %pk: !pk) -> !ct
      attributes {client.enc_func = {func_name = "forward", index = 0 : i64}} {
    %ct = lwe.rlwe_encrypt %pt, %pk : (!pt, !pk) -> !ct
    return %ct : !ct
  }
  func.func @forward__decrypt__result0(%ct: !ct, %sk: !sk) -> !pt
      attributes {client.dec_func = {func_name = "forward", index = 0 : i64}} {
    %pt = lwe.rlwe_decrypt %ct, %sk : (!ct, !sk) -> !pt
    return %pt : !pt
  }
}
