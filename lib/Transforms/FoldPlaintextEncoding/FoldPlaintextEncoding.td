#ifndef LIB_TRANSFORMS_FOLDPLAINTEXTENCODING_FOLDPLAINTEXTENCODING_TD_
#define LIB_TRANSFORMS_FOLDPLAINTEXTENCODING_FOLDPLAINTEXTENCODING_TD_

include "mlir/Pass/PassBase.td"

def FoldPlaintextEncoding : Pass<"fold-plaintext-encoding", "mlir::ModuleOp"> {
  let summary = "Compile-time fold of lwe.rlwe_encode on constant operands.";
  let description = [{
    Runs after `--ckks-to-lwe`, before `--lwe-to-linalg`. For each
    `lwe.rlwe_encode` whose input traces back to an `arith.constant`
    (inline or `dense_resource<KEY>`), evaluates the CKKS
    InverseCanonicalEmbedding encoder in C++ — iFFT, scale by
    Δ = 2^logDefaultScale, round, reduce mod q_i per RNS limb — and
    stamps the result on the op as a discardable `lwe.encoded_limbs`
    attribute of type `tensor<degree x num_limbs x iK>`. The op's SSA
    shape is unchanged; downstream `--lwe-to-linalg` patterns for
    `lwe.radd_plain` / `lwe.rsub_plain` read the attribute and
    materialize an `arith.constant` to add/sub against the ciphertext
    limbs.

    Scheme parameters (logN, Q, logDefaultScale) are read from the
    module-level `ckks.schemeParam` attribute set by
    `--generate-param-ckks`.

    If the operand doesn't trace to a constant the pass emits a
    diagnostic and leaves the op alone. No-op when the module has no
    `ckks.schemeParam` or no `lwe.rlwe_encode` ops.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::heir::lwe::LWEDialect",
  ];
}

#endif  // LIB_TRANSFORMS_FOLDPLAINTEXTENCODING_FOLDPLAINTEXTENCODING_TD_
