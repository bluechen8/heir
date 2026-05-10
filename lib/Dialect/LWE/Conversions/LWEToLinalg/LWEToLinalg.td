#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOLINALG_LWETOLINALG_TD_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOLINALG_LWETOLINALG_TD_

include "mlir/Pass/PassBase.td"

def LWEToLinalg : Pass<"lwe-to-linalg", "mlir::ModuleOp"> {
  let summary = "Lower `lwe` to `linalg` + `tensor` + `arith` for IREE.";

  let description = [{
    Forks the LWE-level lowering to portable `linalg` IR, bypassing the
    polynomial dialect. Each `lwe.*` op is rewritten as a `linalg.generic`
    over a flattened `tensor<num_polys x degree x num_limbs x iK>`
    representation; the RNS limb axis is materialized as a plain trailing
    tensor dim and per-limb moduli flow as `arith.constant` block-args.

    Phase 1 covers the trivial elementwise ops (radd, rsub, rnegate,
    radd_plain, rsub_plain). NTT-bearing ops (rmul, rrotate, ...) land in
    later phases.
  }];

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect",
    "mlir::heir::lwe::LWEDialect",
    "mlir::heir::polynomial::PolynomialDialect",
    "mlir::heir::rns::RNSDialect",
    "mlir::heir::mod_arith::ModArithDialect"
  ];
}

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOLINALG_LWETOLINALG_TD_
