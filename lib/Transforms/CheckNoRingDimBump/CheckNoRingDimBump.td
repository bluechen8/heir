#ifndef LIB_TRANSFORMS_CHECKNORINGDIMBUMP_CHECKNORINGDIMBUMP_TD_
#define LIB_TRANSFORMS_CHECKNORINGDIMBUMP_CHECKNORINGDIMBUMP_TD_

include "mlir/Pass/PassBase.td"

def CheckNoRingDimBump : Pass<"check-no-ringdim-bump", "mlir::ModuleOp"> {
  let summary = "Hard-fail if generate-param-ckks bumped the slot count above the user's --ciphertext-degree request.";
  let description = [{
    Runs after `--generate-param-ckks` (which writes both
    `scheme.requested_slot_count` and `scheme.actual_slot_count` on
    the module). Asserts:

      `1 << ckks.schemeParam.logN == 2 × scheme.actual_slot_count`
      `scheme.actual_slot_count == scheme.requested_slot_count`

    Either equality can fail when CKKS param-gen bumps the slot
    count above the user's `--ciphertext-degree` request — typically
    when bootstrap is required (Lattigo forces logN ≥ 14, so
    slotNumber ≥ 8192) or when the modulus chain pushes the total
    `logPQ` past the security table's `logMaxQ` at the structural
    minimum ringDim.

    Rotom emits slot-packing layouts and rotation indices against
    the user-supplied slot count; if the CKKS scheme then operates
    on a larger slot count, rotations wrap at the wrong modulus and
    decrypt to wrong values silently. Failing here forces the caller
    to re-run the front-end (Rotom + the HEIR pipeline) with a
    `--ciphertext-degree` value that won't trigger the bump.

    Diagnostic includes the requested vs. actual slot count and the
    smallest `--ciphertext-degree` that would clear the bump.

    No-op when the module has no `ckks.schemeParam` (BGV/BFV
    pipelines, or pre-param-gen IR).
  }];
  let dependentDialects = [];
}

#endif  // LIB_TRANSFORMS_CHECKNORINGDIMBUMP_CHECKNORINGDIMBUMP_TD_
