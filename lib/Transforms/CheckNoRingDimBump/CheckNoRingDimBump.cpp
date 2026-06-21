// CheckNoRingDimBump: assert that --generate-param-ckks didn't bump the
// slot count above the user's `--ciphertext-degree` request, and that
// the resulting `logN` matches the structural CKKS minimum
// `1 << logN == 2 × slot_count`.
//
// Rotom emits packing / rotation IR against the user-supplied slot
// count. If CKKS param-gen later picks a larger slot count (bootstrap
// floor, security floor), the IR still references the original slot
// count → rotations wrap at the wrong modulus and decrypts come back
// silently wrong. We fail loudly here so the caller can re-run with a
// `--ciphertext-degree` that won't get bumped.

#include "lib/Transforms/CheckNoRingDimBump/CheckNoRingDimBump.h"

#include <cstdint>

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"         // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_CHECKNORINGDIMBUMP
#include "lib/Transforms/CheckNoRingDimBump/CheckNoRingDimBump.h.inc"

namespace {

struct CheckNoRingDimBump
    : public impl::CheckNoRingDimBumpBase<CheckNoRingDimBump> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    auto schemeAttr = module->getAttrOfType<ckks::SchemeParamAttr>(
        ckks::CKKSDialect::kSchemeParamAttrName);
    if (!schemeAttr) {
      // No CKKS scheme param attached — not our concern (BGV/BFV
      // pipelines, or modules that haven't reached generate-param yet).
      return;
    }

    auto requestedAttr =
        module->getAttrOfType<IntegerAttr>(kRequestedSlotCountAttrName);
    auto actualAttr =
        module->getAttrOfType<IntegerAttr>(kActualSlotCountAttrName);
    if (!requestedAttr || !actualAttr) {
      module.emitError()
          << "check-no-ringdim-bump: missing `"
          << kRequestedSlotCountAttrName << "` or `"
          << kActualSlotCountAttrName
          << "` on the module; --generate-param-ckks must run before "
             "this pass.";
      signalPassFailure();
      return;
    }

    int64_t requested = requestedAttr.getInt();
    int64_t actual = actualAttr.getInt();
    int64_t logN = schemeAttr.getLogN();
    int64_t ringDim = int64_t{1} << logN;

    if (requested != actual) {
      // Param-gen bumped the slot count above the user's request
      // (e.g., bootstrap floor at slotNumber=8192 in
      // GenerateParamCKKS.cpp:155-158).
      module.emitError()
          << "check-no-ringdim-bump: --generate-param-ckks bumped the "
             "CKKS slot count from "
          << requested << " (the value passed via `--ciphertext-degree`) "
                          "to "
          << actual
          << ". Rotom emits packing / rotation IR against the original "
             "slot count, so the resulting circuit would have wrong "
             "rotation semantics. Re-run the front-end pipeline with "
             "`--ciphertext-degree="
          << actual << "` (or higher) so Rotom and CKKS agree.";
      signalPassFailure();
      return;
    }

    if (ringDim != 2 * actual) {
      // Security-driven ringDim bump above the structural minimum
      // `2 × slotNumber` (computeRingDim returned a larger entry from
      // the security table because logPQ exceeded the floor's logMaxQ).
      // In this case, Lattigo's actual ring is larger than what the
      // slot count implies — same correctness hazard for Rotom and
      // for any consumer that derives shape from the slot count.
      int64_t suggested = ringDim / 2;
      module.emitError()
          << "check-no-ringdim-bump: --generate-param-ckks selected "
             "ringDim="
          << ringDim << " (logN=" << logN << ") which exceeds the "
                                            "structural minimum 2 × "
          << actual << " = " << (2 * actual)
          << " — security constraints pushed the ring degree up. Rotom "
             "emits IR against slot count "
          << actual
          << ", so the resulting circuit would have inconsistent slot "
             "modulus. Re-run the front-end pipeline with "
             "`--ciphertext-degree="
          << suggested
          << "` so the requested slot count lands on a ringDim the "
             "security table can satisfy without bumping.";
      signalPassFailure();
      return;
    }
  }
};

}  // namespace
}  // namespace heir
}  // namespace mlir
