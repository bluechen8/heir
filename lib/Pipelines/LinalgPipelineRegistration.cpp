// LinalgPipelineRegistration: glue that wires the new lwe-to-linalg path
// into a full pipeline (`--mlir-to-ciphertext-linalg`) parallel to the
// existing `--mlir-to-openfhe-pke` / `--mlir-to-lattigo`.
//
// Two builders:
//   * lweToLinalgPipelineBuilder — reusable tail consuming a server-only
//     module with `lwe.*` ops; emits portable linalg + tensor + arith
//     IR ready for iree-compile.
//   * toIreeLinalgPipelineBuilder — full pipeline composing the existing
//     mlir-to-ckks front-half with --split-client-interface +
//     --isolate-server-module + the linalg tail.
//
// Additive: no existing HEIR file is modified to use this builder. Only
// `tools/heir-opt.cpp` adds the `PassPipelineRegistration` for the new
// CLI flag.

#include "lib/Pipelines/LinalgPipelineRegistration.h"

#include "lib/Dialect/CKKS/Transforms/CKKSToLWE.h"
#include "lib/Dialect/LWE/Conversions/LWEToLinalg/LWEToLinalg.h"
#include "lib/Pipelines/ArithmeticPipelineRegistration.h"
#include "lib/Transforms/CheckNoRingDimBump/CheckNoRingDimBump.h"
#include "lib/Transforms/FoldPlaintextEncoding/FoldPlaintextEncoding.h"
#include "lib/Transforms/IsolateServerModule/IsolateServerModule.h"
#include "lib/Transforms/SplitClientInterface/SplitClientInterface.h"
#include "mlir/include/mlir/Pass/PassManager.h"   // from @llvm-project
#include "mlir/include/mlir/Transforms/Passes.h"  // from @llvm-project

namespace mlir::heir {

void lweToLinalgPipelineBuilder(OpPassManager& manager) {
  // Compile-time fold of plaintext encoding: rewrite `lwe.rlwe_encode`
  // on constant operands to consume the pre-encoded RNS limb tensor,
  // so the radd_plain / rsub_plain / rmul_plain patterns in
  // --lwe-to-linalg can lower against a constant tensor directly.
  manager.addPass(createFoldPlaintextEncoding());
  manager.addPass(lwe::createLWEToLinalg());
  manager.addPass(createCanonicalizerPass());
  manager.addPass(createCSEPass());
  // Output is linalg + tensor + arith — ready for iree-compile.
}

void toIreeLinalgPipelineBuilder(OpPassManager& pm,
                                 const MlirToRLWEPipelineOptions& options) {
  // Front-half is shared verbatim with the OpenFHE / Lattigo pipelines:
  // it produces secret-annotated CKKS IR with client encrypt/decrypt
  // helpers tagged in place.
  mlirToRLWEPipeline(pm, options, RLWEScheme::ckksScheme);

  // Fail loudly if --generate-param-ckks bumped the slot count above
  // the user-supplied --ciphertext-degree. Rotom emits packing /
  // rotation IR against the original slot count; a bumped CKKS scheme
  // would silently decrypt to wrong values. The diagnostic surfaces
  // the next-higher --ciphertext-degree the caller should try.
  pm.addPass(createCheckNoRingDimBump());

  // Drop CKKS ops to LWE — same fork point Lattigo / OpenFHE use.
  pm.addPass(ckks::createCKKSToLWE());
  pm.addPass(createCanonicalizerPass());

  // Peel encrypt/decrypt helpers into a sibling client module, then
  // hoist the server module up to be the top-level. The encrypt/decrypt
  // path is *not* lowered to linalg under A1 (cf.
  // research-plan/heir-linalg-iree.md): host OpenFHE/Lattigo continues
  // to produce ciphertexts; IREE only computes on them.
  pm.addPass(createSplitClientInterface());
  pm.addPass(createIsolateServerModule());

  // Linalg tail.
  lweToLinalgPipelineBuilder(pm);
}

}  // namespace mlir::heir
