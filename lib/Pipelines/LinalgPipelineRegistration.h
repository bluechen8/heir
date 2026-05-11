#ifndef LIB_PIPELINES_LINALGPIPELINEREGISTRATION_H_
#define LIB_PIPELINES_LINALGPIPELINEREGISTRATION_H_

#include "lib/Pipelines/ArithmeticPipelineRegistration.h"
#include "mlir/include/mlir/Pass/PassManager.h"  // from @llvm-project

namespace mlir::heir {

// Tail pipeline: takes a module with `lwe.*` ops in a server-only shape
// (post `--split-client-interface --isolate-server-module`) and lowers it
// to portable `linalg + tensor + arith` IR ready for `iree-compile`.
//
// Mirrors the tail of `polynomialToLLVMPipelineBuilder` in spirit but
// stops at linalg — IREE owns codegen below this point.
void lweToLinalgPipelineBuilder(OpPassManager& manager);

// Full pipeline: composes `mlirToSecretArithmetic` + CKKS scheme lowering
// + `--ckks-to-lwe` + `--split-client-interface` + `--isolate-server-module`
// + the linalg tail. Registered as `--mlir-to-ciphertext-linalg`.
//
// Parallel to `--mlir-to-openfhe-pke` / `--mlir-to-lattigo`; for the
// linalg/IREE backend we keep only the server compute.
void toIreeLinalgPipelineBuilder(OpPassManager& pm,
                                 const MlirToRLWEPipelineOptions& options);

}  // namespace mlir::heir

#endif  // LIB_PIPELINES_LINALGPIPELINEREGISTRATION_H_
