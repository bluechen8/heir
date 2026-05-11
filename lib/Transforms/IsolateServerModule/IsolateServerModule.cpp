// IsolateServerModule: hoist the nested @heir.server_module child up to
// be the top-level module.
//
// Counterpart to --split-client-interface. After the split, the parent
// module contains two children annotated `heir.client_module` and
// `heir.server_module`. The Lattigo / OpenFHE arms continue to consume
// the parent (their codegen expects encrypt/decrypt/forward all in one
// module). For the linalg/IREE arm we only want `forward` and its
// callees, so this pass:
//   1. Finds the child `module @heir.server_module`.
//   2. Replaces the parent's body with the child's body.
//   3. Carries the parent's scheme attributes (e.g. ckks.schemeParam,
//      scheme.actual_slot_count) forward onto the new top-level —
//      `--lwe-to-linalg` and friends key off them.
//
// Idempotent: if no `heir.server_module` child exists, it's a no-op.

#include "lib/Transforms/IsolateServerModule/IsolateServerModule.h"

#include "lib/Dialect/ModuleAttributes.h"
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_ISOLATESERVERMODULE
#include "lib/Transforms/IsolateServerModule/IsolateServerModule.h.inc"

namespace {

struct IsolateServerModule
    : public impl::IsolateServerModuleBase<IsolateServerModule> {
  void runOnOperation() override {
    ModuleOp parent = getOperation();

    // Find the heir.server_module child, if any.
    ModuleOp serverChild;
    for (auto& op : *parent.getBody()) {
      if (auto inner = dyn_cast<ModuleOp>(&op)) {
        if (inner->hasAttr(kServerModuleAttrName)) {
          serverChild = inner;
          break;
        }
      }
    }
    if (!serverChild) return;  // idempotent no-op

    // Move every op from the server child into the parent body. Track the
    // sibling client module so we can erase it after.
    Block* parentBlock = parent.getBody();
    Block* childBlock = serverChild.getBody();

    // Collect parent children other than the server module to delete after
    // the move (the client module + any stray ops at the parent level).
    llvm::SmallVector<Operation*> toErase;
    for (auto& op : *parentBlock) {
      if (&op == serverChild.getOperation()) continue;
      toErase.push_back(&op);
    }

    // Move the server module's ops into the parent. Splice keeps SSA values
    // and symbol references valid.
    parentBlock->getOperations().splice(parentBlock->begin(),
                                        childBlock->getOperations());

    // Erase the now-empty server module + any sibling ops.
    serverChild.erase();
    for (Operation* op : toErase) op->erase();
  }
};

}  // namespace
}  // namespace heir
}  // namespace mlir
