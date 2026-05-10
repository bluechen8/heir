// SplitClientInterface: peel encrypt/decrypt/pack helper funcs into a
// sibling client module, leaving the server compute in its own module.
//
// Triggers under A1 (cf. research-plan/heir-linalg-iree.md): the IREE
// backend cannot consume `lwe.rlwe_encrypt`/`lwe.rlwe_decrypt`, so we
// fork the module into:
//   - client_module: lowered via --lwe-to-{openfhe,lattigo}
//   - server_module: lowered via --lwe-to-linalg → iree-compile
//
// Any helper func (e.g. a `client.pack_func` plaintext-layout helper)
// that is referenced from BOTH the client and server side is cloned
// into both modules with private visibility, preserving the symbol
// name so SymbolRefAttr operands resolve in their host module.

#include "lib/Transforms/SplitClientInterface/SplitClientInterface.h"

#include "lib/Dialect/ModuleAttributes.h"
#include "llvm/include/llvm/ADT/DenseSet.h"             // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"            // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"            // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"             // from @llvm-project
#include "mlir/include/mlir/IR/SymbolTable.h"           // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_SPLITCLIENTINTERFACE
#include "lib/Transforms/SplitClientInterface/SplitClientInterface.h.inc"

namespace {

// Collect every SymbolRef (e.g. func.call callees) that the given func
// references, transitively through nested ops and nested attributes.
void collectReferencedSymbols(func::FuncOp fn,
                              llvm::DenseSet<llvm::StringRef>& out) {
  auto uses = SymbolTable::getSymbolUses(fn.getOperation());
  if (!uses) return;
  for (const SymbolTable::SymbolUse& use : *uses) {
    out.insert(use.getSymbolRef().getLeafReference().getValue());
  }
}

struct SplitClientInterface
    : public impl::SplitClientInterfaceBase<SplitClientInterface> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Idempotency: bail if already split.
    for (auto& op : *module.getBody()) {
      if (auto inner = dyn_cast<ModuleOp>(&op)) {
        if (inner->hasAttr(kClientModuleAttrName) ||
            inner->hasAttr(kServerModuleAttrName)) {
          return;
        }
      }
    }

    // Step 1: partition tagged top-level funcs.
    llvm::SmallVector<func::FuncOp> clientFuncs;
    llvm::SmallVector<func::FuncOp> serverFuncs;
    for (auto fn : module.getOps<func::FuncOp>()) {
      if (isClientHelper(fn)) {
        clientFuncs.push_back(fn);
      } else {
        serverFuncs.push_back(fn);
      }
    }

    if (clientFuncs.empty()) return;

    // Step 2: figure out which symbols each side needs (transitive).
    // A helper called from both sides gets cloned into both.
    llvm::DenseSet<llvm::StringRef> clientNeeds;
    llvm::DenseSet<llvm::StringRef> serverNeeds;
    for (auto fn : clientFuncs) {
      clientNeeds.insert(fn.getSymName());
      collectReferencedSymbols(fn, clientNeeds);
    }
    for (auto fn : serverFuncs) {
      serverNeeds.insert(fn.getSymName());
      collectReferencedSymbols(fn, serverNeeds);
    }

    OpBuilder builder(module.getContext());
    builder.setInsertionPointToEnd(module.getBody());
    auto clientModule = ModuleOp::create(builder, module.getLoc());
    clientModule->setAttr(kClientModuleAttrName, builder.getUnitAttr());
    auto serverModule = ModuleOp::create(builder, module.getLoc());
    serverModule->setAttr(kServerModuleAttrName, builder.getUnitAttr());

    // Step 3: place each func. If needed by both sides, clone into the
    // non-primary side (with private visibility) and move into the
    // primary side. The primary side is the side it was tagged for
    // (client if isClientHelper, else server).
    OpBuilder clientB(clientModule.getBody(), clientModule.getBody()->end());
    OpBuilder serverB(serverModule.getBody(), serverModule.getBody()->end());

    auto cloneIntoOther = [&](func::FuncOp fn, OpBuilder& destB) {
      auto cloned = cast<func::FuncOp>(destB.clone(*fn.getOperation()));
      cloned.setPrivate();
    };

    for (func::FuncOp fn : clientFuncs) {
      llvm::StringRef name = fn.getSymName();
      bool serverAlsoNeeds = serverNeeds.contains(name);
      if (serverAlsoNeeds) cloneIntoOther(fn, serverB);
      fn->moveBefore(clientModule.getBody(), clientModule.getBody()->end());
    }
    for (func::FuncOp fn : serverFuncs) {
      llvm::StringRef name = fn.getSymName();
      bool clientAlsoNeeds = clientNeeds.contains(name);
      if (clientAlsoNeeds) cloneIntoOther(fn, clientB);
      fn->moveBefore(serverModule.getBody(), serverModule.getBody()->end());
    }
  }
};

}  // namespace
}  // namespace heir
}  // namespace mlir
