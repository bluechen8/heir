#ifndef LIB_TRANSFORMS_ISOLATESERVERMODULE_ISOLATESERVERMODULE_TD_
#define LIB_TRANSFORMS_ISOLATESERVERMODULE_ISOLATESERVERMODULE_TD_

include "mlir/Pass/PassBase.td"

def IsolateServerModule : Pass<"isolate-server-module", "mlir::ModuleOp"> {
  let summary = "Hoist a nested heir.server_module to be the top-level module.";
  let description = [{
    Counterpart to `--split-client-interface`. After
    `--split-client-interface` the parent module contains two children
    `module @heir.client_module` / `module @heir.server_module`; the
    Lattigo arm continues to consume the parent (it expects the client
    helpers next to `forward`). For the linalg/IREE arm we only want
    the server compute, so this pass replaces the parent body with the
    contents of the `heir.server_module` child and copies the parent's
    scheme attrs forward (so `--lwe-to-linalg` still sees
    `ckks.schemeParam` etc).

    Idempotent: a no-op when no `heir.server_module` child exists.
  }];
  let dependentDialects = ["mlir::func::FuncDialect"];
}

#endif  // LIB_TRANSFORMS_ISOLATESERVERMODULE_ISOLATESERVERMODULE_TD_
