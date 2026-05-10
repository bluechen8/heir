#ifndef LIB_TRANSFORMS_SPLITCLIENTINTERFACE_SPLITCLIENTINTERFACE_TD_
#define LIB_TRANSFORMS_SPLITCLIENTINTERFACE_SPLITCLIENTINTERFACE_TD_

include "mlir/Pass/PassBase.td"

def SplitClientInterface : Pass<"split-client-interface", "mlir::ModuleOp"> {
  let summary = "Peel client encrypt/decrypt funcs into a sibling client module.";
  let description = [{
    Walks the top-level funcs of the module. Funcs tagged `client.enc_func`,
    `client.dec_func`, `client.pack_func`, or `client.preprocessed_func` (set
    by `--add-client-interface`) are moved into a new nested
    `builtin.module` annotated `heir.client_module`. Remaining funcs (the
    server-side compute, e.g. `forward`) are moved into a sibling
    `builtin.module` annotated `heir.server_module`.

    Downstream pipelines lower the two nested modules separately:
    the client module continues through `--lwe-to-{openfhe,lattigo}` for host
    codegen; the server module continues through `--lwe-to-linalg` for IREE.
  }];
  let dependentDialects = ["mlir::func::FuncDialect"];
}

#endif  // LIB_TRANSFORMS_SPLITCLIENTINTERFACE_SPLITCLIENTINTERFACE_TD_
