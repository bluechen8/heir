// RUN: heir-opt --split-client-interface --isolate-server-module %s | FileCheck %s

// After --split-client-interface produces two children
// (heir.client_module + heir.server_module), --isolate-server-module
// hoists the server-module body up to be the top-level body. The
// client module and its encrypt/decrypt helpers are dropped (the
// IREE arm doesn't compile them).

// CHECK-LABEL: module
// CHECK-NOT:    module attributes {heir.client_module}
// CHECK-NOT:    module attributes {heir.server_module}
// CHECK:        func.func @forward
// CHECK-NOT:    func.func @forward__encrypt__arg0
// CHECK-NOT:    func.func @forward__decrypt__result0

module {
  func.func @forward(%arg0: tensor<4xi32>) -> tensor<4xi32> {
    return %arg0 : tensor<4xi32>
  }
  func.func @forward__encrypt__arg0(%arg0: tensor<4xi32>) -> tensor<4xi32>
      attributes {client.enc_func = {func_name = "forward", index = 0 : i64}} {
    return %arg0 : tensor<4xi32>
  }
  func.func @forward__decrypt__result0(%arg0: tensor<4xi32>) -> tensor<4xi32>
      attributes {client.dec_func = {func_name = "forward", index = 0 : i64}} {
    return %arg0 : tensor<4xi32>
  }
}
