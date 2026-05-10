// RUN: heir-opt --split-client-interface %s | FileCheck %s

// Mirrors the func layout produced by --add-client-interface (Phase 0.2):
// `forward` plus encrypt/decrypt helpers tagged with client.{enc,dec}_func.
// After the split, helpers should sit in a heir.client_module nested
// module and `forward` in a heir.server_module nested module.

// CHECK-LABEL: module
// CHECK:      module attributes {heir.client_module}
// CHECK:        func.func @forward__encrypt__arg0
// CHECK-SAME:     attributes {client.enc_func
// CHECK:        func.func @forward__decrypt__result0
// CHECK-SAME:     attributes {client.dec_func
// CHECK:      module attributes {heir.server_module}
// CHECK:        func.func @forward
// CHECK-NOT:      attributes {client.

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
