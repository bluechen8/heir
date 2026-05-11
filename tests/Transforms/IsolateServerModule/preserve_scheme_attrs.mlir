// RUN: heir-opt --isolate-server-module %s | FileCheck %s

// Verify that scheme attrs on the parent module survive the hoist
// (--lwe-to-linalg keys off ckks.schemeParam etc.).

// CHECK-LABEL: module attributes
// CHECK-SAME:    ckks.schemeParam
// CHECK-SAME:    scheme.actual_slot_count
// CHECK:        func.func @forward
// CHECK-NOT:    module attributes {heir.client_module}
// CHECK-NOT:    module attributes {heir.server_module}

module attributes {
    ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [36028797018652673], P = [1152921504606994433], logDefaultScale = 45>,
    scheme.actual_slot_count = 4096 : i64,
    scheme.ckks
} {
  module attributes {heir.client_module} {
    func.func @forward__encrypt__arg0(%arg0: tensor<4xi32>) -> tensor<4xi32>
        attributes {client.enc_func = {func_name = "forward", index = 0 : i64}} {
      return %arg0 : tensor<4xi32>
    }
  }
  module attributes {heir.server_module} {
    func.func @forward(%arg0: tensor<4xi32>) -> tensor<4xi32> {
      return %arg0 : tensor<4xi32>
    }
  }
}
