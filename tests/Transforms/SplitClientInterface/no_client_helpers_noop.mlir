// RUN: heir-opt --split-client-interface %s | FileCheck %s

// When no client helpers exist, the pass should be a no-op (no nested
// modules introduced) — keeps the IR shape stable for non-A1 pipelines.

// CHECK-LABEL: module
// CHECK-NOT:   heir.client_module
// CHECK-NOT:   heir.server_module
// CHECK:       func.func @just_compute
module {
  func.func @just_compute(%arg0: tensor<4xi32>) -> tensor<4xi32> {
    return %arg0 : tensor<4xi32>
  }
}
