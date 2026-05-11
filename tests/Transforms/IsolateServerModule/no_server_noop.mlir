// RUN: heir-opt --isolate-server-module %s | FileCheck %s

// No nested heir.server_module → pass is a no-op (the AddModel pre-split
// case shouldn't crash if the pipeline runs `isolate-server-module` on an
// IR that wasn't pre-split).

// CHECK-LABEL: module
// CHECK:        func.func @forward
// CHECK:          return

module {
  func.func @forward(%arg0: tensor<4xi32>) -> tensor<4xi32> {
    return %arg0 : tensor<4xi32>
  }
}
