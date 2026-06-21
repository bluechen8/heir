// RUN: heir-opt --check-no-ringdim-bump %s | FileCheck %s

// Module without ckks.schemeParam → pass is a no-op (BGV/BFV pipelines
// or pre-param-gen IR shouldn't fail).

// CHECK-LABEL: module
// CHECK:        func.func @forward

module {
  func.func @forward(%arg0: tensor<4xi32>) -> tensor<4xi32> {
    return %arg0 : tensor<4xi32>
  }
}
