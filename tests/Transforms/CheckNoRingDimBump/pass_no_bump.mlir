// RUN: heir-opt --check-no-ringdim-bump %s | FileCheck %s

// Requested == actual == 4096, logN == 13 → ringDim = 8192 == 2×4096.
// All invariants hold; pass is a no-op.

// CHECK-LABEL: module
// CHECK:        func.func @forward

module attributes {
    ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [536903681, 67043329], P = [536952833], logDefaultScale = 26>,
    scheme.actual_slot_count = 4096 : i64,
    scheme.ckks,
    scheme.requested_slot_count = 4096 : i64
} {
  func.func @forward(%arg0: tensor<4096xf64>) -> tensor<4096xf64> {
    return %arg0 : tensor<4096xf64>
  }
}
