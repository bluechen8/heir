// RUN: heir-opt --check-no-ringdim-bump %s --verify-diagnostics

// Has ckks.schemeParam but missing the slot count attributes that
// --generate-param-ckks should have stamped. The pass must error.

// expected-error @below {{check-no-ringdim-bump: missing `scheme.requested_slot_count` or `scheme.actual_slot_count`}}
module attributes {
    ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [536903681, 67043329], P = [536952833], logDefaultScale = 26>,
    scheme.ckks
} {
  func.func @forward(%arg0: tensor<4096xf64>) -> tensor<4096xf64> {
    return %arg0 : tensor<4096xf64>
  }
}
