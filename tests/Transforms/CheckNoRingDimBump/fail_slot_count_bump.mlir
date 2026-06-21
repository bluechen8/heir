// RUN: heir-opt --check-no-ringdim-bump %s --verify-diagnostics

// Requested 1024, but param-gen bumped to 4096 (bootstrap floor).
// The pass must emit an error.

// expected-error @below {{check-no-ringdim-bump: --generate-param-ckks bumped the CKKS slot count from 1024}}
module attributes {
    ckks.schemeParam = #ckks.scheme_param<logN = 13, Q = [536903681, 67043329], P = [536952833], logDefaultScale = 26>,
    scheme.actual_slot_count = 4096 : i64,
    scheme.ckks,
    scheme.requested_slot_count = 1024 : i64
} {
  func.func @forward(%arg0: tensor<1024xf64>) -> tensor<1024xf64> {
    return %arg0 : tensor<1024xf64>
  }
}
