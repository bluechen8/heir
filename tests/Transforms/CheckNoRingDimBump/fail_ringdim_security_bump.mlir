// RUN: heir-opt --check-no-ringdim-bump %s --verify-diagnostics

// Requested == actual == 4096, but logN == 14 → ringDim = 16384 != 2×4096.
// Security constraints pushed ringDim above the structural minimum.

// expected-error @below {{check-no-ringdim-bump: --generate-param-ckks selected ringDim=16384 (logN=14) which exceeds the structural minimum}}
module attributes {
    ckks.schemeParam = #ckks.scheme_param<logN = 14, Q = [36028797019389953, 35184372121601, 35184372744193], P = [36028797019488257], logDefaultScale = 45>,
    scheme.actual_slot_count = 4096 : i64,
    scheme.ckks,
    scheme.requested_slot_count = 4096 : i64
} {
  func.func @forward(%arg0: tensor<4096xf64>) -> tensor<4096xf64> {
    return %arg0 : tensor<4096xf64>
  }
}
