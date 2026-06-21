// RUN: heir-opt --fold-plaintext-encoding %s | FileCheck %s

// Non-constant slot vector exercising the SpecialIFFTDouble + per-
// limb negacyclic NTT path with values that distinguish coefficient
// encoding (which would yield (16, 32, 0, 0)) from the full
// Lattigo-form pipeline.
//
// Hand-computed reference (Δ = 2^4 = 16, q = 97, N = 4, halfN = 2,
// M = 2N = 8):
//   slots z = (1+0i, 2+0i)
//   rotGroup = (1, 5)
//   SpecialIFFTDouble inner butterfly (single stage, loglen=1):
//     len=2, lenh=1, lenq=8, logGap = log2(M) - 2 - loglen = 0,
//     mask = 7. twiddle = roots[(8 - (1 & 7)) << 0] = roots[7]
//     where roots[m] = exp(2πi · m / M).  roots[7] = exp(-iπ/4)
//     = (√2/2, -√2/2).
//   buf[0], buf[1] ← buf[0]+buf[1], (buf[0]-buf[1]) · roots[7]
//                  = 3, (-1) · (√2/2 − i√2/2) = (-√2/2 + i√2/2).
//   Divide by halfN=2: buf = (1.5, -√2/4 + i√2/4)
//                          ≈ (1.5, -0.3535… + 0.3535…i).
//   Bit-reverse (logN=1): identity.
//   Polynomial coeffs (coeffs[i]=real(buf[i]), coeffs[i+halfN]=
//   imag(buf[i])): (1.5, -0.3535…, 0.0, 0.3535…).
//   scaled coeffs = round(p · 16) = (24, -6, 0, 6)
//   mod 97                          = (24, 91, 0, 6).
//
// Negacyclic NTT, evaluating p(X) = 24 + 91·X + 0·X² + 6·X³ at the
// odd-power 2N-th roots of unity ψ^(2k+1) mod 97. The compiler picks
// the smallest generator g of (ℤ/97)* found by the Lattigo-compat
// search (start at g=3): g=5 (orders 3 and 4 fail; g=4 fails the
// 2-factor since 4^48 = 2^96 ≡ 1). ψ = 5^((q-1)/2N) = 5^12 = 64 mod
// 97. Natural-order evals (k=0..3) = (37, 11, 11, 37); written to
// `evals[bitReverse(k)]` (Lattigo's `ring.NTTStandard` layout):
//   evals = (37, 11, 11, 37).
//
// Independently cross-checked against a numpy port (Phase 2.2.2)
// and the new ckks_inverse_canonical_encode (Phase 2.2.3b).

!Zq = !mod_arith.int<97 : i64>
!rns_L0 = !rns.rns<!Zq>
#ring_rns = #polynomial.ring<coefficientType = !rns_L0, polynomialModulus = <1 + x**4>>
#ring_f64 = #polynomial.ring<coefficientType = f64, polynomialModulus = <1 + x**4>>
#ic = #lwe.inverse_canonical_encoding<scaling_factor = 4>
!pt = !lwe.lwe_plaintext<plaintext_space = <ring = #ring_f64, encoding = #ic>>

// CHECK-LABEL: func.func @encode_one_two
// CHECK:         lwe.rlwe_encode
// CHECK-SAME:    lwe.encoded_limbs = dense<{{\[}}[37], [11], [11], [37]]> : tensor<4x1xi64>
module attributes {ckks.schemeParam = #ckks.scheme_param<logN = 2, Q = [97], P = [193], logDefaultScale = 4>} {
  func.func @encode_one_two() -> !pt {
    %cst = arith.constant dense<[1.0, 2.0, 0.0, 0.0]> : tensor<4xf64>
    %pt = lwe.rlwe_encode %cst {encoding = #ic, ring = #ring_f64} : tensor<4xf64> -> !pt
    return %pt : !pt
  }
}
