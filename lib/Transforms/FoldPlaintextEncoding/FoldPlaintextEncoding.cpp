// FoldPlaintextEncoding: pre-compute the RNS-encoded limb tensor for
// each `lwe.rlwe_encode` on a constant operand and stamp it as the
// discardable `lwe.encoded_limbs` attribute.
//
// Pipeline (matching Lattigo `Encoder.Encode` ∘ `Encrypt`):
//   1. CKKS InverseCanonicalEmbedding: real inputs become N/2 complex
//      CKKS slots (imag = 0). We run Lattigo's `SpecialIFFTDouble`
//      (rotGroup-indexed butterfly + bit-reverse) on the slot buffer,
//      then split iFFT outputs into the polynomial as
//        coeffs[i]         = real(iFFT[i])
//        coeffs[i + halfN] = imag(iFFT[i])  for i ∈ [0, N/2).
//      Indexing twiddles by `rotGroup[j] = 5^j mod 2N` bakes in the
//      CKKS slot-to-evaluation-point permutation; the same convention
//      is used by every mainstream CKKS library (Lattigo, OpenFHE,
//      SEAL, CROSS), so this is what lets HEIR-encoded plaintexts
//      pair element-wise with Lattigo-encrypted ciphertexts.
//   2. Coefficients are scaled by Δ = 2^logDefaultScale, rounded, and
//      reduced into [0, q_j) per RNS limb.
//   3. Per-limb negacyclic NTT (eval at ψ, ψ^3, …, ψ^{2N-1} mod q_j
//      where ψ is a primitive 2N-th root of unity). Output is in
//      bit-reversed natural order to match Lattigo's
//      `ring.NTTStandard` layout (`ring.Poly.Coeffs[lvl]`).
//
// The resulting `lwe.encoded_limbs` tensor lives in **evaluation
// form**, matching the convention used by Lattigo `*rlwe.Ciphertext`
// values throughout the runtime. Downstream linalg patterns add/sub
// element-wise without any further NTT until multiplication lands
// (Phase 3).
//
// Splat inputs (e.g. nn.Parameter(zeros) biases) skip both the iFFT
// and the NTT — the post-NTT result is a constant tensor whose value
// is `c · Δ mod q_j` tiled across the degree axis, so we write it
// directly. Necessary at production N=8192 where O(N²) trig + O(N²)
// DFT per encode would otherwise dominate heir-opt wall time.
//
// No-op when the module has no `ckks.schemeParam`, or when no encode
// op's input traces back to a compile-time constant (see
// `traceToConstant`).

#include "lib/Transforms/FoldPlaintextEncoding/FoldPlaintextEncoding.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <utility>

#include "lib/Transforms/FoldPlaintextEncoding/Ntt.h"

#include "lib/Dialect/CKKS/IR/CKKSAttributes.h"
#include "lib/Dialect/CKKS/IR/CKKSDialect.h"
#include "lib/Dialect/LWE/IR/LWEAttributes.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "llvm/include/llvm/ADT/SmallVector.h"          // from @llvm-project
#include "llvm/include/llvm/Support/MathExtras.h"       // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"   // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/SCF/IR/SCF.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h" // from @llvm-project
#include "mlir/include/mlir/IR/AsmState.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"              // from @llvm-project
#include "mlir/include/mlir/Interfaces/SideEffectInterfaces.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"     // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"            // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Matchers.h"              // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"          // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                 // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"    // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DEF_FOLDPLAINTEXTENCODING
#include "lib/Transforms/FoldPlaintextEncoding/FoldPlaintextEncoding.h.inc"

namespace {

// Read each float slot from `attr` into `out`. Handles both inline
// DenseElementsAttr (splat + heterogeneous, fp + int variants) and
// resource-backed DenseResourceElementsAttr (f32 / f64). Integer
// inputs are cast to double. Returns failure on unsupported element
// types.
LogicalResult readSlotValues(ElementsAttr attr, SmallVectorImpl<double>& out) {
  auto shape = attr.getShapedType().getShape();
  int64_t numElts = 1;
  for (auto d : shape) numElts *= d;
  out.clear();
  out.reserve(numElts);

  auto elementType = attr.getShapedType().getElementType();
  if (auto fpAttr = dyn_cast<DenseFPElementsAttr>(attr)) {
    for (APFloat f : fpAttr.getValues<APFloat>()) {
      bool losesInfo = false;
      f.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven,
                &losesInfo);
      out.push_back(f.convertToDouble());
    }
    return success();
  }
  if (auto intAttr = dyn_cast<DenseIntElementsAttr>(attr)) {
    for (APInt v : intAttr.getValues<APInt>()) {
      out.push_back(elementType.isUnsignedInteger()
                        ? static_cast<double>(v.getZExtValue())
                        : static_cast<double>(v.getSExtValue()));
    }
    return success();
  }
  // DenseResourceElementsAttr doesn't support the templated getValues<T>
  // iterator; we have to go through the typed wrapper and pull the raw
  // ArrayRef out of the blob.
  if (auto f32Res = dyn_cast<DenseF32ResourceElementsAttr>(attr)) {
    auto data = f32Res.tryGetAsArrayRef();
    if (!data) return failure();
    for (float v : *data) out.push_back(static_cast<double>(v));
    return success();
  }
  if (auto f64Res = dyn_cast<DenseF64ResourceElementsAttr>(attr)) {
    auto data = f64Res.tryGetAsArrayRef();
    if (!data) return failure();
    for (double v : *data) out.push_back(v);
    return success();
  }
  return failure();
}

// Trace `value` back through view-shaping ops to its underlying
// constant elements attribute, if any. Returns nullptr when the chain
// doesn't terminate at a constant we can read.
//
// Peels:
//   - `tensor.extract_slice` from a constant if the slice is a
//     contiguous prefix in the *innermost* dim (the encode op consumes
//     a 1-D slot vector, so this covers the common
//     `tensor.extract_slice` cleartext-shaping pattern emitted by
//     --add-client-interface / --ckks-to-lwe). For other slice shapes
//     we conservatively bail.
ElementsAttr traceToConstant(Value value) {
  Operation* defOp = value.getDefiningOp();
  if (!defOp) return nullptr;
  if (auto constOp = dyn_cast<arith::ConstantOp>(defOp)) {
    return dyn_cast<ElementsAttr>(constOp.getValue());
  }
  // Peel a call to a zero-arg single-result private function — the
  // shape HEIR's ConvertToCiphertextSemantics emits for the
  // `_assign_layout_*` helpers wrapping a constant input.
  if (auto callOp = dyn_cast<func::CallOp>(defOp)) {
    if (callOp.getNumResults() != 1) return nullptr;
    if (callOp.getNumOperands() != 0) return nullptr;
    auto callee = SymbolTable::lookupNearestSymbolFrom(callOp,
                                                       callOp.getCalleeAttr());
    auto func = dyn_cast_or_null<func::FuncOp>(callee);
    if (!func || func.isExternal() || func.getBlocks().size() != 1)
      return nullptr;
    auto retOp =
        dyn_cast<func::ReturnOp>(func.getBody().front().getTerminator());
    if (!retOp || retOp.getNumOperands() != 1) return nullptr;
    return traceToConstant(retOp.getOperand(0));
  }
  // Peel an `scf.for` whose body is a positional copy loop:
  //   for i in [lo, hi): out[..., i, ...] = src[..., i, ...]
  // The result of such a loop equals `src` (init values are fully
  // overwritten across the iteration). Body shape we expect:
  //   - exactly one tensor.extract of `src` and one tensor.insert
  //     into the iter_arg, with matching index lists;
  //   - yield of the insert's result;
  //   - all other body ops are pure index-arithmetic helpers (e.g.
  //     `arith.index_cast`) feeding the indices.
  // At least one extract index must transitively depend on the loop
  // induction variable, so the loop actually scans positions.
  if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
    if (forOp.getNumResults() != 1) return nullptr;
    if (forOp.getRegionIterArgs().size() != 1) return nullptr;
    Block& body = forOp.getRegion().front();

    tensor::ExtractOp extractOp;
    tensor::InsertOp insertOp;
    scf::YieldOp yieldOp;
    for (Operation& op : body) {
      if (auto e = dyn_cast<tensor::ExtractOp>(&op)) {
        if (extractOp) return nullptr;
        extractOp = e;
      } else if (auto i = dyn_cast<tensor::InsertOp>(&op)) {
        if (insertOp) return nullptr;
        insertOp = i;
      } else if (auto y = dyn_cast<scf::YieldOp>(&op)) {
        yieldOp = y;
      } else if (!isPure(&op)) {
        return nullptr;
      }
    }
    if (!extractOp || !insertOp || !yieldOp) return nullptr;
    if (insertOp.getScalar() != extractOp.getResult()) return nullptr;
    if (insertOp.getDest() != body.getArgument(1)) return nullptr;
    if (yieldOp.getNumOperands() != 1 ||
        yieldOp.getOperand(0) != insertOp.getResult())
      return nullptr;
    if (extractOp.getIndices() != insertOp.getIndices()) return nullptr;

    // Coverage check: at least one extract index must transitively
    // reach the loop's induction var (possibly through pure
    // index-arith ops like `arith.index_cast`). We follow a bounded
    // depth chain to keep this analysis cheap.
    Value iv = forOp.getInductionVar();
    auto dependsOnIv = [&](Value v) -> bool {
      llvm::SmallPtrSet<Value, 8> seen;
      llvm::SmallVector<Value, 8> worklist{v};
      while (!worklist.empty()) {
        Value cur = worklist.pop_back_val();
        if (!seen.insert(cur).second) continue;
        if (cur == iv) return true;
        if (auto* op = cur.getDefiningOp()) {
          if (!isPure(op)) continue;
          for (Value operand : op->getOperands()) worklist.push_back(operand);
        }
      }
      return false;
    };
    bool sawIv = false;
    for (Value idx : extractOp.getIndices()) {
      if (dependsOnIv(idx)) { sawIv = true; break; }
    }
    if (!sawIv) return nullptr;
    return traceToConstant(extractOp.getTensor());
  }
  if (auto sliceOp = dyn_cast<tensor::ExtractSliceOp>(defOp)) {
    auto src = traceToConstant(sliceOp.getSource());
    if (!src) return nullptr;
    // Splat sources collapse cleanly to a smaller splat regardless of
    // how the slice was shaped (every element of any sub-region of a
    // splat is the splat value).
    if (auto splat = dyn_cast<SplatElementsAttr>(src)) {
      return SplatElementsAttr::get(cast<ShapedType>(sliceOp.getType()),
                                    splat.getSplatValue<Attribute>());
    }
    // Identity-shaped slice (same total element count, possibly
    // reshaped — e.g. tensor<1x4096xf32> → tensor<1x4096xf32> or
    // ...→ tensor<4096xf32>): the slot vector is unchanged so we
    // pass the source attr through unchanged. The encoder doesn't
    // depend on rank, only on the flat element sequence. Real
    // sub-tensor extraction from a non-splat source (different
    // element count) requires byte-level slicing of the underlying
    // data and is left for a follow-up.
    auto srcType = cast<ShapedType>(sliceOp.getSource().getType());
    auto resultType = cast<ShapedType>(sliceOp.getType());
    if (srcType.getNumElements() == resultType.getNumElements()) {
      return src;
    }
    return nullptr;
  }
  return nullptr;
}

// CKKS-style scale, round, and RNS-decompose. Each output limb is a
// non-negative residue in [0, q_j).
APInt scaleRoundReduce(double slot, int64_t logScale, uint64_t modulus,
                       unsigned bitwidth) {
  // Scale by Δ = 2^logScale. For small N, a direct multiply suffices;
  // for very large logScale this would overflow double. Within v1 the
  // models stay well below that limit (logScale ≤ 60 typical, and the
  // products are ≤ |bias| · 2^60).
  long double scaled = static_cast<long double>(slot);
  scaled = std::ldexp(scaled, static_cast<int>(logScale));
  // Banker-free round-to-nearest-half-away-from-zero is fine for v1;
  // CKKS rounding semantics intentionally tolerate this freedom.
  long double rounded = std::round(scaled);
  // Mod q, mapping negative residues into [0, q).
  long double m = static_cast<long double>(modulus);
  long double r = std::fmod(rounded, m);
  if (r < 0) r += m;
  return APInt(bitwidth, static_cast<uint64_t>(r), /*isSigned=*/false);
}

// Invert the CKKS canonical embedding τ_N, matching Lattigo's
// `ckks.SpecialIFFTDouble` (rotGroup-indexed butterfly). Real-valued
// inputs are interpreted as the real parts of N/2 complex CKKS slots
// (imag = 0). The slot-to-evaluation-point map uses the 5-power
// permutation `rotGroup[i] = 5^i mod 2N` so a 1-slot rotation becomes
// the single automorphism σ₅: X ↦ X⁵. Every mainstream CKKS library
// (Lattigo / OpenFHE / SEAL / CROSS) uses this layout; matching it
// here is what lets HEIR-encoded plaintexts pair element-wise with
// Lattigo-encrypted ciphertexts.
//
// Pipeline (port of `schemes/ckks/ckks_vector_ops.go:SpecialIFFTDouble`
// in Lattigo v6.1.0):
//   1. Copy slot values into a complex buffer of length halfN = N/2.
//   2. In-place inverse butterfly with twiddles
//        roots[(lenq − (rotGroup[j] & mask)) << logGap]
//      where roots[m] = exp(2πi·m / M) and M = 2N. Indexing the
//      twiddle by rotGroup baked the slot permutation into the iFFT.
//   3. Divide by halfN.
//   4. Bit-reverse in place.
//   5. Map back to polynomial coefficients:
//        coeffs[i]         = real(buf[i])
//        coeffs[i + halfN] = imag(buf[i])
//      (Lattigo `Complex128ToFixedPointCRT`, ring.Standard branch.)
//
// A constant slot vector z = (c, c, …, c) maps to the constant
// polynomial p(x) = c (i.e. p[0] = c, p[j>0] = 0) under any of these
// conventions (after iFFT + bit-reverse, only buf[0] is non-zero and
// purely real), so we short-circuit splat input directly.
void inverseCanonicalEmbedding(ArrayRef<double> slots, int N,
                               SmallVectorImpl<double>& coeffs) {
  using Complex = std::complex<double>;
  assert(N > 0 && (N % 2 == 0) && "ring degree must be a positive even");
  const int halfN = N / 2;
  const int M = 2 * N;
  coeffs.assign(N, 0.0);

  const int numCopy = std::min(halfN, static_cast<int>(slots.size()));
  if (numCopy == 0) return;

  bool isSplat = true;
  for (int k = 1; k < numCopy; ++k) {
    if (slots[k] != slots[0]) { isSplat = false; break; }
  }
  // Trailing pads of zero break the splat invariant unless the splat
  // value is itself zero.
  if (numCopy < halfN && slots[0] != 0.0) isSplat = false;
  if (isSplat) {
    coeffs[0] = slots[0];
    return;
  }

  // Slot buffer (length halfN). Unused tail is left at zero.
  SmallVector<Complex, 32> buf(halfN, Complex(0.0, 0.0));
  for (int k = 0; k < numCopy; ++k) buf[k] = Complex(slots[k], 0.0);

  // rotGroup[i] = 5^i mod M, for i in [0, halfN).
  SmallVector<int, 32> rotGroup(halfN);
  int fivePows = 1;
  for (int i = 0; i < halfN; ++i) {
    rotGroup[i] = fivePows;
    fivePows = (fivePows * 5) & (M - 1);
  }

  // roots[m] = exp(2πi · m / M) for m ∈ [0, M]. The butterfly indexes
  // up to (lenq − 1) << logGap = M − (1<<logGap) < M, so size M
  // suffices, but we allocate M+1 to mirror Lattigo's convention.
  SmallVector<Complex, 64> roots(M + 1);
  const double twoPiOverM = 2.0 * M_PI / static_cast<double>(M);
  for (int m = 0; m <= M; ++m) {
    double angle = twoPiOverM * static_cast<double>(m);
    roots[m] = Complex(std::cos(angle), std::sin(angle));
  }

  // logN = log2(halfN), logM = log2(M). Both halfN and M are powers
  // of 2 by construction (N positive and even, M = 2N).
  const int logN = llvm::Log2_64(static_cast<uint64_t>(halfN));
  const int logM = llvm::Log2_64(static_cast<uint64_t>(M));

  // Lattigo `SpecialIFFTDouble` butterfly.
  for (int loglen = logN; loglen > 0; --loglen) {
    int len = 1 << loglen;
    int lenh = len >> 1;
    int lenq = len << 2;
    int logGap = logM - 2 - loglen;
    int mask = lenq - 1;
    for (int i = 0; i < halfN; i += len) {
      for (int j = 0, k = i; j < lenh; ++j, ++k) {
        Complex u = buf[k] + buf[k + lenh];
        int idx = (lenq - (rotGroup[j] & mask)) << logGap;
        Complex v = (buf[k] - buf[k + lenh]) * roots[idx];
        buf[k] = u;
        buf[k + lenh] = v;
      }
    }
  }
  for (int i = 0; i < halfN; ++i)
    buf[i] /= static_cast<double>(halfN);

  // Bit-reverse the slot buffer in place.
  auto bitReverseSmall = [](int x, int bits) {
    int r = 0;
    for (int b = 0; b < bits; ++b) { r = (r << 1) | (x & 1); x >>= 1; }
    return r;
  };
  for (int i = 0; i < halfN; ++i) {
    int j = bitReverseSmall(i, logN);
    if (j > i) std::swap(buf[i], buf[j]);
  }

  // Map iFFT output → real polynomial coefficients of degree N-1.
  for (int i = 0; i < halfN; ++i) {
    coeffs[i] = buf[i].real();
    coeffs[i + halfN] = buf[i].imag();
  }
}

// Build the limb-tensor attribute for the encoded plaintext. Splat
// data (all coefficients equal — by far the most common case for
// nn.Parameter biases and zero-initialized constants) prints
// compactly as inline `dense<value>` even at huge ring sizes, so we
// keep DenseElementsAttr for it. Heterogeneous tensors above
// `kInlineThresholdElements` switch to a `dense_resource<…>` blob to
// keep IR text from blowing up at production ring sizes (N≥1024 with
// 3+ RNS limbs prints tens of thousands of i64s otherwise). The
// `uniqueId` counter just keeps resource handle names distinct
// within the module — the runtime renames if collisions occur but
// stable names give cleaner diffs.
constexpr int64_t kInlineThresholdElements = 256;

ElementsAttr buildLimbsAttr(RankedTensorType tensorTy,
                             ArrayRef<APInt> limbs, int64_t uniqueId) {
  auto dense = DenseElementsAttr::get(tensorTy, limbs);
  if (dense.isSplat() ||
      tensorTy.getNumElements() <= kInlineThresholdElements)
    return dense;

  // Storage type is always i64 in this pass (see encodeOnePlaintext).
  // Re-materialize the values as a packed int64_t buffer so the
  // resource blob holds raw bytes IREE / downstream consumers can map
  // without further parsing.
  SmallVector<int64_t> raw;
  raw.reserve(limbs.size());
  for (const APInt& v : limbs)
    raw.push_back(static_cast<int64_t>(v.getZExtValue()));
  auto blob = HeapAsmResourceBlob::allocateAndCopyInferAlign<int64_t>(raw);
  std::string handle =
      ("lwe_encoded_limbs_" + llvm::Twine(uniqueId)).str();
  return cast<ElementsAttr>(DenseI64ResourceElementsAttr::get(
      tensorTy, handle, std::move(blob)));
}

// Compute the encoded RNS limb tensor for one rlwe_encode op.
// Returns an ElementsAttr of type tensor<degree x numLimbs x iK>
// containing the encoded coefficients — either inline DenseElementsAttr
// or DenseResourceElementsAttr, depending on size + sparsity. Returns
// failure if any required piece (scheme attr, encoding, ring) is
// missing or unsupported.
FailureOr<ElementsAttr> encodeOnePlaintext(
    lwe::RLWEEncodeOp op, ckks::SchemeParamAttr schemeAttr,
    ElementsAttr cleartext, int64_t uniqueId) {
  MLIRContext* ctx = op.getContext();

  // Extract degree from the encode op's ring.
  auto ring = op.getRing();
  auto polyMod = ring.getPolynomialModulus();
  if (!polyMod)
    return op.emitOpError() << "encode op ring lacks a polynomialModulus";
  int64_t degree = polyMod.getPolynomial().getDegree();

  // Read scheme params.
  int64_t logScale = schemeAttr.getLogDefaultScale();
  ArrayRef<int64_t> qiArr = schemeAttr.getQ().asArrayRef();
  if (qiArr.empty())
    return op.emitOpError() << "ckks.schemeParam has empty Q array";

  // Pick the limb storage width. CKKS limb prime sizes are <= 60 bits;
  // i64 is the canonical storage. We hard-pick i64 here; downstream
  // patterns inspect their own ciphertext layout to confirm.
  unsigned bitwidth = 64;
  auto storageType = IntegerType::get(ctx, bitwidth);

  // Read input slot vector.
  SmallVector<double> slots;
  if (failed(readSlotValues(cleartext, slots)))
    return op.emitOpError() << "unsupported cleartext element type";

  int64_t numLimbs = static_cast<int64_t>(qiArr.size());
  uint64_t twoN = static_cast<uint64_t>(2 * degree);

  // NTT-friendly check on every limb prime. Lattigo and OpenFHE param
  // generation always produce friendly primes, so a failure here means
  // scheme params were hand-tuned incorrectly.
  for (int64_t j = 0; j < numLimbs; ++j) {
    uint64_t modulus = static_cast<uint64_t>(qiArr[j]);
    if (modulus < 2 || (modulus - 1) % twoN != 0) {
      return op.emitOpError()
             << "modulus q_" << j << " = " << modulus
             << " does not satisfy q ≡ 1 (mod 2N=" << twoN
             << "); regenerate scheme params";
    }
  }

  // Detect splat input. The canonical embedding of a constant slot
  // vector z = (c, c, …, c) is the constant polynomial p(x) = c; the
  // negacyclic NTT then maps that to the constant evaluation tensor
  // (c, c, …, c) at every position. We skip both the O(N²) iFFT and
  // the O(N²) NTT in this case — important at production N=8192
  // where each constant nn.Parameter bias would otherwise dominate
  // heir-opt wall time.
  bool isSplat = !slots.empty();
  double splatVal = slots.empty() ? 0.0 : slots[0];
  for (double s : slots) {
    if (s != splatVal) { isSplat = false; break; }
  }
  // Slot vectors shorter than N/2 are implicitly zero-padded by the
  // canonical embedding; a non-zero splat with padding isn't a true
  // splat in the polynomial.
  int64_t halfN = degree / 2;
  if (isSplat && static_cast<int64_t>(slots.size()) < halfN &&
      splatVal != 0.0)
    isSplat = false;

  SmallVector<APInt> limbs;
  limbs.reserve(degree * numLimbs);

  if (isSplat) {
    // Per-limb residue c · Δ mod q_j, tiled across the degree axis.
    SmallVector<APInt, 8> perLimb;
    perLimb.reserve(numLimbs);
    for (int64_t j = 0; j < numLimbs; ++j) {
      uint64_t modulus = static_cast<uint64_t>(qiArr[j]);
      perLimb.push_back(
          scaleRoundReduce(splatVal, logScale, modulus, bitwidth));
    }
    for (int64_t i = 0; i < degree; ++i)
      for (int64_t j = 0; j < numLimbs; ++j) limbs.push_back(perLimb[j]);
  } else {
    // General path: ifft → coeff-form residues → per-limb negacyclic NTT.
    SmallVector<double, 32> realCoeffs;
    inverseCanonicalEmbedding(slots, static_cast<int>(degree), realCoeffs);

    // Storage layout in `limbs` is degree-major: limbs[i*numLimbs + j].
    limbs.assign(degree * numLimbs,
                 APInt(bitwidth, 0, /*isSigned=*/false));
    SmallVector<uint64_t> column(degree);
    SmallVector<uint64_t> evals(degree);
    SmallVector<uint64_t> psiPowers(twoN);
    for (int64_t j = 0; j < numLimbs; ++j) {
      uint64_t modulus = static_cast<uint64_t>(qiArr[j]);
      auto bredConst =
          fold_plaintext_encoding::computeBRed(modulus);
      uint64_t psi =
          fold_plaintext_encoding::findPrimitive2NthRoot(modulus, degree);
      if (psi == 0) {
        return op.emitOpError()
               << "could not derive a primitive 2N-th root of unity for q_"
               << j << " = " << modulus << " (N=" << degree << ")";
      }
      // psiPowers[m] = ψ^m mod q for m ∈ [0, 2N).
      psiPowers[0] = 1 % modulus;
      for (uint64_t m = 1; m < twoN; ++m) {
        psiPowers[m] = fold_plaintext_encoding::mulmod(
            psiPowers[m - 1], psi, modulus, bredConst);
      }

      // Gather column j (the per-degree coefficient-form residues for
      // this limb), NTT in place, scatter back into row-major limbs.
      for (int64_t i = 0; i < degree; ++i) {
        APInt v = scaleRoundReduce(realCoeffs[i], logScale, modulus,
                                   bitwidth);
        column[i] = v.getZExtValue();
      }
      fold_plaintext_encoding::nttPerLimb(
          /*coeffs=*/llvm::ArrayRef<uint64_t>(column),
          /*N=*/degree, /*q=*/modulus, /*bredConst=*/bredConst,
          /*psiPowers=*/llvm::ArrayRef<uint64_t>(psiPowers),
          /*evals=*/evals);
      for (int64_t i = 0; i < degree; ++i) {
        limbs[i * numLimbs + j] =
            APInt(bitwidth, evals[i], /*isSigned=*/false);
      }
    }
  }

  auto tensorTy = RankedTensorType::get({degree, numLimbs}, storageType);
  return buildLimbsAttr(tensorTy, limbs, uniqueId);
}

struct FoldPlaintextEncoding
    : public impl::FoldPlaintextEncodingBase<FoldPlaintextEncoding> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Lookup the ckks.schemeParam attribute. If the module nests
    // server/client child modules (post-`--split-client-interface`),
    // the scheme param lives on the parent; walk the encode ops
    // anywhere under the parent.
    auto schemeAttr = module->getAttrOfType<ckks::SchemeParamAttr>(
        ckks::CKKSDialect::kSchemeParamAttrName);

    // Walk the module looking for foldable encode ops. We collect
    // first, then rewrite, to avoid invalidating the walk.
    SmallVector<lwe::RLWEEncodeOp> candidates;
    module.walk([&](lwe::RLWEEncodeOp op) { candidates.push_back(op); });
    if (candidates.empty()) return;

    if (!schemeAttr) {
      // We only fold for CKKS today. Without scheme params we can't
      // know logDefaultScale or Q. Leave ops in place (downstream
      // pipelines for BGV/plaintext handle their own encoding).
      return;
    }

    int64_t uniqueId = 0;
    for (auto encodeOp : candidates) {
      // Idempotency: skip if we've already stamped this op.
      if (encodeOp->hasAttr(kEncodedLimbsAttrName)) continue;

      ElementsAttr cleartext = traceToConstant(encodeOp.getInput());
      if (!cleartext) {
        encodeOp.emitWarning()
            << "lwe.rlwe_encode operand does not trace to an arith.constant; "
               "skipping compile-time fold (runtime-plaintext path required)";
        continue;
      }

      auto encodedOr =
          encodeOnePlaintext(encodeOp, schemeAttr, cleartext, uniqueId++);
      if (failed(encodedOr)) {
        signalPassFailure();
        return;
      }
      encodeOp->setAttr(kEncodedLimbsAttrName, encodedOr.value());
    }
  }
};

}  // namespace
}  // namespace heir
}  // namespace mlir
