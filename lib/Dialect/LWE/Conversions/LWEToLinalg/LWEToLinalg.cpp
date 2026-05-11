// LWEToLinalg: lower the LWE dialect to portable linalg + tensor + arith
// IR consumable by `iree-compile`.
//
// Phase 1 covers the trivial elementwise ops only: radd, rsub, rnegate.
// (radd_plain / rsub_plain are stubbed pending plaintext-encoding work.)
//
// Type conversion: !lwe.lwe_ciphertext → tensor<numPolys x degree x
// numLimbs x iK> (the RNS limb axis materialized as a plain trailing
// tensor dim, matching the optional `--rns-explicit-limbs`
// preprocessing pass baked into the conversion). iK is MLIR's
// K-bit integer type; K is taken from the ciphertext's cmod_bits
// attribute (typically 64, yielding i64). Per-limb moduli flow
// in as `arith.constant tensor<numLimbs x iK>` and broadcast into each
// `linalg.generic` body via a limb-only indexing map.

#include "lib/Dialect/LWE/Conversions/LWEToLinalg/LWEToLinalg.h"

#include <utility>

#include "lib/Dialect/LWE/Conversions/LWEToLinalg/Helpers/Barrett.h"
#include "lib/Dialect/LWE/Conversions/LWEToLinalg/Helpers/RNS.h"
#include "lib/Dialect/LWE/IR/LWEDialect.h"
#include "lib/Dialect/LWE/IR/LWEOps.h"
#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "lib/Dialect/ModuleAttributes.h"
#include "lib/Utils/ConversionUtils.h"
#include "llvm/include/llvm/ADT/SmallVector.h"        // from @llvm-project
#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Linalg/IR/Linalg.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Tensor/IR/Tensor.h"  // from @llvm-project
#include "mlir/include/mlir/IR/AffineMap.h"              // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"               // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"      // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"             // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"           // from @llvm-project
#include "mlir/include/mlir/IR/PatternMatch.h"           // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"                  // from @llvm-project
#include "mlir/include/mlir/IR/ValueRange.h"             // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"              // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"     // from @llvm-project
#include "mlir/include/mlir/Transforms/DialectConversion.h"  // from @llvm-project

namespace mlir::heir::lwe {

#define GEN_PASS_DEF_LWETOLINALG
#include "lib/Dialect/LWE/Conversions/LWEToLinalg/LWEToLinalg.h.inc"

namespace {

// Materialize the per-limb moduli as `arith.constant tensor<numLimbs x iK>`.
Value materializeModuliConstant(OpBuilder& b, Location loc,
                                const CiphertextLayout& layout) {
  auto tensorTy =
      RankedTensorType::get({layout.numLimbs}, layout.storageType);
  SmallVector<APInt> elts;
  elts.reserve(layout.numLimbs);
  unsigned bitwidth = layout.storageType.getWidth();
  for (uint64_t m : layout.moduli) {
    elts.emplace_back(bitwidth, m, /*isSigned=*/false);
  }
  auto attr = DenseElementsAttr::get(tensorTy, elts);
  return arith::ConstantOp::create(b, loc, tensorTy, attr);
}

// Build a `linalg.generic` over `tensor<numPolys x degree x numLimbs x iK>`
// with two ciphertext inputs broadcast identity, one moduli input broadcast
// across the limb axis only, and a body that computes `body(a, b, m)`.
template <typename BodyFn>
linalg::GenericOp emitElementwiseGeneric(
    OpBuilder& b, Location loc, RankedTensorType resultTy, Value lhs,
    Value rhs, Value moduli, BodyFn&& bodyFn) {
  MLIRContext* ctx = b.getContext();
  unsigned rank = resultTy.getRank();
  AffineMap idMap = AffineMap::getMultiDimIdentityMap(rank, ctx);
  AffineMap limbMap =
      AffineMap::get(rank, /*symbolCount=*/0,
                     {b.getAffineDimExpr(rank - 1)}, ctx);
  SmallVector<AffineMap, 4> indexingMaps = {idMap, idMap, limbMap, idMap};
  SmallVector<utils::IteratorType> iterators(rank,
                                             utils::IteratorType::parallel);

  Value empty = tensor::EmptyOp::create(b, loc, resultTy.getShape(),
                                        resultTy.getElementType());
  return linalg::GenericOp::create(
      b, loc, /*resultTensorTypes=*/TypeRange{resultTy},
      /*inputs=*/ValueRange{lhs, rhs, moduli},
      /*outputs=*/ValueRange{empty}, indexingMaps, iterators,
      [&](OpBuilder& nestedB, Location nestedLoc, ValueRange args) {
        Value a = args[0], bv = args[1], m = args[2];
        Value out = bodyFn(nestedB, nestedLoc, a, bv, m);
        linalg::YieldOp::create(nestedB, nestedLoc, out);
      });
}

// Same shape but unary (rnegate): one ciphertext input + moduli + output.
template <typename BodyFn>
linalg::GenericOp emitUnaryGeneric(OpBuilder& b, Location loc,
                                   RankedTensorType resultTy, Value src,
                                   Value moduli, BodyFn&& bodyFn) {
  MLIRContext* ctx = b.getContext();
  unsigned rank = resultTy.getRank();
  AffineMap idMap = AffineMap::getMultiDimIdentityMap(rank, ctx);
  AffineMap limbMap =
      AffineMap::get(rank, 0, {b.getAffineDimExpr(rank - 1)}, ctx);
  SmallVector<AffineMap, 3> indexingMaps = {idMap, limbMap, idMap};
  SmallVector<utils::IteratorType> iterators(rank,
                                             utils::IteratorType::parallel);

  Value empty = tensor::EmptyOp::create(b, loc, resultTy.getShape(),
                                        resultTy.getElementType());
  return linalg::GenericOp::create(
      b, loc, TypeRange{resultTy}, ValueRange{src, moduli}, ValueRange{empty},
      indexingMaps, iterators,
      [&](OpBuilder& nestedB, Location nestedLoc, ValueRange args) {
        Value a = args[0], m = args[1];
        Value out = bodyFn(nestedB, nestedLoc, a, m);
        linalg::YieldOp::create(nestedB, nestedLoc, out);
      });
}

class CiphertextToTensorTypeConverter : public TypeConverter {
 public:
  CiphertextToTensorTypeConverter(MLIRContext* ctx) {
    addConversion([](Type t) { return t; }); // identity: everything else passes through
    addConversion([](LWECiphertextType t) -> Type {
      auto layoutOr = inspectCiphertext(t);
      if (failed(layoutOr)) return Type();
      return buildFlatTensorType(t.getContext(), *layoutOr);
    });  // scalar ciphertext → flat tensor
    // tensor<...x !lwe.lwe_ciphertext> → tensor<...x numPolys x degree x
    // numLimbs x iK> (limb axes appended).
    addConversion([](RankedTensorType t) -> std::optional<Type> {
      auto ctEltTy = dyn_cast<LWECiphertextType>(t.getElementType());
      if (!ctEltTy) return std::nullopt;
      auto layoutOr = inspectCiphertext(ctEltTy);
      if (failed(layoutOr)) return std::nullopt;
      SmallVector<int64_t> shape(t.getShape());
      shape.push_back(layoutOr->numPolys);
      shape.push_back(layoutOr->degree);
      shape.push_back(layoutOr->numLimbs);
      return RankedTensorType::get(shape, layoutOr->storageType);
    }); // tensor-of-ciphertext → bigger tensor
  }
};

struct ConvertRAdd : public OpConversionPattern<RAddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RAddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto layoutOr = inspectCiphertext(cast<LWECiphertextType>(op.getType()));
    if (failed(layoutOr))
      return rewriter.notifyMatchFailure(op, "unsupported ciphertext type");
    auto resultTy = buildFlatTensorType(getContext(), *layoutOr);
    Value moduli = materializeModuliConstant(rewriter, loc, *layoutOr);

    auto generic = emitElementwiseGeneric(
        rewriter, loc, resultTy, adaptor.getLhs(), adaptor.getRhs(), moduli,
        [](OpBuilder& b, Location l, Value a, Value bv, Value m) -> Value {
          Value sum = arith::AddIOp::create(b, l, a, bv);
          return emitLazyModReduce(b, l, sum, m);
        });
    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

struct ConvertRSub : public OpConversionPattern<RSubOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RSubOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto layoutOr = inspectCiphertext(cast<LWECiphertextType>(op.getType()));
    if (failed(layoutOr))
      return rewriter.notifyMatchFailure(op, "unsupported ciphertext type");
    auto resultTy = buildFlatTensorType(getContext(), *layoutOr);
    Value moduli = materializeModuliConstant(rewriter, loc, *layoutOr);

    auto generic = emitElementwiseGeneric(
        rewriter, loc, resultTy, adaptor.getLhs(), adaptor.getRhs(), moduli,
        [](OpBuilder& b, Location l, Value a, Value bv, Value m) -> Value {
          // (a - b) mod q via add-q-then-lazy-reduce: avoids signed paths
          // by computing (a + q - b) which lies in [0, 2q) when both a and
          // b are pre-reduced to [0, q).
          Value qMinusB = arith::SubIOp::create(b, l, m, bv);
          Value sum = arith::AddIOp::create(b, l, a, qMinusB);
          return emitLazyModReduce(b, l, sum, m);
        });
    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

// radd_plain / rsub_plain: result = (ct[0] ± pt, ct[1], ...). Only
// the constant-term polynomial is touched. We extract ct[0,:,:], run
// the same per-limb `emitElementwiseGeneric` helper as `radd`/`rsub`
// but against the pre-encoded plaintext constant, and insert back.
enum class CtPlainKind { Add, Sub };

LogicalResult lowerCtPlain(Operation* op, Value origLhs, Value origRhs,
                           Value adaptedLhs, CtPlainKind kind,
                           ConversionPatternRewriter& rewriter) {
  auto loc = op->getLoc();
  auto ctType =
      dyn_cast<LWECiphertextType>(getElementTypeOrSelf(origLhs.getType()));
  if (!ctType)
    return rewriter.notifyMatchFailure(op, "lhs is not a ciphertext");
  auto layoutOr = inspectCiphertext(ctType);
  if (failed(layoutOr))
    return rewriter.notifyMatchFailure(op, "unsupported ciphertext type");
  auto& layout = *layoutOr;

  auto encodeOp = origRhs.getDefiningOp<RLWEEncodeOp>();
  if (!encodeOp)
    return rewriter.notifyMatchFailure(
        op, "plaintext rhs does not come from lwe.rlwe_encode");
  // `ElementsAttr` covers both inline DenseElementsAttr and
  // resource-backed DenseResourceElementsAttr — `--fold-plaintext-encoding`
  // picks the storage based on data size/sparsity.
  auto limbsAttr = encodeOp->getAttrOfType<ElementsAttr>(kEncodedLimbsAttrName);
  if (!limbsAttr)
    return rewriter.notifyMatchFailure(
        op, "encode op missing lwe.encoded_limbs attr; run "
            "--fold-plaintext-encoding first");
  auto ptTy = dyn_cast<RankedTensorType>(limbsAttr.getType());
  if (!ptTy || !isa<IntegerType>(ptTy.getElementType()) ||
      ptTy.getRank() != 2 || ptTy.getShape()[0] != layout.degree ||
      ptTy.getShape()[1] != layout.numLimbs ||
      cast<IntegerType>(ptTy.getElementType()) != layout.storageType)
    return rewriter.notifyMatchFailure(
        op, "lwe.encoded_limbs shape does not match ciphertext layout");

  auto sliceTy = RankedTensorType::get({layout.degree, layout.numLimbs},
                                       layout.storageType);
  SmallVector<OpFoldResult> offsets(3, rewriter.getIndexAttr(0));
  SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(1),
                                     rewriter.getIndexAttr(layout.degree),
                                     rewriter.getIndexAttr(layout.numLimbs)};
  SmallVector<OpFoldResult> strides(3, rewriter.getIndexAttr(1));
  Value ct0 = tensor::ExtractSliceOp::create(rewriter, loc, sliceTy,
                                             adaptedLhs, offsets, sizes,
                                             strides);
  Value ptInput = arith::ConstantOp::create(rewriter, loc, ptTy, limbsAttr);
  Value moduli = materializeModuliConstant(rewriter, loc, layout);

  auto generic = emitElementwiseGeneric(
      rewriter, loc, sliceTy, ct0, ptInput, moduli,
      [kind](OpBuilder& b, Location nloc, Value a, Value p,
             Value m) -> Value {
        if (kind == CtPlainKind::Add) {
          Value s = arith::AddIOp::create(b, nloc, a, p);
          return emitLazyModReduce(b, nloc, s, m);
        }
        // (a - p) mod q via add-q-then-lazy-reduce, mirrors ConvertRSub.
        Value qMinusP = arith::SubIOp::create(b, nloc, m, p);
        Value s = arith::AddIOp::create(b, nloc, a, qMinusP);
        return emitLazyModReduce(b, nloc, s, m);
      });
  Value result = tensor::InsertSliceOp::create(rewriter, loc,
                                               generic.getResult(0),
                                               adaptedLhs, offsets, sizes,
                                               strides);
  rewriter.replaceOp(op, result);
  return success();
}

struct ConvertRAddPlain : public OpConversionPattern<RAddPlainOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      RAddPlainOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    return lowerCtPlain(op, op.getLhs(), op.getRhs(), adaptor.getLhs(),
                        CtPlainKind::Add, rewriter);
  }
};

struct ConvertRSubPlain : public OpConversionPattern<RSubPlainOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      RSubPlainOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    return lowerCtPlain(op, op.getLhs(), op.getRhs(), adaptor.getLhs(),
                        CtPlainKind::Sub, rewriter);
  }
};

struct ConvertRNegate : public OpConversionPattern<RNegateOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      RNegateOp op, OpAdaptor adaptor,
      ConversionPatternRewriter& rewriter) const override {
    auto loc = op.getLoc();
    auto layoutOr = inspectCiphertext(cast<LWECiphertextType>(op.getType()));
    if (failed(layoutOr))
      return rewriter.notifyMatchFailure(op, "unsupported ciphertext type");
    auto resultTy = buildFlatTensorType(getContext(), *layoutOr);
    Value moduli = materializeModuliConstant(rewriter, loc, *layoutOr);

    auto generic = emitUnaryGeneric(
        rewriter, loc, resultTy, adaptor.getInput(), moduli,
        [](OpBuilder& b, Location l, Value a, Value m) -> Value {
          // (-a) mod q == (q - a) when a in (0, q); == 0 when a == 0.
          // Both cases handled by `(q - a) mod q` via lazy-reduce
          // (q - 0 = q, which gets reduced back to 0).
          Value qMinusA = arith::SubIOp::create(b, l, m, a);
          return emitLazyModReduce(b, l, qMinusA, m);
        });
    rewriter.replaceOp(op, generic.getResult(0));
    return success();
  }
};

// Identify which top-level container(s) `lwe-to-linalg` should rewrite.
// Post-`--split-client-interface` modules nest a `heir.client_module`
// (which we must skip — its `lwe.rlwe_encrypt`/`lwe.rlwe_decrypt` ops
// flow through host codegen) and a `heir.server_module` (which we
// rewrite). If no nested server module is present, fall back to
// rewriting the top-level module.
SmallVector<Operation*> collectTargetContainers(ModuleOp module) {
  SmallVector<Operation*> targets;
  for (auto& op : *module.getBody()) {
    if (auto inner = dyn_cast<ModuleOp>(&op)) {
      if (inner->hasAttr(kServerModuleAttrName)) targets.push_back(inner);
    }
  }
  if (targets.empty()) targets.push_back(module);
  return targets;
}

struct LWEToLinalg : public impl::LWEToLinalgBase<LWEToLinalg> {
  void runOnOperation() override {
    MLIRContext* context = &getContext();
    ModuleOp module = getOperation();
    auto containers = collectTargetContainers(module);

    CiphertextToTensorTypeConverter typeConverter(context);

    for (Operation* container : containers) {
      ConversionTarget target(*context);
      target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect,
                             tensor::TensorDialect>();
      target.addLegalOp<ModuleOp>();
      target.addIllegalOp<RAddOp, RSubOp, RNegateOp, RAddPlainOp,
                          RSubPlainOp>();
      // `lwe.rlwe_encode` stays legal; it becomes dead after the
      // ct-plain patterns consume its `lwe.encoded_limbs` attribute and
      // is DCE'd downstream by `--canonicalize`.

      RewritePatternSet patterns(context);
      patterns.add<ConvertRAdd, ConvertRSub, ConvertRNegate, ConvertRAddPlain,
                   ConvertRSubPlain>(typeConverter, context);
      addStructuralConversionPatterns(typeConverter, patterns, target);
      addTensorOfTensorConversionPatterns(typeConverter, patterns, target);
      addTensorConversionPatterns(typeConverter, patterns, target);

      ConversionConfig config;
      config.allowPatternRollback = false;
      if (failed(applyPartialConversion(container, target, std::move(patterns),
                                        config))) {
        signalPassFailure();
        return;
      }
    }
  }
};

}  // namespace

}  // namespace mlir::heir::lwe
