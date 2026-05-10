#include "lib/Dialect/LWE/Conversions/LWEToLinalg/Helpers/RNS.h"

#include "lib/Dialect/ModArith/IR/ModArithTypes.h"
#include "lib/Dialect/Polynomial/IR/PolynomialAttributes.h"
#include "lib/Dialect/RNS/IR/RNSTypes.h"
#include "llvm/include/llvm/ADT/TypeSwitch.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"     // from @llvm-project

namespace mlir::heir::lwe {

::mlir::FailureOr<CiphertextLayout> inspectCiphertext(
    LWECiphertextType type) {
  CiphertextLayout layout;

  auto ctSpace = type.getCiphertextSpace();
  layout.numPolys = ctSpace.getSize();

  auto ring = ctSpace.getRing();

  // Extract polynomial degree from the polynomialModulus.
  auto polyMod = ring.getPolynomialModulus();
  if (!polyMod) return failure();
  layout.degree = polyMod.getPolynomial().getDegree();

  // Extract RNS basis types from the coefficient type.
  auto coeffTy = ring.getCoefficientType();
  auto rnsTy = dyn_cast<rns::RNSType>(coeffTy);
  if (!rnsTy) return failure();

  auto basisTypes = rnsTy.getBasisTypes();
  layout.numLimbs = basisTypes.size();

  IntegerType sharedStorage;
  for (auto bt : basisTypes) {
    auto matTy = dyn_cast<mod_arith::ModArithType>(bt);
    if (!matTy) return failure();

    auto modAttr = matTy.getModulus();  // IntegerAttr
    auto storageTy = dyn_cast<IntegerType>(modAttr.getType());
    if (!storageTy) return failure();

    if (!sharedStorage) {
      sharedStorage = storageTy;
    } else if (sharedStorage != storageTy) {
      // Heterogeneous storage widths across limbs are not yet supported;
      // require the LWE pipeline to harmonize storage types upstream.
      return failure();
    }

    layout.moduli.push_back(modAttr.getValue().getZExtValue());
  }
  if (!sharedStorage) return failure();
  layout.storageType = sharedStorage;

  return layout;
}

RankedTensorType buildFlatTensorType(MLIRContext* ctx,
                                     const CiphertextLayout& layout) {
  return RankedTensorType::get(
      {layout.numPolys, layout.degree, layout.numLimbs}, layout.storageType);
}

}  // namespace mlir::heir::lwe
