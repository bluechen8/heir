#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOLINALG_HELPERS_RNS_H_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOLINALG_HELPERS_RNS_H_

#include <cstdint>

#include "lib/Dialect/LWE/IR/LWETypes.h"
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"         // from @llvm-project
#include "mlir/include/mlir/Support/LogicalResult.h"  // from @llvm-project

// Helpers for translating an LWE ciphertext type into an RNS-explicit
// flat tensor type, and for materializing the per-limb modulus list
// alongside it. Used by every LWEToLinalg pattern's prologue.

namespace mlir::heir::lwe {

// Description of an LWE ciphertext in flat-tensor form:
//   tensor<numPolys x degree x numLimbs x storageType>
struct CiphertextLayout {
  int64_t numPolys;          // ciphertext_space.size, e.g. 2 for fresh RLWE
  int64_t degree;            // polynomial modulus degree (e.g. 16, 8192)
  int64_t numLimbs;          // RNS basis size
  ::mlir::IntegerType storageType;  // shared storage type of all limbs (e.g. i64)
  ::llvm::SmallVector<int64_t> moduli;  // per-limb moduli, length = numLimbs
};

// Inspect an LWE ciphertext type and extract its flattened layout.
// Fails if the ciphertext's coefficient type is not an RNS of mod_arith
// integers with a uniform storage type.
::mlir::FailureOr<CiphertextLayout> inspectCiphertext(
    ::mlir::heir::lwe::LWECiphertextType type);

// Convenience: build the converted tensor type from a layout.
::mlir::RankedTensorType buildFlatTensorType(::mlir::MLIRContext* ctx,
                                             const CiphertextLayout& layout);

}  // namespace mlir::heir::lwe

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOLINALG_HELPERS_RNS_H_
