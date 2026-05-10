#ifndef LIB_DIALECT_LWE_CONVERSIONS_LWETOLINALG_HELPERS_BARRETT_H_
#define LIB_DIALECT_LWE_CONVERSIONS_LWETOLINALG_HELPERS_BARRETT_H_

#include "mlir/include/mlir/IR/Builders.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"   // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"      // from @llvm-project

// Modular-reduction emit helpers for use inside `linalg.generic` /
// `linalg.map` bodies. These are plain C++ utility functions (no MLIR
// pass) — each pattern's body builder calls into them directly.
//
// Two flavours, matching CROSS:
//   - emitLazyModReduce: assumes input in [0, 2q), one conditional
//     subtract. Used for radd, rsub_plain, etc.
//     (CROSS reference: jaxite_word/add.py:7-27.)
//   - emitBarrettReduce: full Barrett reduction for products in
//     [0, q^2). Used downstream by the multiply patterns (Phase 3).
//     (CROSS reference: jaxite_word/finite_field.py:200-266.)

namespace mlir::heir::lwe {

// Emit `select(x >= modulus, x - modulus, x)` and return the SSA result.
// Inputs and modulus must share the same integer type.
::mlir::Value emitLazyModReduce(::mlir::OpBuilder& builder,
                                ::mlir::Location loc, ::mlir::Value x,
                                ::mlir::Value modulus);

// Symmetric variant: input may be in (-q, q) (e.g. result of subi), output
// in [0, q). Adds modulus when negative, otherwise pass-through.
// Implemented as `select(x signed-lt 0, x + modulus, x)`.
::mlir::Value emitLazyModReduceSigned(::mlir::OpBuilder& builder,
                                      ::mlir::Location loc, ::mlir::Value x,
                                      ::mlir::Value modulus);

}  // namespace mlir::heir::lwe

#endif  // LIB_DIALECT_LWE_CONVERSIONS_LWETOLINALG_HELPERS_BARRETT_H_
