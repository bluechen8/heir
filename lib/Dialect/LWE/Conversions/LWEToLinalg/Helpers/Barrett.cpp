#include "lib/Dialect/LWE/Conversions/LWEToLinalg/Helpers/Barrett.h"

#include "mlir/include/mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project

namespace mlir::heir::lwe {

Value emitLazyModReduce(OpBuilder& builder, Location loc, Value x,
                        Value modulus) {
  Value cond = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::uge,
                                     x, modulus);
  Value subbed = arith::SubIOp::create(builder, loc, x, modulus);
  return arith::SelectOp::create(builder, loc, cond, subbed, x);
}

Value emitLazyModReduceSigned(OpBuilder& builder, Location loc, Value x,
                              Value modulus) {
  Value zero =
      arith::ConstantIntOp::create(builder, loc, modulus.getType(), 0);
  Value cond = arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                     x, zero);
  Value added = arith::AddIOp::create(builder, loc, x, modulus);
  return arith::SelectOp::create(builder, loc, cond, added, x);
}

}  // namespace mlir::heir::lwe
