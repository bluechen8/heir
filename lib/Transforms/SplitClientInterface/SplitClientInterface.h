#ifndef LIB_TRANSFORMS_SPLITCLIENTINTERFACE_SPLITCLIENTINTERFACE_H_
#define LIB_TRANSFORMS_SPLITCLIENTINTERFACE_SPLITCLIENTINTERFACE_H_

#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project

namespace mlir {
namespace heir {

#define GEN_PASS_DECL
#include "lib/Transforms/SplitClientInterface/SplitClientInterface.h.inc"

#define GEN_PASS_REGISTRATION
#include "lib/Transforms/SplitClientInterface/SplitClientInterface.h.inc"

}  // namespace heir
}  // namespace mlir

#endif  // LIB_TRANSFORMS_SPLITCLIENTINTERFACE_SPLITCLIENTINTERFACE_H_
