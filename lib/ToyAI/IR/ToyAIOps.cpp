#include "ToyAI/ToyAIOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::toyai;

//===----------------------------------------------------------------------===//
// MatmulOp
//===----------------------------------------------------------------------===//

LogicalResult MatmulOp::inferReturnTypes(MLIRContext *context,
                                         std::optional<Location> location,
                                         Adaptor adaptor,
                                         SmallVectorImpl<Type> &inferred) {
  auto lhs = dyn_cast<RankedTensorType>(adaptor.getLhs().getType());
  auto rhs = dyn_cast<RankedTensorType>(adaptor.getRhs().getType());
  if (!lhs || !rhs)
    return emitOptionalError(location, "operands must be ranked tensors");
  if (lhs.getRank() != 2 || rhs.getRank() != 2)
    return emitOptionalError(location, "expects rank-2 operands, got ", lhs,
                             " and ", rhs);
  if (lhs.getDimSize(1) != rhs.getDimSize(0))
    return emitOptionalError(location, "contraction dimension mismatch: lhs ",
                             lhs, " has ", lhs.getDimSize(1),
                             " columns but rhs ", rhs, " has ",
                             rhs.getDimSize(0), " rows");
  inferred.push_back(RankedTensorType::get(
      {lhs.getDimSize(0), rhs.getDimSize(1)}, lhs.getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//
// Conv2DOp
//===----------------------------------------------------------------------===//

LogicalResult Conv2DOp::inferReturnTypes(MLIRContext *context,
                                         std::optional<Location> location,
                                         Adaptor adaptor,
                                         SmallVectorImpl<Type> &inferred) {
  auto input = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
  auto filter = dyn_cast<RankedTensorType>(adaptor.getFilter().getType());
  if (!input || !filter)
    return emitOptionalError(location, "operands must be ranked tensors");
  if (input.getRank() != 4 || filter.getRank() != 4)
    return emitOptionalError(
        location, "expects NCHW input and FCHW filter (rank 4), got ", input,
        " and ", filter);

  ArrayRef<int64_t> strides = adaptor.getStrides();
  ArrayRef<int64_t> padding = adaptor.getPadding();
  if (strides.size() != 2 || padding.size() != 2)
    return emitOptionalError(location,
                             "strides and padding must have 2 entries");
  if (strides[0] < 1 || strides[1] < 1)
    return emitOptionalError(location, "strides must be >= 1");
  if (padding[0] < 0 || padding[1] < 0)
    return emitOptionalError(location, "padding must be >= 0");

  if (input.getDimSize(1) != filter.getDimSize(1))
    return emitOptionalError(location, "channel mismatch: input has ",
                             input.getDimSize(1), " channels, filter expects ",
                             filter.getDimSize(1));

  int64_t oh = (input.getDimSize(2) + 2 * padding[0] - filter.getDimSize(2)) /
                   strides[0] +
               1;
  int64_t ow = (input.getDimSize(3) + 2 * padding[1] - filter.getDimSize(3)) /
                   strides[1] +
               1;
  if (oh < 1 || ow < 1)
    return emitOptionalError(location, "filter ", filter,
                             " does not fit the (padded) input ", input);

  inferred.push_back(RankedTensorType::get(
      {input.getDimSize(0), filter.getDimSize(0), oh, ow},
      input.getElementType()));
  return success();
}

//===----------------------------------------------------------------------===//
// ReluOp
//===----------------------------------------------------------------------===//

OpFoldResult ReluOp::fold(FoldAdaptor) {
  // relu is idempotent: relu(relu(x)) == relu(x).
  if (auto inner = getInput().getDefiningOp<ReluOp>())
    return inner.getResult();
  return {};
}

#define GET_OP_CLASSES
#include "ToyAI/ToyAIOps.cpp.inc"
