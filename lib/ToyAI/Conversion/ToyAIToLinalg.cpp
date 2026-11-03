//===- ToyAIToLinalg.cpp - Lower ToyAI ops to linalg on tensors -----------===//

#include "ToyAI/Passes.h"
#include "ToyAI/ToyAIOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace toyai {
#define GEN_PASS_DEF_CONVERTTOYAITOLINALG
#include "ToyAI/Passes.h.inc"
} // namespace toyai
} // namespace mlir

using namespace mlir;
using namespace mlir::toyai;

namespace {

/// Creates `tensor.empty` of `type` filled with the scalar `value`.
Value createFilledTensor(OpBuilder &b, Location loc, RankedTensorType type,
                         double value) {
  Value empty = b.create<tensor::EmptyOp>(loc, type.getShape(),
                                          type.getElementType());
  Value scalar = b.create<arith::ConstantOp>(
      loc, b.getFloatAttr(type.getElementType(), value));
  return b.create<linalg::FillOp>(loc, ValueRange{scalar}, ValueRange{empty})
      .getResult(0);
}

/// Builds an all-parallel elementwise linalg.generic over `inputs`.
Value createElementwise(
    OpBuilder &b, Location loc, RankedTensorType type, ValueRange inputs,
    function_ref<Value(OpBuilder &, Location, ValueRange)> scalarFn) {
  Value init = b.create<tensor::EmptyOp>(loc, type.getShape(),
                                         type.getElementType());
  unsigned rank = type.getRank();
  SmallVector<AffineMap> maps(inputs.size() + 1,
                              b.getMultiDimIdentityMap(rank));
  SmallVector<utils::IteratorType> iterators(rank,
                                             utils::IteratorType::parallel);
  auto generic = b.create<linalg::GenericOp>(
      loc, TypeRange{type}, inputs, ValueRange{init}, maps, iterators,
      [&](OpBuilder &nb, Location nloc, ValueRange args) {
        Value result = scalarFn(nb, nloc, args.drop_back());
        nb.create<linalg::YieldOp>(nloc, result);
      });
  return generic.getResult(0);
}

struct MatmulLowering : OpConversionPattern<MatmulOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = cast<RankedTensorType>(op.getType());
    Value acc = createFilledTensor(rewriter, op.getLoc(), type, 0.0);
    auto matmul = rewriter.create<linalg::MatmulOp>(
        op.getLoc(), TypeRange{type},
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{acc});
    rewriter.replaceOp(op, matmul.getResult(0));
    return success();
  }
};

struct Conv2DLowering : OpConversionPattern<Conv2DOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(Conv2DOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto type = cast<RankedTensorType>(op.getType());
    Value input = adaptor.getInput();
    ArrayRef<int64_t> padding = op.getPadding();
    ArrayRef<int64_t> strides = op.getStrides();

    if (padding[0] != 0 || padding[1] != 0) {
      auto inType = cast<RankedTensorType>(input.getType());
      SmallVector<int64_t> paddedShape(inType.getShape());
      paddedShape[2] += 2 * padding[0];
      paddedShape[3] += 2 * padding[1];
      SmallVector<OpFoldResult> pads = {
          rewriter.getIndexAttr(0), rewriter.getIndexAttr(0),
          rewriter.getIndexAttr(padding[0]), rewriter.getIndexAttr(padding[1])};
      Value zero = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getFloatAttr(inType.getElementType(), 0.0));
      input = rewriter.create<tensor::PadOp>(
          loc, RankedTensorType::get(paddedShape, inType.getElementType()),
          input, pads, pads, zero);
    }

    Value acc = createFilledTensor(rewriter, loc, type, 0.0);
    auto conv = rewriter.create<linalg::Conv2DNchwFchwOp>(
        loc, TypeRange{type}, ValueRange{input, adaptor.getFilter()},
        ValueRange{acc}, rewriter.getI64VectorAttr(strides),
        rewriter.getI64VectorAttr({1, 1}));
    rewriter.replaceOp(op, conv.getResult(0));
    return success();
  }
};

struct AddLowering : OpConversionPattern<AddOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value result = createElementwise(
        rewriter, op.getLoc(), cast<RankedTensorType>(op.getType()),
        ValueRange{adaptor.getLhs(), adaptor.getRhs()},
        [](OpBuilder &b, Location loc, ValueRange args) -> Value {
          return b.create<arith::AddFOp>(loc, args[0], args[1]);
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ReluLowering : OpConversionPattern<ReluOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto type = cast<RankedTensorType>(op.getType());
    Value result = createElementwise(
        rewriter, op.getLoc(), type, ValueRange{adaptor.getInput()},
        [&](OpBuilder &b, Location loc, ValueRange args) -> Value {
          Value zero = b.create<arith::ConstantOp>(
              loc, b.getFloatAttr(type.getElementType(), 0.0));
          return b.create<arith::MaximumFOp>(loc, args[0], zero);
        });
    rewriter.replaceOp(op, result);
    return success();
  }
};

struct ConstantLowering : OpConversionPattern<ConstantOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(
        op, createFilledTensor(rewriter, op.getLoc(),
                               cast<RankedTensorType>(op.getType()),
                               op.getValue().convertToDouble()));
    return success();
  }
};

struct ConvertToyAIToLinalgPass
    : toyai::impl::ConvertToyAIToLinalgBase<ConvertToyAIToLinalgPass> {
  void runOnOperation() override {
    MLIRContext *ctx = &getContext();
    ConversionTarget target(*ctx);
    target.addIllegalDialect<ToyAIDialect>();
    target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect,
                           tensor::TensorDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    RewritePatternSet patterns(ctx);
    patterns.add<MatmulLowering, Conv2DLowering, AddLowering, ReluLowering,
                 ConstantLowering>(ctx);
    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
