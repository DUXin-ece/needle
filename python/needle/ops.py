"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(init.zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return a**self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return self.scalar * node.inputs[0] ** (self.scalar - 1) * out_grad
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a / b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a, b = node.inputs
        return out_grad / b, -a / b**2 * out_grad
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # astype is necessary if we use numpy backend, because sometimes when the result is small,
        # numpy will change the dtype from float32 to float64
        # for example: a = np.array([1, 2], dtype='float32')
        # (a/10000).dtype is float32, however, (a/100000).dtype will be float64
        # return (a / self.scalar).astype("float32")
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        axis = list(range(a.ndim))
        if self.axes == None:
            temp = axis[-1]
            axis[-1] = axis[-2]
            axis[-2] = temp
        else:
            temp = axis[self.axes[0]]
            axis[self.axes[0]] = axis[self.axes[1]]
            axis[self.axes[1]] = temp
        return a.permute(tuple(axis))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        transpose(out_grad, self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.reshape(self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        return out_grad.reshape(input_shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape).compact()

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        # use input_reshape to determine which axes need to be broadcasted
        if len(input_shape) != len(self.shape):
            input_reshape = (1,) * (len(self.shape) - len(input_shape)) + input_shape
        else:
            input_reshape = input_shape
        axes = []
        for i in range(len(input_reshape)):
            if input_reshape[i] != self.shape[i] or input_reshape[i] == 1:
                axes.append(i)
        if len(axes) == 1:
            axes = axes[0]
        else:
            axes = tuple(axes)
        return reshape(summation(out_grad, axes=axes), input_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    # axes can be none, int, or tuple.
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            self.axes = (axes,)
        else:
            self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if isinstance(self.axes, int):
            return array_api.summation(a, axis=self.axes, keepdims=False)
        elif isinstance(self.axes, tuple):
            for axis in self.axes:
                temp = array_api.summation(a, axis=self.axes, keepdims=False)
            return temp
        else:
            for axis in range(len(a.shape)):
                temp = array_api.summation(a, axis=self.axes, keepdims=False)
            return temp
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input_shape = node.inputs[0].shape
        shape = []
        if isinstance(self.axes, int):
            shape = list(input_shape)
            shape.insert(self.axes, 1)
        elif isinstance(self.axes, tuple):
            j = 0
            for i in range(len(input_shape)):
                if i in self.axes:
                    shape.append(1)
                else:
                    shape.append(out_grad.shape[j])
                    j = j + 1
        else:
            for i in range(len(input_shape)):
                shape.append(1)
        return broadcast_to(out_grad.reshape(shape), input_shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return a @ b
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        matA, matB = node.inputs
        if len(matA.shape) > len(matB.shape):
            axes = tuple(range(len(matA.shape) - len(matB.shape)))
            return (
                matmul(out_grad, matB.transpose()),
                summation(matmul(matA.transpose(), out_grad), axes),
            )
        elif len(matA.shape) < len(matB.shape):
            axes = tuple(range(len(matB.shape) - len(matA.shape)))
            return (
                summation(matmul(out_grad, matB.transpose()), axes),
                matmul(matA.transpose(), out_grad),
            )
        else:
            return (
                matmul(out_grad, matB.transpose()),
                matmul(matA.transpose(), out_grad),
            )
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return multiply(
            divide(
                broadcast_to(Tensor((1,), dtype="float32"), node.shape), node.inputs[0]
            ),
            out_grad,
        )
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return multiply(node, out_grad)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.maximum(0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return Tensor(node.numpy() > 0, dtype="float32") * out_grad
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if isinstance(axes, int):
            self.axes = (axes,)
        else:
            self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        Z_max = array_api.max(Z, axis=self.axes, keepdims=True)
        return array_api.log(
            array_api.summation(
                array_api.exp(
                    Z
                    - array_api.broadcast_to(
                        array_api.max(Z, axis=self.axes, keepdims=True), Z.shape
                    )
                ),
                axis=self.axes,
            )
        ) + array_api.max(Z, axis=self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        out_shape = out_grad.shape
        temp_shape = list(out_shape)
        if self.axes == None:
            temp_shape = len(node.inputs[0].shape) * (1,)
        else:
            for axis in self.axes:
                temp_shape.insert(axis, 1)
        Z_max = Tensor(
            array_api.max(node.inputs[0].numpy(), axis=self.axes, keepdims=True),
            dtype="float32",
            device=node.device,
        )
        Z_stable = node.inputs[0] - broadcast_to(Z_max, node.inputs[0].shape)

        return divide(
            exp(Z_stable),
            broadcast_to(
                reshape(summation(exp(Z_stable), self.axes), tuple(temp_shape)),
                Z_stable.shape,
            ),
        ) * broadcast_to(reshape(out_grad, tuple(temp_shape)), Z_stable.shape)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (
            broadcast_to(
                reshape(
                    Tensor((1,), dtype="float32", device=node.device),
                    len(node.shape) * (1,),
                ),
                node.shape,
            )
            - tanh(node.inputs[0]) * tanh(node.inputs[0])
        ) * out_grad
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):  # args -> tuple
        ### BEGIN YOUR SOLUTION
        shape = list(args[0].shape)
        shape.insert(self.axis, len(args))
        result = array_api.empty(tuple(shape), dtype="float32", device=args[0].device)
        idx = []
        for i in range(len(shape)):
            if i == self.axis:
                idx.append(0)
            else:
                idx.append(slice(0, shape[i], 1))
        for i in range(len(args)):
            idx[self.axis] = i
            result[tuple(idx)] = args[i]
        return result
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        temp = split(out_grad, self.axis)
        length = len(node.inputs[0])
        return make_tuple(*(temp[i] for i in range(length)))
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        idx = []
        new_shape = []
        for i in range(len(A.shape)):
            if i == self.axis:
                idx.append(0)
            else:
                idx.append(slice(0, A.shape[i], 1))
                new_shape.append(A.shape[i])
        result = []
        for i in range(A.shape[self.axis]):
            idx[self.axis] = i
            result.append(A[tuple(idx)].reshape(new_shape))
        return tuple(result)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):  # out_grad -> TensorTuple
        ### BEGIN YOUR SOLUTION
        return stack(out_grad, self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)


class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)


class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride: Optional[int] = 1, padding: Optional[int] = 0):
        self.stride = stride
        self.padding = padding

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=1):
    return Conv(stride, padding)(a, b)
