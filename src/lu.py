"""带部分主元的 LU 分解与线性方程组求解。"""

from __future__ import annotations

import math
from typing import Sequence

try:
    from .matrix import Matrix
except ImportError:
    from matrix import Matrix


MatrixInput = Matrix | Sequence[Sequence[float]]


def _as_list(A: MatrixInput) -> list[list[float]]:
    """将输入矩阵转换为独立的浮点二维列表。"""
    if isinstance(A, Matrix):
        data = A.to_list()
    else:
        data = [list(row) for row in A]

    if not data or not data[0]:
        raise ValueError("矩阵不能为空")

    n = len(data)

    if any(len(row) != n for row in data):
        raise ValueError("LU 分解要求输入为方阵")

    result = [[float(value) for value in row] for row in data]

    if not all(math.isfinite(value) for row in result for value in row):
        raise ValueError("矩阵元素必须是有限数值")

    return result


def lu_decompose(
    A: MatrixInput,
) -> tuple[Matrix, Matrix, list[int]]:
    """计算带部分主元的 LU 分解。

    返回：
        L: 对角线为 1 的下三角矩阵。
        U: 上三角矩阵。
        piv: 行置换索引，满足 P @ A = L @ U，
             其中 P 的第 i 行为单位矩阵的第 piv[i] 行。

    Raises:
        ValueError: 输入不是合法的非空方阵，或矩阵奇异。
        ArithmeticError: 计算产生非有限数值。
    """
    U = _as_list(A)
    n = len(U)

    L = [
        [1.0 if i == j else 0.0 for j in range(n)]
        for i in range(n)
    ]
    piv = list(range(n))

    for k in range(n):
        # 在当前列中选择绝对值最大的主元。
        pivot_row = max(
            range(k, n),
            key=lambda i: abs(U[i][k]),
        )
        pivot = U[pivot_row][k]

        if pivot == 0.0:
            raise ValueError("矩阵奇异，无法进行 LU 分解")

        if pivot_row != k:
            U[k], U[pivot_row] = U[pivot_row], U[k]
            piv[k], piv[pivot_row] = piv[pivot_row], piv[k]

            # 只交换 L 中已经计算完成的列。
            L[k][:k], L[pivot_row][:k] = (
                L[pivot_row][:k],
                L[k][:k],
            )

        pivot = U[k][k]

        for i in range(k + 1, n):
            factor = U[i][k] / pivot
            L[i][k] = factor
            U[i][k] = 0.0

            for j in range(k + 1, n):
                U[i][j] -= factor * U[k][j]

        if not all(
            math.isfinite(value)
            for row in U
            for value in row
        ):
            raise ArithmeticError("LU 分解产生了非有限数值")

        if not all(
            math.isfinite(value)
            for row in L
            for value in row
        ):
            raise ArithmeticError("LU 分解产生了非有限数值")

    return Matrix(L), Matrix(U), piv


def solve(
    A: MatrixInput,
    b: Sequence[float],
) -> list[float]:
    """利用带部分主元的 LU 分解求解 Ax = b。

    Args:
        A: 非空方阵，可以是 Matrix 或二维序列。
        b: 长度与 A 的阶数一致的右端向量。

    Returns:
        线性方程组的数值解。

    Raises:
        ValueError: 输入维度不匹配或矩阵奇异。
        ArithmeticError: 求解过程产生非有限数值。
    """
    matrix = _as_list(A)
    n = len(matrix)

    if len(b) != n:
        raise ValueError("向量 b 的长度必须与矩阵阶数一致")

    rhs = [float(value) for value in b]

    if not all(math.isfinite(value) for value in rhs):
        raise ValueError("向量 b 的元素必须是有限数值")

    L_matrix, U_matrix, piv = lu_decompose(matrix)
    L = L_matrix.to_list()
    U = U_matrix.to_list()

    # P @ b：根据分解时记录的行置换重排右端向量。
    pb = [rhs[piv[i]] for i in range(n)]

    # 前代：L @ y = P @ b。
    y = [0.0] * n

    for i in range(n):
        value = pb[i]

        for j in range(i):
            value -= L[i][j] * y[j]

        # L 的对角线为 1。
        y[i] = value

    # 回代：U @ x = y。
    x = [0.0] * n

    for i in range(n - 1, -1, -1):
        value = y[i]

        for j in range(i + 1, n):
            value -= U[i][j] * x[j]

        pivot = U[i][i]

        if pivot == 0.0:
            raise ValueError("矩阵奇异，无法求解线性方程组")

        x[i] = value / pivot

        if not math.isfinite(x[i]):
            raise ArithmeticError("求解结果包含非有限数值")

    return x