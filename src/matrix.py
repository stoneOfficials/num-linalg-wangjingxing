"""基础矩阵运算模块。

提供矩阵封装、行变换、转置、矩阵乘法和行列式计算。
本模块仅依赖 Python 标准库。
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence


MatrixData = Sequence[Sequence[float]]


def _validate_matrix(data: MatrixData) -> tuple[int, int]:
    """验证矩阵非空且每一行长度一致，返回矩阵形状。"""
    if not data:
        raise ValueError("矩阵不能为空")

    if not data[0]:
        raise ValueError("矩阵每行不能为空")

    rows = len(data)
    cols = len(data[0])

    for i, row in enumerate(data):
        if len(row) != cols:
            raise ValueError(f"矩阵不是矩形：第 {i} 行长度不一致")

        for value in row:
            if not math.isfinite(float(value)):
                raise ValueError("矩阵元素必须是有限数值")

    return rows, cols


class Matrix:
    """简单的二维浮点矩阵。

    Args:
        data: 二维序列。构造时会复制数据，避免修改原始输入。
    """

    def __init__(self, data: MatrixData) -> None:
        _validate_matrix(data)
        self.data: list[list[float]] = [
            [float(value) for value in row]
            for row in data
        ]

    @property
    def shape(self) -> tuple[int, int]:
        """返回矩阵形状 (行数, 列数)。"""
        return len(self.data), len(self.data[0])

    def __getitem__(self, key):
        """支持 A[i]、A[i, j] 两种索引方式。"""
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("矩阵元素索引必须包含两个维度")
            i, j = key
            return self.data[i][j]

        return self.data[key]

    def __setitem__(self, key, value) -> None:
        """支持 A[i] = row 和 A[i, j] = value。"""
        if isinstance(key, tuple):
            if len(key) != 2:
                raise IndexError("矩阵元素索引必须包含两个维度")

            i, j = key
            number = float(value)

            if not math.isfinite(number):
                raise ValueError("矩阵元素必须是有限数值")

            self.data[i][j] = number
            return

        row = [float(item) for item in value]

        if len(row) != self.shape[1]:
            raise ValueError("替换行的长度必须与矩阵列数一致")

        if not all(math.isfinite(item) for item in row):
            raise ValueError("矩阵元素必须是有限数值")

        self.data[key] = row

    def swap_rows(self, i: int, j: int) -> None:
        """交换第 i 行和第 j 行。"""
        self.data[i], self.data[j] = self.data[j], self.data[i]

    def scale_row(self, i: int, factor: float) -> None:
        """将第 i 行乘以 factor。"""
        factor = float(factor)

        if not math.isfinite(factor):
            raise ValueError("缩放系数必须是有限数值")

        self.data[i] = [value * factor for value in self.data[i]]

    def add_row_to(self, i: int, j: int, factor: float) -> None:
        """执行第 i 行 += factor * 第 j 行。"""
        factor = float(factor)

        if not math.isfinite(factor):
            raise ValueError("行变换系数必须是有限数值")

        source = self.data[j]
        target = self.data[i]

        # 先计算新行，避免 i == j 时发生原地更新干扰。
        updated = [
            target[k] + factor * source[k]
            for k in range(self.shape[1])
        ]

        if not all(math.isfinite(value) for value in updated):
            raise ArithmeticError("行变换产生了非有限数值")

        self.data[i] = updated

    @property
    def T(self) -> Matrix:
        """返回转置矩阵。"""
        rows, cols = self.shape
        return Matrix([
            [self.data[i][j] for i in range(rows)]
            for j in range(cols)
        ])

    def __matmul__(self, other: Matrix) -> Matrix:
        """使用 A @ B 执行矩阵乘法。"""
        if not isinstance(other, Matrix):
            return NotImplemented

        rows_a, cols_a = self.shape
        rows_b, cols_b = other.shape

        if cols_a != rows_b:
            raise ValueError(
                f"矩阵维度不兼容：{self.shape} 与 {other.shape}"
            )

        # i-k-j 顺序可以复用 A[i][k]，减少重复索引。
        result = [[0.0] * cols_b for _ in range(rows_a)]

        for i in range(rows_a):
            result_row = result[i]
            row_a = self.data[i]

            for k in range(cols_a):
                value = row_a[k]
                row_b = other.data[k]

                for j in range(cols_b):
                    result_row[j] += value * row_b[j]

        return Matrix(result)

    def det(self) -> float:
        """通过带部分主元的消元计算方阵行列式。

        注意：行列式可能因浮点误差而不精确；
        对病态矩阵，结果的相对误差可能较大。
        """
        rows, cols = self.shape

        if rows != cols:
            raise ValueError("只有方阵才能计算行列式")

        # 只复制一次，后续在工作副本上进行消元。
        work = [row[:] for row in self.data]
        determinant = 1.0

        for k in range(rows):
            pivot_row = max(
                range(k, rows),
                key=lambda i: abs(work[i][k]),
            )
            pivot = work[pivot_row][k]

            if pivot == 0.0:
                return 0.0

            if pivot_row != k:
                work[k], work[pivot_row] = (
                    work[pivot_row],
                    work[k],
                )
                determinant = -determinant

            pivot = work[k][k]
            determinant *= pivot

            for i in range(k + 1, rows):
                factor = work[i][k] / pivot
                work[i][k] = 0.0

                for j in range(k + 1, rows):
                    work[i][j] -= factor * work[k][j]

        return determinant

    def to_list(self) -> list[list[float]]:
        """返回独立的二维列表副本。"""
        return [row[:] for row in self.data]

    def __repr__(self) -> str:
        return f"Matrix({self.data!r})"


def matmul(A: MatrixData, B: MatrixData) -> list[list[float]]:
    """计算两个二维列表所表示的矩阵的乘积。"""
    rows_a, cols_a = _validate_matrix(A)
    rows_b, cols_b = _validate_matrix(B)

    if cols_a != rows_b:
        raise ValueError(
            f"矩阵维度不兼容：({rows_a}, {cols_a}) "
            f"与 ({rows_b}, {cols_b})"
        )

    result = [[0.0] * cols_b for _ in range(rows_a)]

    for i in range(rows_a):
        row_a = A[i]
        result_row = result[i]

        for k in range(cols_a):
            value = float(row_a[k])

            for j in range(cols_b):
                result_row[j] += value * float(B[k][j])

    return result