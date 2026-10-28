"""自然三次样条插值模块。

每个区间使用局部坐标 t = x - xs[i]，
对应多项式 S_i(x) = a + b*t + c*t^2 + d*t^3。
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence


Segment = tuple[float, float, float, float]


def _validate_nodes(
    xs: Sequence[float],
    ys: Sequence[float] | None = None,
) -> tuple[list[float], list[float] | None]:
    """检查节点数量、有限性和严格递增条件。"""
    x_values = [float(value) for value in xs]

    if len(x_values) < 2:
        raise ValueError("至少需要两个插值节点")

    if not all(math.isfinite(value) for value in x_values):
        raise ValueError("节点位置必须是有限数值")

    if any(
        x_values[i] >= x_values[i + 1]
        for i in range(len(x_values) - 1)
    ):
        raise ValueError("节点位置必须严格递增")

    if ys is None:
        return x_values, None

    y_values = [float(value) for value in ys]

    if len(y_values) != len(x_values):
        raise ValueError("xs 和 ys 的长度必须相同")

    if not all(math.isfinite(value) for value in y_values):
        raise ValueError("节点函数值必须是有限数值")

    return x_values, y_values


def natural_cubic_spline(
    xs: Sequence[float],
    ys: Sequence[float],
) -> list[Segment]:
    """构建自然三次样条插值的分段多项式系数。

    Args:
        xs: 严格递增的节点位置。
        ys: 各节点对应的函数值。

    Returns:
        每个区间对应一个 (a, b, c, d)，满足：
        S_i(x) = a + b*t + c*t^2 + d*t^3，
        其中 t = x - xs[i]。

    自然边界条件为首尾节点的二阶导数均为零。
    """
    x_values, y_values = _validate_nodes(xs, ys)
    assert y_values is not None

    n = len(x_values)

    # h[i] 是相邻节点间距。
    h = [
        x_values[i + 1] - x_values[i]
        for i in range(n - 1)
    ]

    # M[i] 是样条在节点 xs[i] 处的二阶导数。
    # 自然边界条件：M[0] = M[n - 1] = 0。
    M = [0.0] * n

    if n > 2:
        size = n - 2

        # 对内部二阶导数建立三对角线性方程组。
        # 这里复用项目已有的 LU 求解器，避免引入额外依赖。
        A = [[0.0] * size for _ in range(size)]
        rhs = [0.0] * size

        for i in range(1, n - 1):
            row = i - 1

            A[row][row] = 2.0 * (h[i - 1] + h[i])

            if row > 0:
                A[row][row - 1] = h[i - 1]

            if row + 1 < size:
                A[row][row + 1] = h[i]

            left_slope = (
                (y_values[i] - y_values[i - 1]) / h[i - 1]
            )
            right_slope = (
                (y_values[i + 1] - y_values[i]) / h[i]
            )

            rhs[row] = 6.0 * (right_slope - left_slope)

        try:
            from .lu import solve
        except ImportError:
            from lu import solve

        interior_M = solve(A, rhs)
        M[1:n - 1] = interior_M

    segments: list[Segment] = []

    for i in range(n - 1):
        hi = h[i]
        yi = y_values[i]
        yi1 = y_values[i + 1]
        Mi = M[i]
        Mi1 = M[i + 1]

        a = yi
        b = (yi1 - yi) / hi - hi * (2.0 * Mi + Mi1) / 6.0
        c = Mi / 2.0
        d = (Mi1 - Mi) / (6.0 * hi)

        coefficients = (a, b, c, d)

        if not all(math.isfinite(value) for value in coefficients):
            raise ArithmeticError(
                f"第 {i} 个区间的样条系数出现非有限数值"
            )

        segments.append(coefficients)

    return segments


def eval_spline(
    segments: Sequence[Segment],
    xs: Sequence[float],
    x: float,
) -> float:
    """计算样条在指定位置的值。

    区间外使用首段或末段多项式进行外推。
    """
    x_values, _ = _validate_nodes(xs)
    x = float(x)

    if not math.isfinite(x):
        raise ValueError("待求值位置 x 必须是有限数值")

    if len(segments) != len(x_values) - 1:
        raise ValueError("样条区间数量必须等于节点数量减一")

    # bisect_right 可在 O(log n) 时间内定位区间。
    # 超出范围时分别使用首段和末段进行外推。
    i = bisect.bisect_right(x_values, x) - 1
    i = max(0, min(i, len(segments) - 1))

    a, b, c, d = segments[i]
    t = x - x_values[i]

    # Horner 形式减少乘法次数。
    result = ((d * t + c) * t + b) * t + a

    if not math.isfinite(result):
        raise ArithmeticError("样条求值产生了非有限数值")

    return result