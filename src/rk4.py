"""经典四阶 Runge-Kutta 常微分方程求解器。"""

from __future__ import annotations

import math
from collections.abc import Callable


Derivative = Callable[[float, float], float]


def _finite(value: float, name: str) -> float:
    """转换为浮点数并检查是否有限。"""
    value = float(value)

    if not math.isfinite(value):
        raise ValueError(f"{name} 必须是有限数值")

    return value


def rk4_step(
    f: Derivative,
    x: float,
    y: float,
    h: float,
) -> float:
    """执行一次经典四阶 Runge-Kutta 积分。

    求解 y' = f(x, y)，返回 x + h 处的近似解。
    """
    x = _finite(x, "x")
    y = _finite(y, "y")
    h = _finite(h, "h")

    if h == 0.0:
        raise ValueError("步长 h 不能为 0")

    k1 = float(f(x, y))
    k2 = float(f(x + h / 2.0, y + h * k1 / 2.0))
    k3 = float(f(x + h / 2.0, y + h * k2 / 2.0))
    k4 = float(f(x + h, y + h * k3))

    if not all(math.isfinite(k) for k in (k1, k2, k3, k4)):
        raise ArithmeticError("微分函数返回了非有限数值")

    result = y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    if not math.isfinite(result):
        raise ArithmeticError("RK4 单步计算产生了非有限数值")

    return result


def rk4_solve(
    f: Derivative,
    x0: float,
    y0: float,
    x_end: float,
    h: float,
) -> tuple[list[float], list[float]]:
    """在给定区间内求解一阶常微分方程初值问题。

    Args:
        f: 导数函数 f(x, y)。
        x0: 起始位置。
        y0: 初始函数值。
        x_end: 终止位置。
        h: 非零步长。实际积分方向由 x0 和 x_end 决定，
           步长绝对值决定每一步的最大长度。

    Returns:
        (xs, ys)，分别为积分位置和对应的近似解。
        最后一个位置严格设置为 x_end。
    """
    x0 = _finite(x0, "x0")
    y0 = _finite(y0, "y0")
    x_end = _finite(x_end, "x_end")
    h = _finite(h, "h")

    if h == 0.0:
        raise ValueError("步长 h 不能为 0")

    xs = [x0]
    ys = [y0]

    if x0 == x_end:
        return xs, ys

    direction = 1.0 if x_end > x0 else -1.0
    step_size = abs(h)

    x = x0
    y = y0

    while x != x_end:
        remaining = abs(x_end - x)

        # 最后一步可以小于标准步长，确保不越过终点。
        current_h = direction * min(step_size, remaining)

        if current_h == 0.0:
            raise ArithmeticError(
                "步长过小，无法在浮点数精度下继续推进积分位置"
            )

        next_y = rk4_step(f, x, y, current_h)

        if abs(current_h) == remaining:
            next_x = x_end
        else:
            next_x = x + current_h

        # 对极小步长和极大 x，浮点加法可能无法推进位置。
        if next_x == x:
            raise ArithmeticError(
                "积分位置无法推进，请检查步长和区间尺度"
            )

        if not math.isfinite(next_x):
            raise ArithmeticError("积分位置产生了非有限数值")

        xs.append(next_x)
        ys.append(next_y)

        x, y = next_x, next_y

    return xs, ys