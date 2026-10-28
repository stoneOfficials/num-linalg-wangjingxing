"""演示经典四阶 Runge-Kutta 方法求解初值问题。

示例方程：y' = y，y(0) = 1，精确解为 y = exp(x)。

运行方式：
    python demo_rk4.py

要求：demo_rk4.py 与 rk4.py 放在同一目录。
"""

import math

from rk4 import rk4_step, rk4_solve


def main():
    def derivative(x, y):
        # y' = y；此例中导数不依赖 x。
        return y

    x0, y0 = 0.0, 1.0
    x_end, h = 2.0, 0.25

    # 先展示单步计算。
    one_step = rk4_step(derivative, x0, y0, h)
    print("单步 RK4 示例")
    print(f"  起点：x={x0:.2f}, y={y0:.8f}")
    print(f"  步长：h={h:.2f}")
    print(f"  一步后的近似值：y({x0 + h:.2f})={one_step:.10f}")
    print(f"  精确值：exp({x0 + h:.2f})={math.exp(x0 + h):.10f}")
    print(f"  绝对误差：{abs(one_step - math.exp(x0 + h)):.3e}")

    # 求解整个区间。rk4_solve 会将最后一个 x 精确设为 x_end。
    xs, ys = rk4_solve(derivative, x0, y0, x_end, h)

    print("\n区间积分结果：y' = y，y(0) = 1")
    print(f"{'x':>8} {'RK4 近似值':>18} {'精确值 exp(x)':>18} {'绝对误差':>14}")
    print("-" * 62)

    for x, y in zip(xs, ys):
        exact = math.exp(x)
        error = abs(y - exact)
        print(f"{x:8.4f} {y:18.10f} {exact:18.10f} {error:14.3e}")

    print(f"\n积分点数量：{len(xs)}")
    print(f"最终位置：{xs[-1]:.10f}（目标终点：{x_end:.10f}）")
    print(f"终点绝对误差：{abs(ys[-1] - math.exp(x_end)):.3e}")


if __name__ == "__main__":
    main()
