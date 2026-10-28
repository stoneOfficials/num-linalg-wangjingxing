"""演示自然三次样条插值。

运行方式：
    python demo_spline.py

要求：demo_spline.py 与 spline.py、lu.py 放在同一目录。
"""

from spline import natural_cubic_spline, eval_spline


def main():
    # 一组非等距采样节点，样条会通过所有给定节点。
    xs = [0.0, 1.0, 2.5, 4.0, 5.0]
    ys = [0.0, 0.8, 0.6, 1.4, 1.0]

    segments = natural_cubic_spline(xs, ys)

    print("插值节点：")
    for x, y in zip(xs, ys):
        print(f"  ({x:.3f}, {y:.6f})")

    print("\n各区间的局部多项式系数：")
    print("形式：S_i(x) = a + b*t + c*t^2 + d*t^3，t = x - xs[i]")
    for i, (a, b, c, d) in enumerate(segments):
        print(
            f"  区间 [{xs[i]:.3f}, {xs[i + 1]:.3f}]: "
            f"a={a:.8f}, b={b:.8f}, c={c:.8f}, d={d:.8f}"
        )

    print("\n在原始节点处验证插值：")
    for x, expected in zip(xs, ys):
        actual = eval_spline(segments, xs, x)
        print(
            f"  x={x:6.3f}, 插值值={actual: .10f}, "
            f"原始值={expected: .10f}, 误差={abs(actual - expected):.3e}"
        )

    print("\n在更密集的位置求值：")
    sample_xs = [0.0, 0.5, 1.0, 1.75, 2.5, 3.25, 4.0, 4.5, 5.0]
    print(f"{'x':>8} {'S(x)':>16}")
    print("-" * 26)
    for x in sample_xs:
        print(f"{x:8.3f} {eval_spline(segments, xs, x):16.8f}")

    print("\n自然边界条件检查（首段左端、末段右端的二阶导数应接近 0）：")
    first_a, first_b, first_c, first_d = segments[0]
    last_a, last_b, last_c, last_d = segments[-1]
    left_second_derivative = 2.0 * first_c
    last_width = xs[-1] - xs[-2]
    right_second_derivative = (
        2.0 * last_c + 6.0 * last_d * last_width
    )
    print(f"  S''({xs[0]:.3f}) = {left_second_derivative:.3e}")
    print(f"  S''({xs[-1]:.3f}) = {right_second_derivative:.3e}")


if __name__ == "__main__":
    main()
