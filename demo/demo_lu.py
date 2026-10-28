"""演示带部分主元的 LU 分解与线性方程组求解。

运行方式：
    python demo_lu.py

要求：demo_lu.py 与 lu.py、matrix.py 放在同一目录。
"""

from lu import lu_decompose, solve


def matmul(a, b):
    """纯 Python 矩阵乘法，仅用于验证 P @ A = L @ U。"""
    if not a or not b or len(a[0]) != len(b):
        raise ValueError("矩阵维度不匹配")

    rows, inner, cols = len(a), len(b), len(b[0])
    return [
        [
            sum(a[i][k] * b[k][j] for k in range(inner))
            for j in range(cols)
        ]
        for i in range(rows)
    ]


def max_abs_diff(a, b):
    return max(
        abs(a[i][j] - b[i][j])
        for i in range(len(a))
        for j in range(len(a[0]))
    )


def main():
    # 特意让第一列的最大主元不在第一行，以演示行交换。
    A = [
        [0.0, 2.0, 1.0],
        [1.0, 1.0, 0.0],
        [2.0, 0.0, 1.0],
    ]
    b = [7.0, 3.0, 5.0]

    print("原始矩阵 A：")
    for row in A:
        print("  ", row)
    print("右端向量 b：", b)

    L_matrix, U_matrix, piv = lu_decompose(A)
    L = L_matrix.to_list()
    U = U_matrix.to_list()

    print("\nL（单位下三角矩阵）：")
    for row in L:
        print("  ", [round(value, 6) for value in row])

    print("\nU（上三角矩阵）：")
    for row in U:
        print("  ", [round(value, 6) for value in row])

    print("\n行置换索引 piv：", piv)
    print("含义：P 的第 i 行取单位矩阵的第 piv[i] 行。")

    # 按 piv 对 A 的行重排，得到 P @ A。
    PA = [A[piv[i]][:] for i in range(len(piv))]
    LU = matmul(L, U)
    print("\nP @ A：")
    for row in PA:
        print("  ", [round(value, 6) for value in row])
    print("\nL @ U：")
    for row in LU:
        print("  ", [round(value, 6) for value in row])
    print("分解最大绝对误差：", f"{max_abs_diff(PA, LU):.3e}")

    x = solve(A, b)
    print("\n求解 Ax = b 得到 x：", [round(value, 10) for value in x])

    # 直接代回原方程，检查 A @ x 与 b 的差异。
    Ax = [
        sum(A[i][j] * x[j] for j in range(len(x)))
        for i in range(len(A))
    ]
    residual = [Ax[i] - b[i] for i in range(len(b))]
    print("A @ x：", [round(value, 10) for value in Ax])
    print("残差 A @ x - b：", [f"{value:.3e}" for value in residual])


if __name__ == "__main__":
    main()
