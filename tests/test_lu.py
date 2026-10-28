import pytest

from src.matrix import Matrix
from src.lu import lu_decompose, solve


def multiply(a, b):
    rows, inner, cols = len(a), len(b), len(b[0])
    result = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for k in range(inner):
            for j in range(cols):
                result[i][j] += a[i][k] * b[k][j]
    return result


def approx_matrix(actual, expected, *, rel=1e-9, abs_=1e-9):
    assert len(actual) == len(expected)
    for row_a, row_e in zip(actual, expected):
        assert len(row_a) == len(row_e)
        for a, e in zip(row_a, row_e):
            assert a == pytest.approx(e, rel=rel, abs=abs_)


def permute_rows(a, piv):
    return [list(a[i]) for i in piv]


def test_lu_decompose_reconstructs_matrix_without_pivot():
    a = [[4, 3], [6, 3]]
    l, u, piv = lu_decompose(a)
    pa = permute_rows(a, piv)
    approx_matrix(multiply(l.to_list(), u.to_list()), pa)
    assert l[0, 0] == pytest.approx(1)
    assert l[1, 1] == pytest.approx(1)
    assert u[1, 0] == pytest.approx(0)


def test_lu_decompose_with_row_pivoting():
    a = [[0, 2], [1, 3]]
    l, u, piv = lu_decompose(a)
    pa = permute_rows(a, piv)
    approx_matrix(multiply(l.to_list(), u.to_list()), pa)
    assert piv == [1, 0]


@pytest.mark.parametrize("a", [[], [[1, 2], [3]], [[1, 2, 3], [4, 5, 6]]])
def test_lu_rejects_empty_or_non_square_matrix(a):
    with pytest.raises(ValueError):
        lu_decompose(a)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_lu_rejects_non_finite_matrix(value):
    with pytest.raises(ValueError):
        lu_decompose([[value]])


def test_lu_rejects_singular_matrix():
    with pytest.raises(ValueError):
        lu_decompose([[1, 2], [2, 4]])


def test_solve_known_solution():
    a = [[3, 1], [1, 2]]
    expected_x = [2, -1]
    b = [sum(a[i][j] * expected_x[j] for j in range(2)) for i in range(2)]
    result = solve(a, b)
    assert result == pytest.approx(expected_x)


def test_solve_requires_matching_rhs_length():
    with pytest.raises(ValueError):
        solve([[1, 0], [0, 1]], [1])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_solve_rejects_non_finite_rhs(value):
    with pytest.raises(ValueError):
        solve([[1]], [value])


def test_solve_with_row_pivoting():
    a = [[0, 1], [2, 3]]
    expected_x = [4, 5]
    b = [sum(a[i][j] * expected_x[j] for j in range(2)) for i in range(2)]
    assert solve(a, b) == pytest.approx(expected_x)


def test_solve_does_not_mutate_input():
    a = [[2, 1], [1, 3]]
    b = [1, 2]
    a_before = [row[:] for row in a]
    b_before = b[:]
    solve(a, b)
    assert a == a_before
    assert b == b_before


def test_solve_accepts_matrix_object():
    a = Matrix([[2, 0], [0, 4]])
    assert solve(a, [6, 8]) == pytest.approx([3, 2])
