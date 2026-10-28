import math
import pytest

from src.matrix import Matrix, matmul


def approx_matrix(actual, expected, *, rel=1e-9, abs_=1e-9):
    assert len(actual) == len(expected)
    for row_a, row_e in zip(actual, expected):
        assert len(row_a) == len(row_e)
        for a, e in zip(row_a, row_e):
            assert a == pytest.approx(e, rel=rel, abs=abs_)


def test_matrix_construction_copies_input():
    source = [[1, 2], [3, 4]]
    matrix = Matrix(source)
    source[0][0] = 99
    assert matrix.to_list() == [[1.0, 2.0], [3.0, 4.0]]


@pytest.mark.parametrize("data", [[], [[]], [[1, 2], [3]], [[1], [2, 3]]])
def test_matrix_rejects_empty_or_ragged_data(data):
    with pytest.raises(ValueError):
        Matrix(data)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_matrix_rejects_non_finite_values(value):
    with pytest.raises(ValueError):
        Matrix([[value]])


def test_shape_and_indexing():
    matrix = Matrix([[1, 2, 3], [4, 5, 6]])
    assert matrix.shape == (2, 3)
    assert matrix[0] == [1.0, 2.0, 3.0]
    assert matrix[1, 2] == 6.0
    matrix[0, 1] = 20
    matrix[1] = [7, 8, 9]
    assert matrix.to_list() == [[1.0, 20.0, 3.0], [7.0, 8.0, 9.0]]


def test_setitem_rejects_bad_row_and_non_finite_element():
    matrix = Matrix([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        matrix[0] = [1]
    with pytest.raises(ValueError):
        matrix[0, 0] = float("nan")
    with pytest.raises(ValueError):
        matrix[0] = [1, float("inf")]


def test_swap_rows():
    matrix = Matrix([[1, 2], [3, 4]])
    matrix.swap_rows(0, 1)
    assert matrix.to_list() == [[3.0, 4.0], [1.0, 2.0]]
    matrix.swap_rows(0, 0)
    assert matrix.to_list() == [[3.0, 4.0], [1.0, 2.0]]


def test_scale_row():
    matrix = Matrix([[1, -2], [3, 4]])
    matrix.scale_row(0, -2)
    assert matrix.to_list() == [[-2.0, 4.0], [3.0, 4.0]]
    with pytest.raises(ValueError):
        matrix.scale_row(0, float("inf"))


def test_add_row_to_and_self_update():
    matrix = Matrix([[1, 2], [3, 4]])
    matrix.add_row_to(0, 1, 2)
    assert matrix.to_list() == [[7.0, 10.0], [3.0, 4.0]]
    matrix.add_row_to(0, 0, 1)
    assert matrix.to_list() == [[14.0, 20.0], [3.0, 4.0]]


def test_add_row_to_rejects_non_finite_factor():
    matrix = Matrix([[1, 2], [3, 4]])
    with pytest.raises(ValueError):
        matrix.add_row_to(0, 1, float("nan"))


def test_transpose_and_double_transpose():
    matrix = Matrix([[1, 2, 3], [4, 5, 6]])
    assert matrix.T.to_list() == [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
    assert matrix.T.T.to_list() == matrix.to_list()


def test_matrix_multiplication():
    a = Matrix([[1, 2, 3], [4, 5, 6]])
    b = Matrix([[7, 8], [9, 10], [11, 12]])
    expected = [[58, 64], [139, 154]]
    approx_matrix((a @ b).to_list(), expected)
    approx_matrix(matmul(a.to_list(), b.to_list()), expected)


def test_matrix_multiplication_dimension_mismatch():
    with pytest.raises(ValueError):
        _ = Matrix([[1, 2]]) @ Matrix([[1, 2]])
    with pytest.raises(ValueError):
        matmul([[1, 2]], [[1, 2]])


@pytest.mark.parametrize(
    "data, expected",
    [
        ([[5]], 5),
        ([[1, 2], [3, 4]], -2),
        ([[0, 1], [2, 3]], -2),
        ([[1, 2], [2, 4]], 0),
        ([[2, 0, 0], [0, 3, 0], [0, 0, 4]], 24),
    ],
)
def test_determinant(data, expected):
    assert Matrix(data).det() == pytest.approx(expected)


def test_determinant_rejects_non_square():
    with pytest.raises(ValueError):
        Matrix([[1, 2, 3], [4, 5, 6]]).det()


def test_to_list_returns_independent_copy():
    matrix = Matrix([[1, 2], [3, 4]])
    copied = matrix.to_list()
    copied[0][0] = 999
    assert matrix[0, 0] == 1.0


def test_repr_contains_matrix_data():
    assert "Matrix" in repr(Matrix([[1, 2]]))
