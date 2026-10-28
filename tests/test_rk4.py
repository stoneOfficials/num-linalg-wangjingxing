import math
import pytest

from src.rk4 import rk4_step, rk4_solve


def test_rk4_step_constant_derivative():
    assert rk4_step(lambda x, y: 2.0, 0.0, 1.0, 0.25) == pytest.approx(1.5)


def test_rk4_step_exponential_one_step():
    result = rk4_step(lambda x, y: y, 0.0, 1.0, 0.1)
    assert result == pytest.approx(math.exp(0.1), rel=1e-5)


def test_rk4_step_negative_step():
    result = rk4_step(lambda x, y: 1.0, 1.0, 2.0, -0.5)
    assert result == pytest.approx(1.5)


def test_rk4_step_rejects_zero_step():
    with pytest.raises(ValueError):
        rk4_step(lambda x, y: y, 0, 1, 0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_rk4_step_rejects_non_finite_inputs(value):
    with pytest.raises(ValueError):
        rk4_step(lambda x, y: y, value, 1, 0.1)


def test_rk4_step_rejects_non_finite_derivative():
    with pytest.raises(ArithmeticError):
        rk4_step(lambda x, y: float("nan"), 0, 1, 0.1)


def test_rk4_solve_matches_exponential_solution():
    xs, ys = rk4_solve(lambda x, y: y, 0.0, 1.0, 1.0, 0.1)
    assert xs[0] == 0.0
    assert xs[-1] == 1.0
    assert ys[-1] == pytest.approx(math.e, rel=1e-5)
    assert len(xs) == len(ys)


def test_rk4_solve_backward_integration():
    xs, ys = rk4_solve(lambda x, y: 1.0, 1.0, 2.0, 0.0, 0.2)
    assert xs[0] == 1.0
    assert xs[-1] == 0.0
    assert ys[-1] == pytest.approx(1.0)


def test_rk4_solve_same_start_and_end():
    xs, ys = rk4_solve(lambda x, y: y, 2.0, 3.0, 2.0, 0.1)
    assert xs == [2.0]
    assert ys == [3.0]


def test_rk4_solve_final_short_step():
    xs, ys = rk4_solve(lambda x, y: 1.0, 0.0, 0.0, 1.0, 0.3)
    assert xs[-1] == 1.0
    assert ys[-1] == pytest.approx(1.0)
    assert all(b > a for a, b in zip(xs, xs[1:]))


def test_rk4_solve_rejects_zero_step():
    with pytest.raises(ValueError):
        rk4_solve(lambda x, y: y, 0, 1, 1, 0)


def test_rk4_fourth_order_convergence_trend():
    f = lambda x, y: y
    exact = math.e
    _, y1 = rk4_solve(f, 0.0, 1.0, 1.0, 0.2)
    _, y2 = rk4_solve(f, 0.0, 1.0, 1.0, 0.1)
    err1 = abs(y1[-1] - exact)
    err2 = abs(y2[-1] - exact)
    assert err1 / err2 > 10.0
