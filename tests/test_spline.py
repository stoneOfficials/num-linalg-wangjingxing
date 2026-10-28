import math
import pytest

from src.spline import natural_cubic_spline, eval_spline


def derivative(segment, t):
    _, b, c, d = segment
    return b + 2.0 * c * t + 3.0 * d * t * t


def second_derivative(segment, t):
    _, _, c, d = segment
    return 2.0 * c + 6.0 * d * t


def test_two_nodes_produce_linear_spline():
    xs = [0.0, 2.0]
    ys = [1.0, 5.0]
    segments = natural_cubic_spline(xs, ys)
    assert len(segments) == 1
    assert eval_spline(segments, xs, 0.0) == pytest.approx(1.0)
    assert eval_spline(segments, xs, 1.0) == pytest.approx(3.0)
    assert eval_spline(segments, xs, 2.0) == pytest.approx(5.0)
    assert segments[0][2] == pytest.approx(0.0)
    assert segments[0][3] == pytest.approx(0.0)


def test_spline_interpolates_all_nodes():
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [0.0, 1.0, 0.0, 1.0]
    segments = natural_cubic_spline(xs, ys)
    for x, y in zip(xs, ys):
        assert eval_spline(segments, xs, x) == pytest.approx(y, abs=1e-9)


def test_linear_data_remains_linear():
    xs = [-1.0, 0.0, 1.0, 2.0]
    ys = [2.0 * x + 3.0 for x in xs]
    segments = natural_cubic_spline(xs, ys)
    for x in [-1.0, -0.5, 0.5, 1.5, 2.0]:
        assert eval_spline(segments, xs, x) == pytest.approx(2.0 * x + 3.0, abs=1e-9)


@pytest.mark.parametrize("xs, ys", [
    ([0.0], [1.0]),
    ([0.0, 0.0], [1.0, 2.0]),
    ([1.0, 0.0], [1.0, 2.0]),
    ([0.0, 1.0], [1.0]),
])
def test_spline_rejects_invalid_nodes(xs, ys):
    with pytest.raises(ValueError):
        natural_cubic_spline(xs, ys)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_spline_rejects_non_finite_nodes(value):
    with pytest.raises(ValueError):
        natural_cubic_spline([0.0, 1.0], [0.0, value])


def test_natural_boundary_conditions():
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [0.0, 1.0, -1.0, 2.0]
    segments = natural_cubic_spline(xs, ys)
    assert second_derivative(segments[0], 0.0) == pytest.approx(0.0, abs=1e-9)
    last_t = xs[-1] - xs[-2]
    assert second_derivative(segments[-1], last_t) == pytest.approx(0.0, abs=1e-9)


def test_first_and_second_derivative_continuity():
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [0.0, 2.0, -1.0, 1.0]
    segments = natural_cubic_spline(xs, ys)
    for i in range(len(segments) - 1):
        t_left = xs[i + 1] - xs[i]
        assert derivative(segments[i], t_left) == pytest.approx(
            derivative(segments[i + 1], 0.0), abs=1e-8
        )
        assert second_derivative(segments[i], t_left) == pytest.approx(
            second_derivative(segments[i + 1], 0.0), abs=1e-8
        )


def test_eval_spline_extrapolates_using_end_segments():
    xs = [0.0, 1.0, 2.0]
    ys = [0.0, 1.0, 0.0]
    segments = natural_cubic_spline(xs, ys)
    left_expected = segments[0][0] + segments[0][1] * (-1.0) + segments[0][2] + segments[0][3] * (-1.0) ** 3
    right_t = 2.0
    right_expected = ((segments[-1][3] * right_t + segments[-1][2]) * right_t + segments[-1][1]) * right_t + segments[-1][0]
    assert eval_spline(segments, xs, -1.0) == pytest.approx(left_expected)
    assert eval_spline(segments, xs, 3.0) == pytest.approx(right_expected)


def test_eval_spline_rejects_wrong_segment_count():
    with pytest.raises(ValueError):
        eval_spline([], [0.0, 1.0], 0.5)


def test_eval_spline_rejects_non_finite_x():
    segments = natural_cubic_spline([0.0, 1.0], [0.0, 1.0])
    with pytest.raises(ValueError):
        eval_spline(segments, [0.0, 1.0], math.inf)
