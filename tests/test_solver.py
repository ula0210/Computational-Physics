import numpy as np

from src.heat_equation_1d.solver import solve_heat_equation_1d


def test_solver_returns_numpy_array():
    u = solve_heat_equation_1d(tf=0.1)
    assert isinstance(u, np.ndarray)


def test_solver_returns_correct_shape():
    L = 1.0
    N = 20
    tf = 0.1
    alpha = 1.0e-2
    dx = L / N
    dt = (dx**2) / (4 * alpha)
    nt = int(tf / dt) + 1

    u = solve_heat_equation_1d(L=L, N=N, tf=tf, alpha=alpha)
    assert u.shape == (nt, N + 1)


def test_boundary_conditions_are_preserved():
    u = solve_heat_equation_1d(tf=0.1, u_left=10.0, u_right=0.0)
    assert np.allclose(u[:, 0], 10.0)
    assert np.allclose(u[:, -1], 0.0)


def test_initial_condition_matches_boundaries():
    u = solve_heat_equation_1d(tf=0.1, u_left=10.0, u_right=0.0)
    assert u[0, 0] == 10.0
    assert u[0, -1] == 0.0