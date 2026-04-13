import numpy as np


def solve_heat_equation_1d(
    L: float = 1.0,
    N: int = 100,
    tf: float = 10.0,
    alpha: float = 1.0e-2,
    u_left: float = 10.0,
    u_right: float = 0.0,
) -> np.ndarray:
    dx = L / N
    dt = (dx**2) / (4 * alpha)
    nt = int(tf / dt) + 1

    u = np.empty((nt, N + 1), dtype=float)

    x = np.linspace(0, L, N + 1)
    u0 = 5 + 5 * np.sin(3 * np.pi * x / L)

    u[0, :] = u0
    u[:, 0] = u_left
    u[:, -1] = u_right

    coeff = alpha * dt / dx**2

    for ti in range(nt - 1):
        for ni in range(1, N):
            u[ti + 1, ni] = (
                u[ti, ni]
                + coeff * (u[ti, ni + 1] - 2 * u[ti, ni] + u[ti, ni - 1])
            )

    return u