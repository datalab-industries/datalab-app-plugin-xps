"""Utility functions for XPS data processing."""

import numpy as np


def shirley_background(x, y, tol=1e-5, max_iter=100):
    """Compute iterative Shirley background for XPS data.

    Parameters:
        x: Binding energy array.
        y: Intensity array.
        tol: Convergence tolerance.
        max_iter: Maximum number of iterations.

    Returns:
        background: The Shirley background array.
    """
    x = np.array(x)
    y = np.array(y)

    background = np.linspace(y[0], y[-1], len(y))

    for _ in range(max_iter):
        prev = background.copy()

        diff = y - background
        integral = np.cumsum(diff[::-1])[::-1]

        background = y[-1] + (y[0] - y[-1]) * integral / integral[0]

        if np.linalg.norm(background - prev) < tol:
            break

    return background
