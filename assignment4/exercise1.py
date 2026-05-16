from scipy.optimize import minimize
import numpy as np
from jax import grad, jit, hessian, config

config.update("jax_enable_x64", True)
np.random.seed(42)

A = 1
B = 1
C = 1
X_0 = np.random.randn(2)


def f(x) -> float:
    return A * (x[1] - x[0] ** 2) ** 2 + (B - x[0]) ** 2


def constraint(x):
    return x[0] ** 2 + x[1] ** 2 - C


def no_gradients():
    return minimize(
        f, X_0, method="trust-constr", constraints=[{"type": "eq", "fun": constraint}]
    )


def with_gradients():
    h = jit(hessian(constraint))
    return minimize(
        f,
        X_0,
        method="trust-constr",
        jac=jit(grad(f)),
        hess=jit(hessian(f)),
        constraints=[
            {
                "type": "eq",
                "fun": constraint,
                "jac": jit(grad(constraint)),
                "hess": lambda x, v: v[0] * h(x),
            }
        ],
    )


def print_results(res, header: str):
    print(header)
    print("Number of iterations: ", res.nit)
    print("Number of function evaluations: ", res.nfev)
    print("x1: ", res.x[0])
    print("x2: ", res.x[1])
    print("Constrained: ", np.isclose(constraint(res.x), 0.0))


print_results(no_gradients(), "-- Minimization without gradients --")
print_results(with_gradients(), "-- Minimization with gradients --")
