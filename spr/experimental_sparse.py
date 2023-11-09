# Pulled from the tutorial at:
# https://jax.readthedocs.io/en/latest/jax.experimental.sparse.html

from jax.experimental import sparse
import jax.numpy as jnp
import numpy as np

from jax import grad, jit

import functools
from sklearn.datasets import make_classification

from jax.scipy import optimize


def spr_tx(M, v):
    return 2 * jnp.dot(jnp.log1p(M.T), v) + 1


def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)


def y_model(params, X):
    return sigmoid(jnp.dot(X, params[1:]) + params[0])


def loss(params, X, y):
    y_hat = y_model(params, X)
    return -jnp.mean(y * jnp.log(y_hat) + (1 - y) * jnp.log(1 - y_hat))


def fit_logreg(X, y):
    params = jnp.zeros(X.shape[1] + 1)
    result = optimize.minimize(
        functools.partial(loss, X=X, y=y), x0=params, method="BFGS"
    )
    return result.x


if __name__ == "__main__":
    M = jnp.array([[0.0, 1.0, 0.0, 2.0], [3.0, 0.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0]])
    M_sp = sparse.BCOO.fromdense(M)
    print(M_sp)
    print(M_sp.todense())
    print(M_sp.data)
    print(M_sp.indices)
    print(M_sp.ndim)
    print(M_sp.shape)
    print(M_sp.dtype)
    print(M_sp.nse)  # nmber of specified elements

    y = jnp.array([3.0, 6.0, 5.0])

    P_sp = M_sp.T @ y
    P = M.T @ y

    print(P_sp)
    print(P)

    def f(y):
        return (M_sp.T @ y).sum()

    J = jit(grad(f))(y)

    f_sp = sparse.sparsify(spr_tx)
    Q_sp = f_sp(M_sp, y)
    print(Q_sp)

    X, y = make_classification(n_classes=2, random_state=1701)
    params_dense = fit_logreg(X, y)
    print(params_dense)

    Xsp = sparse.BCOO.fromdense(X)  # Sparse version of the input

    fit_logreg_sp = sparse.sparsify(fit_logreg)  # Sparse-transformed fit function
    params_sparse = fit_logreg_sp(Xsp, y)
    print(params_sparse)
