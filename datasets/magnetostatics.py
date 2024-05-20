import jax.numpy as jnp
import sympy as sp
import itertools

from scipy.special import roots_legendre
from jax import random

def np_random_cos(x, amplitudes, k = 2):
    N_terms = amplitudes.shape[0]
    frequencies = sorted([*itertools.product(range(N_terms), repeat=2)], key=lambda x: sum(x))[:N_terms]
    a = 0
    for n, (i, j) in enumerate(frequencies):
        a += amplitudes[n]*sp.cos(sp.pi*i*x[0])*sp.cos(sp.pi*j*x[1]) / (1 + (sp.pi*i)**2 + (sp.pi*j)**2)**k
    return a

def get_coordinates():
    N_mesh = 64
    coords_train = jnp.linspace(0, 1, N_mesh)
    coords_train = jnp.stack(jnp.meshgrid(coords_train, coords_train), 2)
    coords_train = coords_train.reshape(-1, 2)

    N_mesh_eval = 200

    x_, weights_ = roots_legendre(N_mesh_eval)
    x_ = jnp.array((x_ + 1)/2)
    weights_ = jnp.array(weights_).reshape(1, -1)
    coords_legendre = jnp.stack(jnp.meshgrid(x_, x_), 2)
    coords_legendre = coords_legendre.reshape(-1, 2)

    coords_eval = jnp.linspace(0, 1, N_mesh_eval)
    coords_eval = jnp.stack(jnp.meshgrid(coords_eval, coords_eval), 2)
    coords_eval = coords_eval.reshape(-1, 2)

    return coords_train, weights_, coords_legendre, coords_eval

def get_sample(key, coords_train, weights_, coords_legendre, coords_eval):
    keys = random.split(key, 3)
    N_terms = 20
    amplitudes = random.normal(keys[0], (N_terms,))
    amplitudes_x = random.normal(keys[1], (N_terms,))
    amplitudes_y = random.normal(keys[2], (N_terms,))

    x, y = sp.Symbol("x"), sp.Symbol("y")

    mu_ = np_random_cos([x, y], amplitudes, k = 1)
    MU_ = sp.lambdify([x, y], mu_, 'jax')
    MU = lambda x: MU_(x[0], x[1])

    N_mesh = 200
    x_ = jnp.linspace(0, 1, N_mesh)
    coords = jnp.stack(jnp.meshgrid(x_, x_), 2)
    val_MU = MU(coords.T)
    min_MU, max_MU = jnp.min(val_MU).item(), jnp.max(val_MU).item()

    mu = (5 * (mu_ - min_MU) / (max_MU - min_MU) + 1)
    dx_mu = mu.diff("x")
    dy_mu = mu.diff("y")

    phi__ = np_random_cos([x, y], amplitudes_x, k = 1)
    phi_x = phi__.diff("y")
    phi_y = -phi__.diff("x")

    f_x = (mu * (phi_y.diff("x") - phi_x.diff("y"))).diff("y")
    f_y = -(mu * (phi_y.diff("x") - phi_x.diff("y"))).diff("x")

    f_x_ = sp.lambdify([x, y], f_x, 'jax')
    f_y_ = sp.lambdify([x, y], f_y, 'jax')
    mu_ = sp.lambdify([x, y], mu, 'jax')
    dx_mu_ = sp.lambdify([x, y], dx_mu, 'jax')
    dy_mu_ = sp.lambdify([x, y], dy_mu, 'jax')

    sol_x_ = sp.lambdify([x, y], phi_x, 'jax')
    sol_y_ = sp.lambdify([x, y], phi_y, 'jax')
    dx_sol_y_ = sp.lambdify([x, y], phi_y.diff('x'), 'jax')
    dy_sol_x_ = sp.lambdify([x, y], phi_x.diff('y'), 'jax')

    f_x = lambda x: f_x_(x[0], x[1])
    f_y = lambda x: f_y_(x[0], x[1])
    mu = lambda x: mu_(x[0], x[1])
    dx_mu = lambda x: dx_mu_(x[0], x[1])
    dy_mu = lambda x: dy_mu_(x[0], x[1])

    sol_x = lambda x: sol_x_(x[0], x[1])
    sol_y = lambda x: sol_y_(x[0], x[1])
    dx_sol_y = lambda x: dx_sol_y_(x[0], x[1])
    dy_sol_x = lambda x: dy_sol_x_(x[0], x[1])

    mu_train = mu(coords_train.T).T
    f_x_train = f_x(coords_train.T).T
    f_y_train = f_y(coords_train.T).T
    dx_mu_train = dx_mu(coords_train.T).T
    dy_mu_train = dy_mu(coords_train.T).T

    dx_sol_y_legendre = dx_sol_y(coords_legendre.T).T
    dy_sol_x_legendre = dy_sol_x(coords_legendre.T).T
    sol_x_legendre = sol_x(coords_legendre.T).T
    sol_y_legendre = sol_y(coords_legendre.T).T
    f_x_legendre = f_x(coords_legendre.T).T
    f_y_legendre = f_y(coords_legendre.T).T
    mu_legendre = mu(coords_legendre.T).T

    sol_x_eval = sol_x(coords_eval.T).T
    sol_y_eval = sol_y(coords_eval.T).T

    return mu_train, f_x_train, f_y_train, dx_mu_train, dy_mu_train, dx_sol_y_legendre, dy_sol_x_legendre, sol_x_legendre, sol_y_legendre, f_x_legendre, f_y_legendre, mu_legendre, sol_x_eval, sol_y_eval

def get_dataset(key, N_samples):
    coords_train, weights_, coords_legendre, coords_eval = get_coordinates()
    mu_train, f_x_train, f_y_train, dx_mu_train, dy_mu_train, dx_sol_y_legendre, dy_sol_x_legendre, sol_x_legendre, sol_y_legendre, f_x_legendre, f_y_legendre, mu_legendre, sol_x_eval, sol_y_eval = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    keys = random.split(key, N_samples)
    for key in keys:
        mu_train_, f_x_train_, f_y_train_, dx_mu_train_, dy_mu_train_, dx_sol_y_legendre_, dy_sol_x_legendre_, sol_x_legendre_, sol_y_legendre_, f_x_legendre_, f_y_legendre_, mu_legendre_, sol_x_eval_, sol_y_eval_ = get_sample(key, coords_train, weights_, coords_legendre, coords_eval)
        mu_train.append(mu_train_)
        f_x_train.append(f_x_train_)
        f_y_train.append(f_y_train_)
        dx_mu_train.append(dx_mu_train_)
        dy_mu_train.append(dy_mu_train_)
        dx_sol_y_legendre.append(dx_sol_y_legendre_)
        dy_sol_x_legendre.append(dy_sol_x_legendre_)
        sol_x_legendre.append(sol_x_legendre_)
        sol_y_legendre.append(sol_y_legendre_)
        f_x_legendre.append(f_x_legendre_)
        f_y_legendre.append(f_y_legendre_)
        mu_legendre.append(mu_legendre_)
        sol_x_eval.append(sol_x_eval_)
        sol_y_eval.append(sol_y_eval_)

    mu_train = jnp.stack(mu_train)
    f_x_train = jnp.stack(f_x_train)
    f_y_train = jnp.stack(f_y_train)
    dx_mu_train = jnp.stack(dx_mu_train)
    dy_mu_train = jnp.stack(dy_mu_train)
    dx_sol_y_legendre = jnp.stack(dx_sol_y_legendre)
    dy_sol_x_legendre = jnp.stack(dy_sol_x_legendre)
    sol_x_legendre = jnp.stack(sol_x_legendre)
    sol_y_legendre = jnp.stack(sol_y_legendre)
    f_x_legendre = jnp.stack(f_x_legendre)
    f_y_legendre = jnp.stack(f_y_legendre)
    mu_legendre = jnp.stack(mu_legendre)
    sol_x_eval = jnp.stack(sol_x_eval)
    sol_y_eval = jnp.stack(sol_y_eval)

    data = {
        "mu_train": mu_train,
        "f_x_train": f_x_train,
        "f_y_train": f_y_train,
        "dx_mu_train": dx_mu_train,
        "dy_mu_train": dy_mu_train,
        "dx_sol_y_legendre": dx_sol_y_legendre,
        "dy_sol_x_legendre": dy_sol_x_legendre,
        "sol_x_legendre": sol_x_legendre,
        "sol_y_legendre": sol_y_legendre,
        "f_x_legendre": f_x_legendre,
        "f_y_legendre": f_y_legendre,
        "mu_legendre": mu_legendre,
        "sol_x_eval": sol_x_eval,
        "sol_y_eval": sol_y_eval,
        "coords_train": coords_train,
        "weights": weights_,
        "coords_legendre": coords_legendre,
        "coords_eval": coords_eval
    }

    return data

if __name__ == "__main__":
    key = random.PRNGKey(23)
    N_samples = 1000
    data = get_dataset(key, N_samples)
    jnp.savez("magnetostatics.npz", **data)