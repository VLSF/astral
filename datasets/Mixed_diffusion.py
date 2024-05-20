import jax.numpy as jnp
import sympy as sp
import itertools
import argparse
from scipy.special import roots_legendre
from jax import random, vmap


def np_random_sin(x, amplitudes, k=2):
    N_terms = amplitudes.shape[0]
    frequencies = sorted([*itertools.product(range(1, N_terms + 1), repeat=2)], key=lambda x: sum(x))[:N_terms]
    a = 0
    for n, (i, j) in enumerate(frequencies):
        a += amplitudes[n] * sp.sin(sp.pi * i * x[0]) * sp.sin(sp.pi * j * x[1]) / (
                1 + (sp.pi * i) ** 2 + (sp.pi * j) ** 2) ** k
    return a


def get_coordinates():
    N_mesh = 64
    coords_train = jnp.linspace(0, 1, N_mesh)
    coords_train = jnp.stack(jnp.meshgrid(coords_train, coords_train), 2)
    coords_train = coords_train.reshape(-1, 2)

    N_mesh_eval = 200

    x_, weights_ = roots_legendre(N_mesh_eval)
    x_ = jnp.array((x_ + 1) / 2)
    weights_ = jnp.array(weights_).reshape(1, -1)
    coords_legendre = jnp.stack(jnp.meshgrid(x_, x_), 2)
    coords_legendre = coords_legendre.reshape(-1, 2)

    coords_eval = jnp.linspace(0, 1, N_mesh_eval)
    coords_eval = jnp.stack(jnp.meshgrid(coords_eval, coords_eval), 2)
    coords_eval = coords_eval.reshape(-1, 2)

    return coords_train, weights_, coords_legendre, coords_eval


def get_sample(key, coords_train, weights_, coords_legendre, coords_eval):
    keys = random.split(key)
    N_terms = 20
    amplitudes = random.normal(keys[0], (N_terms,))
    amplitudes_ = random.normal(keys[1], (N_terms,))

    x, y = sp.Symbol("x"), sp.Symbol("y")

    a_ = np_random_sin([x, y], amplitudes, k=1)
    A_ = sp.lambdify([x, y], a_, 'jax')
    A = lambda x: A_(x[0], x[1])

    N_mesh = 200
    x_ = jnp.linspace(0, 1, N_mesh)
    coords = jnp.stack(jnp.meshgrid(x_, x_), 2)
    val_A = A(coords.T)
    min_a, max_a = jnp.min(val_A).item(), jnp.max(val_A).item()

    a = (5 * (a_ - min_a) / (max_a - min_a) + 1)
    dx_a = a.diff("x")
    dy_a = a.diff("y")

    phi = sp.sin(sp.pi * x) * sp.sin(sp.pi * y) * np_random_sin([x, y], amplitudes_, k=1)
    f = - (a * phi.diff("x")).diff("x") - (a * phi.diff("y")).diff("y") - eps * (
            (a * phi.diff("y")).diff("x") + (a * phi.diff("x")).diff("y"))

    rhs_ = sp.lambdify([x, y], f, 'jax')
    a_ = sp.lambdify([x, y], a, 'jax')
    dx_a_ = sp.lambdify([x, y], dx_a, 'jax')
    dy_a_ = sp.lambdify([x, y], dy_a, 'jax')

    sol_ = sp.lambdify([x, y], phi, 'jax')
    dx_sol_ = sp.lambdify([x, y], phi.diff('x'), 'jax')
    dy_sol_ = sp.lambdify([x, y], phi.diff('y'), 'jax')

    rhs = lambda x: rhs_(x[0], x[1])
    a = lambda x: a_(x[0], x[1])
    dx_a = lambda x: dx_a_(x[0], x[1])
    dy_a = lambda x: dy_a_(x[0], x[1])
    sol = lambda x: sol_(x[0], x[1])
    dx_sol = lambda x: dx_sol_(x[0], x[1])
    dy_sol = lambda x: dy_sol_(x[0], x[1])

    N_mesh = 200
    x = jnp.linspace(0, 1, N_mesh)
    coords = jnp.stack(jnp.meshgrid(x, x), 2).reshape(-1, 2)
    A = lambda x, y: jnp.stack([jnp.stack([a_(x, y), eps * a_(x, y)]),jnp.stack([eps * a_(x, y), a_(x, y)])]).reshape(2,2)
    A_ = vmap(A,(0,0))(coords[:, 0], coords[:, 1])
    A_min=jnp.min(jnp.linalg.eigvalsh(A_))
    C_F = 1 / (jnp.sqrt(A_min) * jnp.pi * 2)
    a_train = a(coords_train.T).T
    rhs_train = rhs(coords_train.T).T
    dx_a_train = dx_a(coords_train.T).T
    dy_a_train = dy_a(coords_train.T).T

    a_eval_legendre = a(coords_legendre.T).T
    dx_sol_eval_legendre = dx_sol(coords_legendre.T).T
    dy_sol_eval_legendre = dy_sol(coords_legendre.T).T
    rhs_legendre = rhs(coords_legendre.T).T
    a_legendre = a(coords_legendre.T).T

    sol_eval = sol(coords_eval.T).T

    return a_train, rhs_train, dx_a_train, dy_a_train, a_eval_legendre, dx_sol_eval_legendre, dy_sol_eval_legendre, rhs_legendre, a_legendre, sol_eval, C_F


def get_dataset(key, N_samples, eps):
    coords_train, weights_, coords_legendre, coords_eval = get_coordinates()
    a_train, rhs_train, dx_a_train, dy_a_train, a_eval_legendre, dx_sol_eval_legendre, dy_sol_eval_legendre, rhs_legendre, a_legendre, sol_eval, C_F = [], [], [], [], [], [], [], [], [], [], []
    keys = random.split(key, N_samples)
    for key in keys:
        a_train_, rhs_train_, dx_a_train_, dy_a_train_, a_eval_legendre_, dx_sol_eval_legendre_, dy_sol_eval_legendre_, rhs_legendre_, a_legendre_, sol_eval_, C_F_ = get_sample(
            key, coords_train, weights_, coords_legendre, coords_eval)
        a_train.append(a_train_)
        rhs_train.append(rhs_train_)
        dx_a_train.append(dx_a_train_)
        dy_a_train.append(dy_a_train_)
        a_eval_legendre.append(a_eval_legendre_)
        dx_sol_eval_legendre.append(dx_sol_eval_legendre_)
        dy_sol_eval_legendre.append(dy_sol_eval_legendre_)
        rhs_legendre.append(rhs_legendre_)
        a_legendre.append(a_legendre_)
        sol_eval.append(sol_eval_)
        C_F.append(C_F_)

    a_train = jnp.stack(a_train)
    rhs_train = jnp.stack(rhs_train)
    dx_a_train = jnp.stack(dx_a_train)
    dy_a_train = jnp.stack(dy_a_train)
    a_eval_legendre = jnp.stack(a_eval_legendre)
    dx_sol_eval_legendre = jnp.stack(dx_sol_eval_legendre)
    dy_sol_eval_legendre = jnp.stack(dy_sol_eval_legendre)
    rhs_legendre = jnp.stack(rhs_legendre)
    a_legendre = jnp.stack(a_legendre)
    sol_eval = jnp.stack(sol_eval)
    C_F = jnp.array(C_F)
    data = {
        "a_train": a_train,
        "rhs_train": rhs_train,
        "dx_a_train": dx_a_train,
        "dy_a_train": dy_a_train,
        "a_eval_legendre": a_eval_legendre,
        "dx_sol_eval_legendre": dx_sol_eval_legendre,
        "dy_sol_eval_legendre": dy_sol_eval_legendre,
        "rhs_legendre": rhs_legendre,
        "a_legendre": a_legendre,
        "sol_eval": sol_eval,
        "C_F": C_F,
        "coords_train": coords_train,
        "weights": weights_,
        "coords_legendre": coords_legendre,
        "coords_eval": coords_eval
    }
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PINN")
    parser.add_argument("--eps", type=float, default=0.5, help="eps")
    args = parser.parse_args()

    key = random.PRNGKey(23)
    N_samples = 100
    eps = args.eps
    data = get_dataset(key, N_samples, eps)
    jnp.savez(f"mixed_diffusion_{eps}.npz", **data)
