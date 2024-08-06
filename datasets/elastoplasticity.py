import jax.numpy as jnp
import itertools
import sys

from jax import random, vmap, grad
from scipy.special import roots_legendre
from jax.tree_util import tree_map
from collections import defaultdict

def gamma(u, mu, k_star, delta):
    t0 = k_star/(2*jnp.sqrt(mu))
    return (u > t0) * ((2*mu-delta)*t0/jnp.abs(u)+delta) + (u <= t0)*2*mu

def sigma_11(u1, u2, x, K_0, mu, k_star, delta):
    du1 = grad(u1)(x)
    du2 = grad(u2)(x)
    eps_D_F = jnp.sqrt(jnp.abs((du1[0] - du2[1])**2 / 2 + (du1[1] + du2[0])**2 / 2))
    return K_0*(du1[0] + du2[1]) + gamma(eps_D_F, mu, k_star, delta)*(du1[0] - du2[1]) / 2

def sigma_22(u1, u2, x, K_0, mu, k_star, delta):
    du1 = grad(u1)(x)
    du2 = grad(u2)(x)
    eps_D_F = jnp.sqrt(jnp.abs((du1[0] - du2[1])**2 / 2 + (du1[1] + du2[0])**2 / 2))
    return K_0*(du1[0] + du2[1]) + gamma(eps_D_F, mu, k_star, delta)*(du2[1] - du1[0]) / 2

def sigma_12(u1, u2, x, K_0, mu, k_star, delta):
    du1 = grad(u1)(x)
    du2 = grad(u2)(x)
    eps_D_F = jnp.sqrt(jnp.abs((du1[0] - du2[1])**2 / 2 + (du1[1] + du2[0])**2 / 2))
    return gamma(eps_D_F, mu, k_star, delta)*(du1[1] + du2[0]) / 2

def random_sin(x, amplitudes, ind, k, A):
    return A*jnp.sum(amplitudes*jnp.sin(jnp.pi*ind[:, 0]*x[0])*jnp.sin(jnp.pi*ind[:, 1]*x[1]) / (1 + jnp.sum((jnp.pi*ind)**2, 1))**k)

def get_f(x, amplitudes1, amplitudes2, ind, k, A, K_0, mu, k_star, delta):
    u1 = lambda x: random_sin(x, amplitudes1, ind, k, A)
    u2 = lambda x: random_sin(x, amplitudes2, ind, k, A)
    d1_sigma_11 = vmap(grad(sigma_11, argnums=2), in_axes=(None, None, 0, None, None, None, None))(u1, u2, x, K_0, mu, k_star, delta)[:, 0]
    d_sigma_12 = vmap(grad(sigma_12, argnums=2), in_axes=(None, None, 0, None, None, None, None))(u1, u2, x, K_0, mu, k_star, delta)
    d2_sigma_22 = vmap(grad(sigma_22, argnums=2), in_axes=(None, None, 0, None, None, None, None))(u1, u2, x, K_0, mu, k_star, delta)[:, 1]
    return -(d1_sigma_11 + d_sigma_12[:, 1]), -(d2_sigma_22 + d_sigma_12[:, 0])

def get_sigma(x, amplitudes1, amplitudes2, ind, k, A, K_0, mu, k_star, delta):
    u1 = lambda x: random_sin(x, amplitudes1, ind, k, A)
    u2 = lambda x: random_sin(x, amplitudes2, ind, k, A)
    sigma_11_ = vmap(sigma_11, in_axes=(None, None, 0, None, None, None, None))(u1, u2, x, K_0, mu, k_star, delta)
    sigma_12_ = vmap(sigma_12, in_axes=(None, None, 0, None, None, None, None))(u1, u2, x, K_0, mu, k_star, delta)
    sigma_22_ = vmap(sigma_22, in_axes=(None, None, 0, None, None, None, None))(u1, u2, x, K_0, mu, k_star, delta)
    return sigma_11_, sigma_12_, sigma_22_

def get_sample(key, coords, K_0, mu, k_star, delta, A, k, N_terms):
    keys = random.split(key, 2)
    ind = jnp.array(sorted([*itertools.product(range(1, N_terms+1), repeat=2)], key=lambda x: sum(x))[:N_terms])
    amplitudes1 = random.normal(keys[0], (N_terms, ))
    amplitudes2 = random.normal(keys[1], (N_terms, ))
    u1 = vmap(random_sin, in_axes=(0, None, None, None, None))(coords, amplitudes1, ind, k, A)
    u2 = vmap(random_sin, in_axes=(0, None, None, None, None))(coords, amplitudes2, ind, k, A)
    du1 = vmap(grad(random_sin), in_axes=(0, None, None, None, None))(coords, amplitudes1, ind, k, A)
    du2 = vmap(grad(random_sin), in_axes=(0, None, None, None, None))(coords, amplitudes2, ind, k, A)
    f1, f2 = get_f(coords, amplitudes1, amplitudes2, ind, k, A, K_0, mu, k_star, delta)
    sigma_11_, sigma_12_, sigma_22_ = get_sigma(coords, amplitudes1, amplitudes2, ind, k, A, K_0, mu, k_star, delta)
    data = {
        "u1": u1,
        "u2": u2,
        "du1": du1,
        "du2": du2,
        "f1": f1,
        "f2": f2,
        "sigma_11": sigma_11_,
        "sigma_12": sigma_12_,
        "sigma_22": sigma_22_,
    }
    return data

def get_alpha_1_star(K_0, mu):
    M = jnp.array([
        [1/(4*K_0) + 1/(4*mu), 1/(4*K_0) - 1/(4*mu), 0],
        [1/(4*K_0) - 1/(4*mu), 1/(4*K_0) + 1/(4*mu), 0],
        [0, 0, 1/mu]
    ])
    return jnp.min(jnp.linalg.eigvalsh(M))

def get_C0(K_0, mu, delta):
    mu2 = (2*mu - delta)/(4*mu*delta)
    alpha_1_star = get_alpha_1_star(K_0, mu)
    C0 = 1 + 2*mu2 / alpha_1_star
    return C0

def get_dataset(key, N_samples, K_0, mu, k_star, delta, N_terms, A, k):
    keys = random.split(key, N_samples)
    data = defaultdict(list)
    data["C0"] = get_C0(K_0, mu, delta)
    data["K_0"] = K_0
    data["k_star"] = k_star
    data["mu"] = mu
    data["delta"] = delta
    data["N_terms"] = N_terms
    data["A"] = A
    data["k"] = k

    N_x_train = 64
    N_x_test = 2*N_x_train

    x = jnp.linspace(0, 1, N_x_train+2)[1:-1]
    coords_train = jnp.stack(jnp.meshgrid(x, x), 2).reshape(-1, 2)
    data["coords_train"] = coords_train
    
    x, weights = roots_legendre(N_x_test)
    x = jnp.array((x + 1)/2)
    weights = jnp.array(weights).reshape(1, -1)
    weights = weights / 2
    coords_leg = jnp.stack(jnp.meshgrid(x, x), 2).reshape(-1, 2)
    data["coords_leg"] = coords_leg
    data["weights"] = weights

    x = jnp.linspace(0, 1, N_x_test+2)[1:-1]
    coords_test = jnp.stack(jnp.meshgrid(x, x), 2).reshape(-1, 2)
    data["coords_test"] = coords_test
    
    for key in keys:
        data_train = get_sample(key, coords_train, K_0, mu, k_star, delta, A, k, N_terms)
        data_test = get_sample(key, coords_test, K_0, mu, k_star, delta, A, k, N_terms)
        data_leg = get_sample(key, coords_leg, K_0, mu, k_star, delta, A, k, N_terms)
        for k_ in data_train.keys():
            data[k_ + "_train"].append(data_train[k_])
            data[k_ + "_test"].append(data_test[k_])
            data[k_ + "_leg"].append(data_leg[k_])

    data = tree_map(lambda x: jnp.stack(x) if type(x) == list else x, data, is_leaf = lambda x: type(x) == list)
    return data

def dataset_1(N_samples):
    key = random.PRNGKey(33)
    K_0, mu, k_star, delta = 1.0, 1.0, 0.01, 1.0
    N_terms, A, k = 50, 1, 2
    return get_dataset(key, N_samples, K_0, mu, k_star, delta, N_terms, A, k)

def dataset_2(N_samples):
    key = random.PRNGKey(61)
    K_0, mu, k_star, delta = 1.0, 1.0, 20, 1.0
    N_terms, A, k = 50, 1, 0.1
    return get_dataset(key, N_samples, K_0, mu, k_star, delta, N_terms, A, k)

if __name__ == "__main__":
    dataset = sys.argv[1]
    N_samples = int(sys.argv[2])
    if dataset == "1":
        data = dataset_1(N_samples)
    else:
        data = dataset_2(N_samples)
    jnp.savez(f"elastoplasticity_{dataset}.npz", **data)