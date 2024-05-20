import jax.numpy as jnp
import itertools

from scipy.special import roots_legendre
from jax import random, vmap

def get_weights(N, L, s, p):
    weights = 1/(1 + (jnp.arange(N) * jnp.pi / L)**s)**p
    return weights

def draw_coefficients(key, weights):
    c = jnp.expand_dims(random.normal(key, weights.shape)*weights, 1)
    return c

def get_eigvalues(N, a):
    l = jnp.expand_dims((jnp.arange(N)*jnp.pi)**2 + a**2 / 4, 1)
    return l

def get_eigenvectors(N_x, N, a):
    x = jnp.expand_dims(jnp.linspace(0, 1, N_x), 1)
    k = jnp.expand_dims(jnp.arange(N)*jnp.pi, 0)
    psi = jnp.exp(x*a/2) * jnp.sin(k*x)
    return psi

def get_eigenvectors_leg(N_x, N, a):
    x = jnp.expand_dims((roots_legendre(N_x)[0] + 1)/2, 1)
    k = jnp.expand_dims(jnp.arange(N)*jnp.pi, 0)
    psi = jnp.exp(x*a/2) * jnp.sin(k*x)
    return psi

def get_ic(x, phi_k, k, a):
    return jnp.sum(jnp.exp(x*a/2) * jnp.sin(k*x) * phi_k)

def get_d_eigenvectors(N_x, N, a):
    x = jnp.expand_dims(jnp.linspace(0, 1, N_x), 1)
    k = jnp.expand_dims(jnp.arange(N)*jnp.pi, 0)
    d_psi = jnp.exp(x*a/2)* (a * jnp.sin(k*x) / 2 + k * jnp.cos(k*x))
    return d_psi

def get_d_eigenvectors_leg(N_x, N, a):
    x = jnp.expand_dims((roots_legendre(N_x)[0] + 1)/2, 1)
    k = jnp.expand_dims(jnp.arange(N)*jnp.pi, 0)
    d_psi = jnp.exp(x*a/2)* (a * jnp.sin(k*x) / 2 + k * jnp.cos(k*x))
    return d_psi

def get_PDE_data(key, t, weights, eigenvectors, d_eigenvectors, eigenvalues):
    # t.shape = (1, N_t)
    keys = random.split(key)
    f_k = draw_coefficients(keys[0], weights)
    phi_k = draw_coefficients(keys[1], weights)
    u_c = phi_k * jnp.exp(-eigenvalues*t) + f_k*t / (1 + eigenvalues*t)
    solution = eigenvectors @ u_c
    dx_solution = d_eigenvectors @ u_c
    f = eigenvectors @ f_k
    phi = eigenvectors @ phi_k
    return solution, dx_solution, f, phi, phi_k[:, 0]

def get_grid_data():
    N_x = 64
    N_t = 64
    N = 100
    T = 0.1

    t_train = jnp.linspace(0, T, N_t)
    x_train = jnp.linspace(0, 1, N_x)
    coords_train = jnp.stack(jnp.meshgrid(x_train, t_train, indexing="ij"), 2)

    t_validation = jnp.linspace(0, T, 2*N_t)
    x_validation = jnp.linspace(0, 1, 2*N_x)
    coords_validation = jnp.stack(jnp.meshgrid(x_validation, t_validation, indexing="ij"), 2)

    t_leg = T*(roots_legendre(2*N_x)[0] + 1)/2
    x_leg = (roots_legendre(2*N_x)[0] + 1)/2
    coords_leg = jnp.stack(jnp.meshgrid(x_leg, t_leg, indexing="ij"), 2)

    weights = jnp.expand_dims(roots_legendre(2*N_x)[1], 0)
    k = jnp.arange(N)*jnp.pi

    return coords_train, coords_validation, coords_leg, weights, k, T

def get_dataset(key, N_samples):
    N_x = 64
    N_t = 64
    N = 100
    L = 5
    s = 2
    p = 2
    T = 0.1
    keys = random.split(key, N_samples+1)
    a = random.uniform(keys[-1], (N_samples,), minval=0, maxval=10)

    weights = get_weights(N, L, s, p)
    eigenvalues = vmap(get_eigvalues, in_axes=(None, 0))(N, a)

    t_train = jnp.linspace(0, T, N_t)
    eigenvectors_train = vmap(get_eigenvectors, in_axes=(None, None, 0))(N_x, N, a)
    d_eigenvectors_train = vmap(get_d_eigenvectors, in_axes=(None, None, 0))(N_x, N, a)
    solution_train, dx_solution_train, f_train, phi_train, phi_k = vmap(get_PDE_data, in_axes=(0, None, None, 0, 0, 0))(keys[:-1], t_train, weights, eigenvectors_train, d_eigenvectors_train, eigenvalues)

    t_validation = jnp.linspace(0, T, 2*N_t)
    eigenvectors_validation =  vmap(get_eigenvectors, in_axes=(None, None, 0))(2*N_x, N, a)
    d_eigenvectors_validation =  vmap(get_d_eigenvectors, in_axes=(None, None, 0))(2*N_x, N, a)
    solution_validation, _, _, _, _ = vmap(get_PDE_data, in_axes=(0, None, None, 0, 0, 0))(keys[:-1], t_validation, weights, eigenvectors_validation, d_eigenvectors_validation, eigenvalues)

    t_leg = T*(roots_legendre(2*N_x)[0] + 1)/2
    eigenvectors_leg =  vmap(get_eigenvectors_leg, in_axes=(None, None, 0))(2*N_x, N, a)
    d_eigenvectors_leg =  vmap(get_d_eigenvectors_leg, in_axes=(None, None, 0))(2*N_x, N, a)
    solution_leg, dx_solution_leg, f_leg, _, _ = vmap(get_PDE_data, in_axes=(0, None, None, 0, 0, 0))(keys[:-1], t_leg, weights, eigenvectors_leg, d_eigenvectors_leg, eigenvalues)

    coords_train, coords_validation, coords_leg, weights, k, T = get_grid_data()

    data = {
         "solution_train": solution_train,
         "dx_solution_train": dx_solution_train,
         "f_train": f_train,
         "phi_train": phi_train,
         "phi_k": phi_k,
         "a": a,
         "solution_validation": solution_validation,
         "solution_leg": solution_leg,
         "dx_solution_leg": dx_solution_leg,
         "f_leg": f_leg,
         "coords_train": coords_train,
         "coords_validation": coords_validation,
         "coords_leg": coords_leg,
         "weights": weights,
         "k": k,
         "T": T
    }

    return data

if __name__ == "__main__":
    key = random.PRNGKey(23)
    N_samples = 1000
    data = get_dataset(key, N_samples)
    jnp.savez("Convection_diffusion.npz", **data)
