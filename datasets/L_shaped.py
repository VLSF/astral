import numpy as np
import jax.numpy as jnp
import itertools

from scipy.special import roots_legendre
from scipy.sparse import coo_matrix, diags, kron
from scipy.sparse.linalg import spsolve
from scipy.interpolate import griddata
from jax import random

def get_weights_and_frequencies(N, k=2, l=10):
    f = np.array([[2*np.pi*i, 2*np.pi*j] for i, j in itertools.product(range(N), repeat=2)])
    w = np.array([1/(1 + (2*np.pi*i/l)**2 + (2*np.pi*j/l)**2)**k for i, j in itertools.product(range(N), repeat=2)]).reshape(-1,)
    return w, f

def get_basis(f, x):
    return np.stack([np.exp(1j*np.sum(np.expand_dims(f_, 0)*x, axis=1)) for f_ in f], 1)

def get_coefficients(key, w):
    return np.array(random.normal(key, w.shape, dtype=jnp.complex64))

def get_problem_data(n=32):
    N = 2*n + 1
    x = y = np.linspace(0, 1, N+2)[1:-1]
    h = (x[1] - x[0]).item()
    train_coords = np.stack(np.meshgrid(x, x, indexing='ij'), 0).reshape(2, -1)
    ind = np.arange(train_coords.shape[1])
    mask = np.logical_not((train_coords[0] >= 0.5) * (train_coords[1] >= 0.5))
    keep_indices = ind[mask]

    Delta = diags([2*np.ones((N,)), -np.ones((N-1,)), -np.ones((N-1,))], offsets=[0, -1, +1]) / h**2
    I = diags([np.ones((N,))], offsets=[0,])
    A = kron(I, Delta) + kron(Delta, I)
    A = A[keep_indices, :].tocsc()[:, keep_indices]

    N_mesh = 100
    x, weights = roots_legendre(N_mesh)
    x = (x + 1)/4
    weights = np.array(weights).reshape(1, -1)
    coords1 = np.stack(np.meshgrid(x, x, indexing="ij"), 2)
    coords1 = coords1.reshape(-1, 2)

    coords2 = np.stack(np.meshgrid(x + 0.5, x, indexing="ij"), 2)
    coords2 = coords2.reshape(-1, 2)

    coords3 = np.stack(np.meshgrid(x, x + 0.5, indexing="ij"), 2)
    coords3 = coords3.reshape(-1, 2)

    coords_leg = np.stack([coords1, coords2, coords3])

    x = y = jnp.linspace(0, 1, 2*n+3)
    coords_for_embedding = np.stack(np.meshgrid(x, y, indexing="ij"), 2)

    x = y = np.linspace(0, 1, 200)[1:-1]
    coords_val = np.stack(np.meshgrid(x, y, indexing="ij"), 2).reshape(-1, 2)
    mask_val = np.logical_not((coords_val[:, 0] >= 0.5) * (coords_val[:, 1] >= 0.5))
    coords_val = coords_val[mask_val]

    return A, train_coords.T, train_coords[:, keep_indices].T, keep_indices, coords_leg, weights, coords_for_embedding, coords_val

def get_dataset(key, N_samples):
    N = 50
    A, coords_full, train_coords, keep_indices, coords_leg, weights, coords_for_embedding, coords_val = get_problem_data()
    w, f = get_weights_and_frequencies(N)
    B_train = get_basis(f, train_coords)
    B_leg = np.stack([get_basis(f, coords_leg[i]) for i in range(3)])
    B2_train = []
    B2_leg = []
    F_train = []
    F_leg = []
    Sol_train = []
    Sol_val = []
    Sol_leg = []
    for key_ in random.split(key, N_samples):
        keys = random.split(key_)
        c_b = get_coefficients(keys[0], w)
        c_f = get_coefficients(keys[1], w)
        b_train = np.real(B_train @ (c_b * w))
        f_train = np.real(B_train @ (c_f * w))
        b_leg = np.real(B_leg @ (c_b * w))
        f_leg = np.real(B_leg @ (c_f * w))
        A_ = diags([b_train**2,], offsets=[0, ]).tocsc() + A
        sol_train = spsolve(A_, f_train)

        embedded = np.zeros_like(coords_full[:, 0])
        embedded[keep_indices] = sol_train
        embedded_with_boundary = np.zeros_like(coords_for_embedding[:, :, 0])
        embedded_with_boundary[1:-1, 1:-1] = embedded.reshape(65, 65)
        sol_leg = np.stack([griddata(coords_for_embedding.reshape(-1, 2), embedded_with_boundary.reshape(-1,), c) for c in coords_leg], 0)
        sol_val = griddata(coords_for_embedding.reshape(-1, 2), embedded_with_boundary.reshape(-1,), coords_val)
        B2_train.append(b_train**2)
        B2_leg.append(b_leg**2)
        F_train.append(f_train)
        F_leg.append(f_leg)
        Sol_train.append(sol_train)
        Sol_val.append(sol_val)
        Sol_leg.append(sol_leg)
    B2_train = jnp.array(np.stack(B2_train))
    B2_leg = jnp.array(np.stack(B2_leg))
    F_train = jnp.array(np.stack(F_train))
    F_leg = jnp.array(np.stack(F_leg))
    Sol_train = jnp.array(np.stack(Sol_train))
    Sol_val = jnp.array(np.stack(Sol_val))
    Sol_leg = jnp.array(np.stack(Sol_leg))
    train_coords = jnp.array(train_coords)
    coords_leg = jnp.array(coords_leg)
    weights = jnp.array(weights)
    coords_val = jnp.array(coords_val)
    data = {
        "solution_train": Sol_train,
        "solution_validation": Sol_val,
        "solution_leg": Sol_leg,
        "coords_train": train_coords,
        "coords_leg": coords_leg,
        "coords_validation": coords_val,
        "b2_train": B2_train,
        "b2_leg": B2_leg,
        "f_train": F_train,
        "f_leg": F_leg,
        "weights": weights
    }
    return data

if __name__ == "__main__":
    key = random.PRNGKey(23)
    N_samples = 1000
    data = get_dataset(key, N_samples)
    jnp.savez("L_shaped.npz", **data)
