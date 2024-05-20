import argparse
import os.path

import jax.numpy as jnp
import equinox as eqx
import time
import optax

from jax.nn import gelu
from jax.lax import scan
from jax import random, jit, vmap, grad


class PiNN(eqx.Module):
    matrices: list
    biases: list

    def __init__(self, N_features, N_layers, key, s0=10):
        keys = random.split(key, N_layers + 1)
        features = [N_features[0], ] + [N_features[1], ] * (N_layers - 1) + [N_features[-1], ]
        self.matrices = [random.uniform(key, (f_in, f_out), minval=-1, maxval=1) * jnp.sqrt(6 / f_in) for
                         f_in, f_out, key in zip(features[:-1], features[1:], keys)]
        keys = random.split(keys[-1], N_layers)
        self.biases = [jnp.zeros((f_out,)) for f_in, f_out, key in zip(features[:-1], features[1:], keys)]
        self.matrices[0] = self.matrices[0] * s0

    def __call__(self, x):
        f = x @ self.matrices[0] + self.biases[0]
        for i in range(1, len(self.matrices)):
            f = jnp.sin(f)
            f = f @ self.matrices[i] + self.biases[i]
        return jnp.sin(jnp.pi * x[0]) * jnp.sin(jnp.pi * x[1]) * f[0]


class PiNNu(eqx.Module):
    matrices: list
    biases: list

    def __init__(self, N_features, N_layers, key, s0=10):
        keys = random.split(key, N_layers + 1)
        features = [N_features[0], ] + [N_features[1], ] * (N_layers - 1) + [N_features[-1], ]
        self.matrices = [random.uniform(key, (f_in, f_out), minval=-1, maxval=1) * jnp.sqrt(6 / f_in) for
                         f_in, f_out, key in zip(features[:-1], features[1:], keys)]
        keys = random.split(keys[-1], N_layers)
        self.biases = [jnp.zeros((f_out,)) for f_in, f_out, key in zip(features[:-1], features[1:], keys)]
        self.matrices[0] = self.matrices[0] * s0

    def __call__(self, x):
        f = x @ self.matrices[0] + self.biases[0]
        for i in range(1, len(self.matrices)):
            f = jnp.sin(f)
            f = f @ self.matrices[i] + self.biases[i]
        return f[0]


class PiNN3(eqx.Module):
    models: list
    beta: jnp.array

    def __init__(self, N_features, N_layers, key):
        keys = random.split(key, 3)
        self.models = [PiNNu(N_features, N_layers, key) for key in keys[:2]]
        self.models.append(PiNN(N_features, N_layers, keys[-1]))
        self.beta = jnp.array([1.0, ])

    def __call__(self, x, i):
        return self.models[i](x)


def get_flux(model, x, i):
    return grad(model, argnums=0)(x, i)


def compute_loss(model, coordinates, rhs, a, C_F, eps):
    flux = vmap(grad(lambda x: model(x, 2), argnums=0), in_axes=0)(coordinates)
    dx_y = [vmap(grad(lambda x: model(x, i), argnums=0), in_axes=0)(coordinates)[:, i] for i in [0, 1]]
    y1 = vmap(model, in_axes=(0, None))(coordinates, 0)
    y2 = vmap(model, in_axes=(0, None))(coordinates, 1)
    y = jnp.stack([y1, y2], 1)
    matrix_a1 = jnp.stack([a, eps * a], 1)
    matrix_a2 = jnp.stack([eps * a, a], 1)
    matrix_a = jnp.stack([matrix_a1, matrix_a2], 2)
    dom = -eps ** 2 * a + a
    inv_a1 = jnp.stack([1/dom, -eps/dom], 1)
    inv_a2 = jnp.stack([-eps/dom, 1/dom], 1)
    inv_a = jnp.stack([inv_a1, inv_a2], 2)
    aflux = vmap(jnp.matmul, (0, 0))(matrix_a, flux)
    loss = (1 + model.beta[0] ** 2) * (C_F ** 2 * (rhs + dx_y[0] + dx_y[1]) ** 2 + (
        vmap(jnp.dot, (0, 0))(vmap(jnp.matmul, (0, 0))(inv_a, aflux - y), aflux - y)) / (
                                               model.beta[0] ** 2))
    return jnp.linalg.norm(loss)


@jit
def compute_upper_bound(model, coordinates, weights, rhs, a, C_F, eps, N_batch=10):
    flux = []
    dx_y_0 = []
    dx_y_1 = []
    y1 = []
    y2 = []
    coordinates = coordinates.reshape(N_batch, -1, 2)
    for j in range(N_batch):
        flux_ = vmap(get_flux, in_axes=(None, 0, None), out_axes=1)(model, coordinates[j], 2)
        dx_y_ = [vmap(grad(lambda x: model(x, i), argnums=0), in_axes=0)(coordinates[j])[:, i] for i in [0, 1]]
        y1_ = vmap(model, in_axes=(0, None))(coordinates[j], 0)
        y2_ = vmap(model, in_axes=(0, None))(coordinates[j], 1)
        flux.append(flux_)
        dx_y_0.append(dx_y_[0])
        dx_y_1.append(dx_y_[1])
        y1.append(y1_)
        y2.append(y2_)
    flux = jnp.concatenate(flux, 1).T
    dx_y = [jnp.concatenate(dx_y_0, 0), jnp.concatenate(dx_y_1, 0)]
    y1 = jnp.concatenate(y1, 0)
    y2 = jnp.concatenate(y2, 0)
    y = jnp.stack([y1, y2], 1)
    matrix_a1 = jnp.stack([a, eps * a], 1)
    matrix_a2 = jnp.stack([eps * a, a], 1)
    matrix_a = jnp.stack([matrix_a1, matrix_a2], 2)
    dom = -eps ** 2 * a + a
    inv_a1 = jnp.stack([1/dom, -eps/dom], 1)
    inv_a2 = jnp.stack([-eps/dom, 1/dom], 1)
    inv_a = jnp.stack([inv_a1, inv_a2], 2)
    aflux = vmap(jnp.matmul, (0, 0))(matrix_a, flux)
    integrand = (1 + model.beta[0] ** 2) * (C_F ** 2 * (rhs + dx_y[0] + dx_y[1]) ** 2 + (
        vmap(jnp.dot, (0, 0))(vmap(jnp.matmul, (0, 0))(inv_a, aflux - y), aflux - y)) / (
                                                    model.beta[0] ** 2))
    l = jnp.sum(jnp.sum(integrand.reshape(weights.size, weights.size) * weights, axis=1) * weights[0]) / 4
    return l


@jit
def compute_error_energy_norm(model, coordinates, a, dx_sol, dy_sol, weights, eps, N_batch=10):
    flux_ = []
    coordinates = coordinates.reshape(N_batch, -1, 2)
    for i in range(N_batch):
        flux = vmap(get_flux, in_axes=(None, 0, None), out_axes=1)(model, coordinates[i], 2)
        flux_.append(flux)
    flux = jnp.concatenate(flux_, 1)
    integrand = a * ((flux[0] - dx_sol) ** 2 + a * (flux[1] - dy_sol) ** 2) + 2 * eps * a * (flux[0] - dx_sol) * (
                flux[1] - dy_sol)
    l = jnp.sum(jnp.sum(integrand.reshape(weights.size, weights.size) * weights, axis=1) * weights[0]) / 4
    return l


compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)


def make_step_scan(carry, ind, optim):
    model, coordinates, rhs, a, C_F, eps, opt_state = carry
    loss, grads = vmap(compute_loss_and_grads, in_axes=(0, None, 0, 0, 0, None))(model, coordinates[ind], rhs[:, ind],
                                                                                 a[:, ind], C_F, eps)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, coordinates, rhs, a, C_F, eps, opt_state], loss


def get_argparser():
    parser = argparse.ArgumentParser()
    args = {
        "-N_batch_x": {
            "default": 15,
            "type": int,
            "help": "number of points used along each dimension for gradient estimation"
        },
        "-N_batch_NN": {
            "default": 100,
            "type": int,
            "help": "number of neural networks trained in parallel"
        },
        "-path_to_dataset": {
            "help": "path to dataset in the .npz format"
        },
        "-path_to_results": {
            "help": "path to folder where to save results"
        },
        "-learning_rate": {
            "default": [1e-4, ],
            "nargs": '+',
            "type": float,
            "help": "learning rate"
        },
        "-gamma": {
            "default": [0.5, ],
            "nargs": '+',
            "type": float,
            "help": "decay parameter for the exponential decay of learning rate"
        },
        "-N_updates": {
            "default": 50000,
            "type": int,
            "help": "number of updates of the model weights"
        },
        "-N_drop": {
            "default": [25000, ],
            "nargs": '+',
            "type": int,
            "help": "number of updates after which learning rate is multiplied by chosen learning rate decay"
        },
        "-N_features": {
            "default": [50, ],
            "nargs": '+',
            "type": int,
            "help": "number of features in a hidden layer"
        },
        "-N_layers": {
            "default": [4, ],
            "nargs": '+',
            "type": int,
            "help": "number of layers in MLP"
        },
        "-eps": {
            "default": [0.5],
            "nargs": '+',
            "type": float,
            "help": "eps"
        }
    }

    for key in args:
        parser.add_argument(key, **args[key])

    return parser


if __name__ == "__main__":
    parser = get_argparser()
    args = vars(parser.parse_args())
    dataset_path = args["path_to_dataset"]
    results_path = args["path_to_results"]
    N_batch_x = args["N_batch_x"]
    N_batch_NN = args["N_batch_NN"]
    learning_rates = args["learning_rate"]
    gammas = args["gamma"]
    N_updates = args["N_updates"]
    Ns_drop = args["N_drop"]
    Ns_features = args["N_features"]
    Ns_layers = args["N_layers"]
    header = "N_batch,NN_batch,learning_rate,gamma,eps,N_updates,N_drop,N_features,N_layers,energy_norm_mean,energy_norm_std,relative_error_mean,relative_error_std,final_loss_mean,final_loss_std,training_time,upper_bound_mean,upper_bound_std"
    save_here = results_path + "astral.csv"
    if not os.path.isfile(save_here):
        with open(save_here, "w") as f:
            f.write(header)

    data = jnp.load(dataset_path)
    a_train = data["a_train"]
    rhs_train = data["rhs_train"]
    dx_a_train = data["dx_a_train"]
    dy_a_train = data["dy_a_train"]
    a_eval_legendre = data["a_eval_legendre"]
    dx_sol_eval_legendre = data["dx_sol_eval_legendre"]
    dy_sol_eval_legendre = data["dy_sol_eval_legendre"]
    rhs_legendre = data["rhs_legendre"]
    a_legendre = data["a_legendre"]
    sol_eval = data["sol_eval"]
    C_F = data["C_F"]
    coords_train = data["coords_train"]
    weights = data["weights"]
    coords_legendre = data["coords_legendre"]
    coords_eval = data["coords_eval"]
    eps = args["eps"][0]

    key = random.PRNGKey(23)
    keys = random.split(key, 2)
    for learning_rate in learning_rates:
        for gamma in gammas:
            for N_drop in Ns_drop:
                for N_features_ in Ns_features:
                    for N_layers in Ns_layers:
                        M = N_batch_x * N_batch_x

                        inds = random.choice(keys[0], a_train.shape[-1], (N_updates, M))

                        N_features = [2, N_features_, 1]
                        NN_batch = N_batch_NN
                        keys = random.split(keys[1], NN_batch)
                        model = vmap(PiNN3, in_axes=(None, None, 0))(N_features, N_layers, keys)

                        sc = optax.exponential_decay(learning_rate, N_drop, gamma)
                        optim = optax.lion(learning_rate=sc)
                        opt_state = optim.init(eqx.filter(model, eqx.is_array))

                        carry = [model, coords_train, rhs_train[:NN_batch], a_train[:NN_batch], C_F[:NN_batch], eps,
                                 opt_state]

                        make_step_scan_ = lambda a, b: make_step_scan(a, b, optim)
                        start = time.time()
                        carry, loss = scan(make_step_scan_, carry, inds)
                        stop = time.time()
                        training_time = stop - start
                        model = carry[0]

                        predicted_ = []
                        n_batch = 10
                        coords_eval_ = coords_eval.reshape(n_batch, -1, 2)
                        for i in range(n_batch):
                            predicted = vmap(lambda model, coords: vmap(jit(lambda x: model(x, 2)))(coords),
                                             in_axes=(0, None))(model, coords_eval_[i])
                            predicted_.append(predicted)
                        predicted = jnp.concatenate(predicted_, 1)
                        relative_errors = jnp.linalg.norm(predicted - sol_eval[:NN_batch], axis=1) / jnp.linalg.norm(
                            sol_eval[:NN_batch], axis=1)
                        energy_norms = jnp.sqrt(
                            vmap(compute_error_energy_norm, in_axes=(0, None, 0, 0, 0, None, None))(model,
                                                                                                    coords_legendre,
                                                                                                    a_eval_legendre[
                                                                                                    :NN_batch],
                                                                                                    dx_sol_eval_legendre[
                                                                                                    :NN_batch],
                                                                                                    dy_sol_eval_legendre[
                                                                                                    :NN_batch], weights,
                                                                                                    eps))
                        upper_bounds = jnp.sqrt(
                            vmap(compute_upper_bound, in_axes=(0, None, None, 0, 0, 0, None))(model, coords_legendre,
                                                                                              weights,
                                                                                              rhs_legendre[:NN_batch],
                                                                                              a_legendre[:NN_batch],
                                                                                              C_F[:NN_batch], eps))

                        final_loss_mean = jnp.mean(loss, 1)[-1]
                        final_loss_std = jnp.sqrt(jnp.var(loss, 1))[-1]
                        relative_error_mean = jnp.mean(relative_errors)
                        relative_error_std = jnp.sqrt(jnp.var(relative_errors))
                        energy_norm_mean = jnp.mean(energy_norms)
                        energy_norm_std = jnp.sqrt(jnp.var(energy_norms))
                        upper_bound_mean = jnp.mean(upper_bounds)
                        upper_bound_std = jnp.sqrt(jnp.var(upper_bounds))

                        res = f"\n{N_batch_x},{NN_batch},{learning_rate},{gamma},{eps},{N_updates},{N_drop},{N_features_},{N_layers},{energy_norm_mean},{energy_norm_std},{relative_error_mean},{relative_error_std},{final_loss_mean},{final_loss_std},{training_time},{upper_bound_mean},{upper_bound_std}"
                        with open(save_here, "a") as f:
                            f.write(res)
