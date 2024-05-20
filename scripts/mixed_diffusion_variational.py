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


def get_flux(model, x):
    return grad(model, argnums=0)(x)


def get_laplacian(model, x):
    return grad(lambda x: grad(model, argnums=0)(x)[0])(x)[0] + \
           grad(lambda x: grad(model, argnums=0)(x)[1])(x)[1]


def get_mixed(model, x):
    return grad(lambda x: grad(model, argnums=0)(x)[0])(x)[1]


def compute_loss(model, coordinates, a, dx_a, dy_a, rhs, eps):
    flux = vmap(get_flux, in_axes=(None, 0), out_axes=1)(model, coordinates)
    u = vmap(model, in_axes=0)(coordinates)
    return jnp.linalg.norm(a * (flux[0]**2 + flux[1]**2+2*eps*(flux[0]*flux[1]))/2 + rhs*u)


@jit
def compute_error_energy_norm(model, coordinates, a, dx_sol, dy_sol, eps, weights, N_batch=10):
    flux_ = []
    coordinates = coordinates.reshape(N_batch, -1, 2)
    for i in range(N_batch):
        flux = vmap(get_flux, in_axes=(None, 0), out_axes=1)(model, coordinates[i])
        flux_.append(flux)
    flux = jnp.concatenate(flux_, 1)
    integrand = a * ((flux[0] - dx_sol)**2 + eps**2*(flux[1] - dy_sol)**2)
    l = jnp.sum(jnp.sum(integrand.reshape(weights.size, weights.size)*weights, axis=1) * weights[0]) / 4
    return l


compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)


def make_step_scan(carry, ind, optim):
    model, coordinates, a, dx_a, dy_a, rhs, eps, opt_state = carry
    loss, grads = vmap(compute_loss_and_grads, in_axes=(0, None, 0, 0, 0, 0, None))(model, coordinates[ind], a[:, ind],
                                                                                    dx_a[:, ind], dy_a[:, ind],
                                                                                    rhs[:, ind], eps)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, coordinates, a, dx_a, dy_a, rhs, eps, opt_state], loss


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
    save_here = results_path + "variational.csv"
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
    eps = args['eps'][0]

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
                        model = vmap(PiNN, in_axes=(None, None, 0))(N_features, N_layers, keys)

                        sc = optax.exponential_decay(learning_rate, N_drop, gamma)
                        optim = optax.lion(learning_rate=sc)
                        opt_state = optim.init(eqx.filter(model, eqx.is_array))

                        carry = [model, coords_train, a_train[:NN_batch], dx_a_train[:NN_batch], dy_a_train[:NN_batch],
                                 rhs_train[:NN_batch], eps, opt_state]

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
                            predicted = vmap(lambda model, coords: vmap(jit(model))(coords), in_axes=(0, None))(model,
                                                                                                                coords_eval_[
                                                                                                                    i])
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
                                                                                                    :NN_batch], eps,
                                                                                                    weights))

                        final_loss_mean = jnp.mean(loss, 1)[-1]
                        final_loss_std = jnp.sqrt(jnp.var(loss, 1))[-1]
                        relative_error_mean = jnp.mean(relative_errors)
                        relative_error_std = jnp.sqrt(jnp.var(relative_errors))
                        energy_norm_mean = jnp.mean(energy_norms)
                        energy_norm_std = jnp.sqrt(jnp.var(energy_norms))

                        res = f"\n{N_batch_x},{NN_batch},{learning_rate},{gamma},{eps},{N_updates},{N_drop},{N_features_},{N_layers},{energy_norm_mean},{energy_norm_std},{relative_error_mean},{relative_error_std},{final_loss_mean},{final_loss_std},{training_time}"
                        with open(save_here, "a") as f:
                            f.write(res)
