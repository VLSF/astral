import argparse
import os.path

import jax.numpy as jnp
import equinox as eqx
import time
import optax

from jax.nn import gelu
from jax.lax import scan
from jax import random, jit, vmap, grad

class PiNNx(eqx.Module):
    matrices: list
    biases: list

    def __init__(self, N_features, N_layers, key, s0=10):
        keys = random.split(key, N_layers+1)
        features = [N_features[0],] + [N_features[1],]*(N_layers-1) + [N_features[-1],]
        self.matrices = [random.uniform(key, (f_in, f_out), minval=-1, maxval=1)*jnp.sqrt(6/f_in) for f_in, f_out, key in zip(features[:-1], features[1:], keys)]
        keys = random.split(keys[-1], N_layers)
        self.biases = [jnp.zeros((f_out,)) for f_in, f_out, key in zip(features[:-1], features[1:], keys)]
        self.matrices[0] = self.matrices[0]*s0

    def __call__(self, x):
        f = x @ self.matrices[0] + self.biases[0]
        for i in range(1, len(self.matrices)):
            f = jnp.sin(f)
            f = f @ self.matrices[i] + self.biases[i]
        return jnp.sin(jnp.pi*x[1]) * f[0]

class PiNNy(eqx.Module):
    matrices: list
    biases: list

    def __init__(self, N_features, N_layers, key, s0=10):
        keys = random.split(key, N_layers+1)
        features = [N_features[0],] + [N_features[1],]*(N_layers-1) + [N_features[-1],]
        self.matrices = [random.uniform(key, (f_in, f_out), minval=-1, maxval=1)*jnp.sqrt(6/f_in) for f_in, f_out, key in zip(features[:-1], features[1:], keys)]
        keys = random.split(keys[-1], N_layers)
        self.biases = [jnp.zeros((f_out,)) for f_in, f_out, key in zip(features[:-1], features[1:], keys)]
        self.matrices[0] = self.matrices[0]*s0

    def __call__(self, x):
        f = x @ self.matrices[0] + self.biases[0]
        for i in range(1, len(self.matrices)):
            f = jnp.sin(f)
            f = f @ self.matrices[i] + self.biases[i]
        return jnp.sin(jnp.pi*x[0]) * f[0]

class PiNNw(eqx.Module):
    matrices: list
    biases: list

    def __init__(self, N_features, N_layers, key, s0=10):
        keys = random.split(key, N_layers+1)
        features = [N_features[0],] + [N_features[1],]*(N_layers-1) + [N_features[-1],]
        self.matrices = [random.uniform(key, (f_in, f_out), minval=-1, maxval=1)*jnp.sqrt(6/f_in) for f_in, f_out, key in zip(features[:-1], features[1:], keys)]
        keys = random.split(keys[-1], N_layers)
        self.biases = [jnp.zeros((f_out,)) for f_in, f_out, key in zip(features[:-1], features[1:], keys)]
        self.matrices[0] = self.matrices[0]*s0

    def __call__(self, x):
        f = x @ self.matrices[0] + self.biases[0]
        for i in range(1, len(self.matrices)):
            f = jnp.sin(f)
            f = f @ self.matrices[i] + self.biases[i]
        return f[0]

class PiNN3(eqx.Module):
    models: list

    def __init__(self, N_features, N_layers, key, s0=10):
        keys = random.split(key, 3)
        self.models = [PiNNx(N_features, N_layers, keys[0], s0=s0), PiNNy(N_features, N_layers, keys[1], s0=s0), PiNNw(N_features, N_layers, keys[2], s0=s0)]

    def __call__(self, x, i):
        return self.models[i](x)

def get_curl(model, x):
    return grad(model, argnums=0)(x, 1)[0] - grad(model, argnums=0)(x, 0)[1]

def get_div(model, x):
    return grad(model, argnums=0)(x, 0)[0] + grad(model, argnums=0)(x, 1)[1]

def get_mixed(model, x):
    return grad(lambda x: grad(model, argnums=0)(x, 0)[1])(x)[0], grad(lambda x: grad(model, argnums=0)(x, 1)[0])(x)[1]

def get_second(model, x):
    return grad(lambda x: grad(model, argnums=0)(x, 0)[1])(x)[1], grad(lambda x: grad(model, argnums=0)(x, 1)[0])(x)[0]

def compute_loss(model, coordinates, mu, f_x, f_y, C_F):
    curl = vmap(get_curl, in_axes=(None, 0))(model, coordinates)
    dw = vmap(lambda x: grad(model, argnums=0)(x, 2), in_axes=0, out_axes=1)(coordinates)
    w = vmap(model, in_axes=(0, None))(coordinates, 2)
    div = vmap(get_div, in_axes=(None, 0))(model, coordinates)
    loss = C_F*jnp.sqrt(jnp.sum((f_x - dw[1])**2 + (f_y + dw[0])**2)) + jnp.sqrt(jnp.sum((w - mu*curl)**2 / mu)) + jnp.sqrt(jnp.sum(div**2))
    return loss

def compute_energy_norm(model, coordinates, mu, sol_x, sol_y, dx_sol_y, dy_sol_x, weights, N_batch=20):
    coordinates = coordinates.reshape(N_batch, -1, 2)
    curl, E_x, E_y = [], [], []
    for i in range(N_batch):
        curl_ = vmap(get_curl, in_axes=(None, 0))(model, coordinates[i])
        E_x_ = vmap(model, in_axes=(0, None))(coordinates[i], 0)
        E_y_ = vmap(model, in_axes=(0, None))(coordinates[i], 1)
        curl.append(curl_)
        E_x.append(E_x_)
        E_y.append(E_y_)
    E_x = jnp.concatenate(E_x, 0)
    E_y = jnp.concatenate(E_y, 0)
    curl = jnp.concatenate(curl, 0)
    integrand = mu*(dx_sol_y - dy_sol_x - curl)**2
    energy_norm = jnp.sqrt(jnp.sum(jnp.sum(weights*integrand.reshape(weights.shape[1], -1), axis=1)*weights[0])) / 2
    return energy_norm

def compute_upper_bound(model, coordinates, mu, f_x, f_y, weights, C_F, N_batch=20):
    coordinates = coordinates.reshape(N_batch, -1, 2)
    curl, E_x, E_y, dxdy_E_x, dxdy_E_y, d2y_E_x, d2x_E_y, w, dw = [], [], [], [], [], [], [], [], []
    for i in range(N_batch):
        curl_ = vmap(get_curl, in_axes=(None, 0))(model, coordinates[i])
        dxdy_E_x_, dxdy_E_y_ = vmap(get_mixed, in_axes=(None, 0))(model, coordinates[i])
        d2y_E_x_, d2x_E_y_ = vmap(get_second, in_axes=(None, 0))(model, coordinates[i])
        E_x_ = vmap(model, in_axes=(0, None))(coordinates[i], 0)
        E_y_ = vmap(model, in_axes=(0, None))(coordinates[i], 1)
        w_ = vmap(model, in_axes=(0, None))(coordinates[i], 2)
        dw_ = vmap(lambda x: grad(model, argnums=0)(x, 2), in_axes=0, out_axes=1)(coordinates[i])
        E_x.append(E_x_)
        E_y.append(E_y_)
        curl.append(curl_)
        dxdy_E_x.append(dxdy_E_x_)
        dxdy_E_y.append(dxdy_E_y_)
        d2y_E_x.append(d2y_E_x_)
        d2x_E_y.append(d2x_E_y_)
        w.append(w_)
        dw.append(dw_)
    E_x = jnp.concatenate(E_x, 0)
    E_y = jnp.concatenate(E_y, 0)
    curl = jnp.concatenate(curl, 0)
    dxdy_E_x = jnp.concatenate(dxdy_E_x, 0)
    dxdy_E_y = jnp.concatenate(dxdy_E_y, 0)
    d2y_E_x = jnp.concatenate(d2y_E_x, 0)
    d2x_E_y = jnp.concatenate(d2x_E_y, 0)
    w = jnp.concatenate(w, 0)
    dw = jnp.concatenate(dw, 1)
    integrand_1 = ((f_x - dw[1])**2 + (f_y + dw[0])**2)
    integrand_2 = (w - mu*curl)**2 / mu
    upper_bound = C_F * jnp.sqrt(jnp.sum(jnp.sum(weights*integrand_1.reshape(weights.shape[1], -1), axis=1)*weights[0])) / 2
    upper_bound += jnp.sqrt(jnp.sum(jnp.sum(weights*integrand_2.reshape(weights.shape[1], -1), axis=1)*weights[0])) / 2
    return upper_bound

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

def make_step_scan(carry, ind, optim):
    model, coordinates, mu, f_x, f_y, C_F, opt_state = carry
    loss, grads = vmap(compute_loss_and_grads, in_axes=(0, None, 0, 0, 0, 0))(model, coordinates[ind], mu[:, ind], f_x[:, ind], f_y[:, ind], C_F)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, coordinates, mu, f_x, f_y, C_F, opt_state], loss

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
            "default": [1e-4,],
            "nargs": '+',
            "type": float,
            "help": "learning rate"
        },
        "-gamma": {
            "default": [0.5,],
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
            "default": [25000,],
            "nargs": '+',
            "type": int,
            "help": "number of updates after which learning rate is multiplied by chosen learning rate decay"
        },
        "-N_features": {
            "default": [50,],
            "nargs": '+',
            "type": int,
            "help": "number of features in a hidden layer"
        },
        "-N_layers": {
            "default": [4,],
            "nargs": '+',
            "type": int,
            "help": "number of layers in MLP"
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
    header = "N_batch,NN_batch,learning_rate,gamma,N_updates,N_drop,N_features,N_layers,energy_norm_mean,energy_norm_std,relative_error_x_mean,relative_error_x_std,relative_error_y_mean,relative_error_y_std,final_loss_mean,final_loss_std,training_time,upper_bound_mean,upper_bound_std"
    save_here = results_path + "astral.csv"
    if not os.path.isfile(save_here):
        with open(save_here, "w") as f:
            f.write(header)
            
    data = jnp.load(dataset_path)
    mu_train = data["mu_train"]
    f_x_train = data["f_x_train"]
    f_y_train = data["f_y_train"]
    dx_mu_train = data["dx_mu_train"]
    dy_mu_train = data["dy_mu_train"]
    dx_sol_y_legendre = data["dx_sol_y_legendre"]
    dy_sol_x_legendre = data["dy_sol_x_legendre"]
    sol_x_legendre = data["sol_x_legendre"]
    sol_y_legendre = data["sol_y_legendre"]
    f_x_legendre = data["f_x_legendre"]
    f_y_legendre = data["f_y_legendre"]
    mu_legendre = data["mu_legendre"]
    sol_x_eval = data["sol_x_eval"]
    sol_y_eval = data["sol_y_eval"]
    coords_train = data["coords_train"]
    weights_ = data["weights"]
    coords_legendre = data["coords_legendre"]
    coords_eval = data["coords_eval"]
    C_F = 1 / (2*jnp.pi*jnp.sqrt(jnp.min(mu_legendre, axis=1)))
    
    key = random.PRNGKey(23)
    keys = random.split(key, 2)
    for learning_rate in learning_rates:
        for gamma in gammas:
            for N_drop in Ns_drop:
                for N_features_ in Ns_features:
                    for N_layers in Ns_layers:
                        M = N_batch_x*N_batch_x

                        inds = random.choice(keys[0], mu_train.shape[-1], (N_updates, M))

                        N_features = [2, N_features_, 1]
                        NN_batch = N_batch_NN
                        keys = random.split(keys[1], NN_batch)
                        model = vmap(PiNN3, in_axes=(None, None, 0))(N_features, N_layers, keys)

                        sc = optax.exponential_decay(learning_rate, N_drop, gamma)
                        optim = optax.lion(learning_rate=sc)
                        opt_state = optim.init(eqx.filter(model, eqx.is_array))

                        carry = [model, coords_train, mu_train[:NN_batch], f_x_train[:NN_batch], f_y_train[:NN_batch], C_F[:NN_batch], opt_state]

                        make_step_scan_ = lambda a, b: make_step_scan(a, b, optim)

                        start = time.time()
                        carry, loss = scan(make_step_scan_, carry, inds)
                        stop = time.time()
                        training_time = stop - start
                        model = carry[0]

                        N_batch = 20
                        coords_eval_ = coords_eval.reshape(N_batch, -1, 2)
                        prediction_x = []
                        prediction_y = []
                        for i in range(N_batch):
                            prediction_x_ = vmap(vmap(lambda model, coords: model(coords, 0), in_axes=(None, 0)), in_axes=(0, None), out_axes=1)(model, coords_eval_[i])
                            prediction_y_ = vmap(vmap(lambda model, coords: model(coords, 1), in_axes=(None, 0)), in_axes=(0, None), out_axes=1)(model, coords_eval_[i])
                            prediction_x.append(prediction_x_)
                            prediction_y.append(prediction_y_)
                        prediction_x = jnp.concatenate(prediction_x, 0).T
                        prediction_y = jnp.concatenate(prediction_y, 0).T

                        errors_x = jnp.linalg.norm(sol_x_eval[:NN_batch] - prediction_x, axis=1) / jnp.linalg.norm(sol_x_eval[:NN_batch], axis=1)
                        errors_y = jnp.linalg.norm(sol_y_eval[:NN_batch] - prediction_y, axis=1) / jnp.linalg.norm(sol_y_eval[:NN_batch], axis=1)

                        energy_norms = vmap(compute_energy_norm, in_axes=(0, None, 0, 0, 0, 0, 0, None))(model, coords_legendre, mu_legendre[:NN_batch], sol_x_legendre[:NN_batch], sol_y_legendre[:NN_batch], dx_sol_y_legendre[:NN_batch], dy_sol_x_legendre[:NN_batch], weights_)
                        upper_bounds = vmap(compute_upper_bound, in_axes=(0, None, 0, 0, 0, None, 0))(model, coords_legendre, mu_legendre[:NN_batch], f_x_legendre[:NN_batch], f_y_legendre[:NN_batch], weights_, C_F[:NN_batch])


                        final_loss_mean = jnp.mean(loss, 1)[-1]
                        final_loss_std = jnp.sqrt(jnp.var(loss, 1))[-1]
                        relative_error_x_mean = jnp.mean(errors_x)
                        relative_error_x_std = jnp.sqrt(jnp.var(errors_x))
                        relative_error_y_mean = jnp.mean(errors_y)
                        relative_error_y_std = jnp.sqrt(jnp.var(errors_y))
                        energy_norm_mean = jnp.mean(energy_norms)
                        energy_norm_std = jnp.sqrt(jnp.var(energy_norms))
                        upper_bound_mean = jnp.mean(upper_bounds)
                        upper_bound_std = jnp.sqrt(jnp.var(upper_bounds))

                        res = f"\n{N_batch_x},{NN_batch},{learning_rate},{gamma},{N_updates},{N_drop},{N_features_},{N_layers},{energy_norm_mean},{energy_norm_std},{relative_error_x_mean},{relative_error_x_std},{relative_error_y_mean},{relative_error_y_std},{final_loss_mean},{final_loss_std},{training_time},{upper_bound_mean},{upper_bound_std}"
                        with open(save_here, "a") as f:
                            f.write(res)