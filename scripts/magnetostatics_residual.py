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

class PiNN2(eqx.Module):
    models: list

    def __init__(self, N_features, N_layers, key, s0=10):
        keys = random.split(key)
        self.models = [PiNNx(N_features, N_layers, keys[0], s0=s0), PiNNy(N_features, N_layers, keys[1], s0=s0)]

    def __call__(self, x, i):
        return self.models[i](x)

def get_curl(model, x):
    return grad(model, argnums=0)(x, 1)[0] - grad(model, argnums=0)(x, 0)[1]

def get_div(model, x):
    return grad(model, argnums=0)(x, 0)[0] + grad(model, argnums=0)(x, 1)[1]

def get_mixed(model, x):
    return grad(lambda x: grad(model, argnums=0)(x, 0)[0])(x)[1], grad(lambda x: grad(model, argnums=0)(x, 1)[0])(x)[1]

def get_second(model, x):
    return grad(lambda x: grad(model, argnums=0)(x, 0)[1])(x)[1], grad(lambda x: grad(model, argnums=0)(x, 1)[0])(x)[0]

def compute_loss(model, coordinates, mu, dx_mu, dy_mu, f_x, f_y):
    curl = vmap(get_curl, in_axes=(None, 0))(model, coordinates)
    dxdy_E_x, dxdy_E_y = vmap(get_mixed, in_axes=(None, 0))(model, coordinates)
    d2y_E_x, d2x_E_y = vmap(get_second, in_axes=(None, 0))(model, coordinates)
    div = vmap(get_div, in_axes=(None, 0))(model, coordinates)
    lx = dy_mu*curl + mu*(dxdy_E_y - d2y_E_x) - f_x
    ly = -dx_mu*curl - mu*(d2x_E_y - dxdy_E_x) - f_y
    return jnp.linalg.norm(lx) + jnp.linalg.norm(ly) + jnp.linalg.norm(div)

def compute_energy_norm(model, coordinates, mu, sol_x, sol_y, dx_sol_y, dy_sol_x, weights, N_batch=20):
    coordinates = coordinates.reshape(N_batch, -1, 2)
    curl = []
    for i in range(N_batch):
        curl_ = vmap(get_curl, in_axes=(None, 0))(model, coordinates[i])
        curl.append(curl_)
    curl = jnp.concatenate(curl, 0)
    integrand = mu*(dx_sol_y - dy_sol_x - curl)**2
    energy_norm = jnp.sqrt(jnp.sum(jnp.sum(weights*integrand.reshape(weights.shape[1], -1), axis=1)*weights[0])) / 2
    return energy_norm

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

def make_step_scan(carry, ind, optim):
    model, coordinates, mu, dx_mu, dy_mu, f_x, f_y, opt_state = carry
    loss, grads = vmap(compute_loss_and_grads, in_axes=(0, None, 0, 0, 0, 0, 0))(model, coordinates[ind], mu[:, ind], dx_mu[:, ind], dy_mu[:, ind], f_x[:, ind], f_y[:, ind])
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, coordinates, mu, dx_mu, dy_mu, f_x, f_y, opt_state], loss

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
    header = "N_batch,NN_batch,learning_rate,gamma,N_updates,N_drop,N_features,N_layers,energy_norm_mean,energy_norm_std,relative_error_x_mean,relative_error_x_std,relative_error_y_mean,relative_error_y_std,final_loss_mean,final_loss_std,training_time"
    save_here = results_path + "residual.csv"
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
                        model = vmap(PiNN2, in_axes=(None, None, 0, None))(N_features, N_layers, keys, 5)

                        sc = optax.exponential_decay(learning_rate, N_drop, gamma)
                        optim = optax.lion(learning_rate=sc)
                        opt_state = optim.init(eqx.filter(model, eqx.is_array))

                        carry = [model, coords_train, mu_train[:NN_batch], dx_mu_train[:NN_batch], dy_mu_train[:NN_batch], f_x_train[:NN_batch], f_y_train[:NN_batch], opt_state]

                        make_step_scan_ = lambda a, b: make_step_scan(a, b, optim)

                        start = time.time()
                        carry, loss = scan(make_step_scan_, carry, inds)
                        stop = time.time()
                        training_time = stop - start
                        model = carry[0]

                        N_batch = 20
                        start = time.time()
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
                        stop = time.time()

                        final_loss_mean = jnp.mean(loss, 1)[-1]
                        final_loss_std = jnp.sqrt(jnp.var(loss, 1))[-1]
                        relative_error_x_mean = jnp.mean(errors_x)
                        relative_error_x_std = jnp.sqrt(jnp.var(errors_x))
                        relative_error_y_mean = jnp.mean(errors_y)
                        relative_error_y_std = jnp.sqrt(jnp.var(errors_y))
                        energy_norm_mean = jnp.mean(energy_norms)
                        energy_norm_std = jnp.sqrt(jnp.var(energy_norms))

                        res = f"\n{N_batch_x},{NN_batch},{learning_rate},{gamma},{N_updates},{N_drop},{N_features_},{N_layers},{energy_norm_mean},{energy_norm_std},{relative_error_x_mean},{relative_error_x_std},{relative_error_y_mean},{relative_error_y_std},{final_loss_mean},{final_loss_std},{training_time}"
                        with open(save_here, "a") as f:
                            f.write(res)