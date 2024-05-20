import argparse
import os.path

import jax.numpy as jnp
import equinox as eqx
import time
import optax

from jax.nn import gelu
from jax.lax import scan
from jax import random, jit, vmap, grad

def get_ic(x, phi_k, k, a):
    return jnp.sum(jnp.exp(x*a/2) * jnp.sin(k*x) * phi_k)

class PiNN(eqx.Module):
    matrices: list
    biases: list

    def __init__(self, N_features, N_layers, key, s0=20):
        keys = random.split(key, N_layers+1)
        features = [N_features[0],] + [N_features[1],]*(N_layers-1) + [N_features[-1],]
        self.matrices = [random.uniform(key, (f_in, f_out), minval=-1, maxval=1)*jnp.sqrt(6/f_in) for f_in, f_out, key in zip(features[:-1], features[1:], keys)]
        keys = random.split(keys[-1], N_layers)
        self.biases = [jnp.zeros((f_out,)) for f_in, f_out, key in zip(features[:-1], features[1:], keys)]
        self.matrices[0] = self.matrices[0]*s0
        
    def __call__(self, x, phi_k, k, a):
        f = x @ self.matrices[0] + self.biases[0]
        for i in range(1, len(self.matrices)):
            f = jnp.sin(f)
            f = f @ self.matrices[i] + self.biases[i]
        return jnp.sin(jnp.pi*x[0]) * x[1] * f[0] + get_ic(x[0], phi_k, k, a)

def compute_loss(model, coordinates, phi_k, k, f, a):
    du = vmap(lambda x: grad(model, argnums=0)(x, phi_k, k, a))(coordinates)
    d2u_dx2 = vmap(grad(lambda x: grad(model, argnums=0)(x, phi_k, k, a)[0]))(coordinates)
    l = du[:, 1] + a*du[:, 0] - d2u_dx2[:, 0] - f
    return jnp.linalg.norm(l)

def compute_error_energy_norm(model, coordinates, weights, phi_k, k, a, d_exact_sol, exact_sol, T, N_batch=16):
    du = []
    coordinates = coordinates.reshape(N_batch, -1, 2)
    for i in range(N_batch):
        du_ = vmap(lambda x: grad(model, argnums=0)(x, phi_k, k, a))(coordinates[i])
        du.append(du_)
    du = jnp.concatenate(du, 0)
    integrand = (du[:, 0] - d_exact_sol)**2
    l = jnp.sum(jnp.sum(integrand.reshape(weights.size, weights.size)*weights, axis=1) * weights[0])*T/4
    u_T = vmap(lambda x: model(x, phi_k, k, a))(coordinates.reshape(weights.size, weights.size, 2)[:, -1, :].reshape(-1, 2))
    l += jnp.sum(weights[0]*(u_T - exact_sol.reshape(weights.size, weights.size)[:, -1])**2)/4
    return jnp.sqrt(l)

compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

def make_step_scan(carry, ind, optim):
    model, opt_state, coordinates, phi_k, k, f, a = carry
    loss, grads = vmap(compute_loss_and_grads, in_axes=(0, None, 0, None, 0, 0))(model, coordinates[ind], phi_k, k, f[:, ind], a)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, opt_state, coordinates, phi_k, k, f, a], loss

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
    header = "N_batch,NN_batch,learning_rate,gamma,N_updates,N_drop,N_features,N_layers,energy_norm_mean,energy_norm_std,relative_error_mean,relative_error_std,final_loss_mean,final_loss_std,training_time"
    save_here = results_path + "residual.csv"
    if not os.path.isfile(save_here):
        with open(save_here, "w") as f:
            f.write(header)
            
    data = jnp.load(dataset_path)
    solution_train = data["solution_train"].reshape(1000, -1)
    dx_solution_train = data["dx_solution_train"].reshape(1000, -1)
    f_train = data["f_train"]
    f_train = jnp.stack([f_train]*f_train.shape[-2], 1).reshape(1000, -1)
    phi_train = data["phi_train"]
    phi_k = data["phi_k"]
    a = data["a"]
    solution_validation = data["solution_validation"].reshape(1000, -1)
    solution_leg = data["solution_leg"].reshape(1000, -1)
    dx_solution_leg = data["dx_solution_leg"].reshape(1000, -1)
    f_leg = data["f_leg"]
    f_leg = jnp.stack([f_leg]*f_leg.shape[-2], 1).reshape(1000, -1)
    coords_train = data["coords_train"].reshape(-1, 2)
    coords_validation = data["coords_validation"].reshape(-1, 2)
    coords_leg = data["coords_leg"].reshape(-1, 2)
    weights = data["weights"]
    k = data["k"]
    T = data["T"]
    
    key = random.PRNGKey(23)
    keys = random.split(key, 2)
    for learning_rate in learning_rates:
        for gamma in gammas:
            for N_drop in Ns_drop:
                for N_features_ in Ns_features:
                    for N_layers in Ns_layers:
                        M = N_batch_x*N_batch_x

                        inds = random.choice(keys[0], f_train.shape[-1], (N_updates, M))

                        N_features = [2, N_features_, 1]
                        NN_batch = N_batch_NN
                        keys = random.split(keys[1], NN_batch)
                        model = vmap(PiNN, in_axes=(None, None, 0, None))(N_features, N_layers, keys, 10)

                        sc = optax.exponential_decay(learning_rate, N_drop, gamma)
                        optim = optax.lion(learning_rate=sc)
                        opt_state = optim.init(eqx.filter(model, eqx.is_array))

                        carry = [model, opt_state, coords_train, phi_k[:NN_batch], k, f_train[:NN_batch], a[:NN_batch]]

                        make_step_scan_ = lambda a, b: make_step_scan(a, b, optim)

                        start = time.time()
                        carry, loss = scan(make_step_scan_, carry, inds)
                        stop = time.time()
                        training_time = stop - start
                        model = carry[0]

                        N_batch = 16
                        coords_validation_ = coords_validation.reshape(N_batch, -1, 2)
                        predictions = []
                        for i in range(N_batch):
                            predictions_ = vmap(vmap(lambda model, coords, phi, a: model(coords, phi, k, a), in_axes=(0, None, 0, 0)), in_axes=(None, 0, None, None), out_axes=1)(model, coords_validation_[i], phi_k[:NN_batch], a[:NN_batch])
                            predictions.append(predictions_)
                        predictions = jnp.concatenate(predictions, 1)
                        relative_errors = jnp.linalg.norm(predictions - solution_validation[:NN_batch], axis=1) / jnp.linalg.norm(solution_validation[:NN_batch], axis=1)
                        energy_norms = vmap(compute_error_energy_norm, in_axes=(0, None, None, 0, None, 0, 0, 0, None))(model, coords_leg, weights, phi_k[:NN_batch], k, a[:NN_batch], dx_solution_leg[:NN_batch], solution_leg[:NN_batch], T)

                        final_loss_mean = jnp.mean(loss, 1)[-1]
                        final_loss_std = jnp.sqrt(jnp.var(loss, 1))[-1]
                        relative_error_mean = jnp.mean(relative_errors)
                        relative_error_std = jnp.sqrt(jnp.var(relative_errors))
                        energy_norm_mean = jnp.mean(energy_norms)
                        energy_norm_std = jnp.sqrt(jnp.var(energy_norms))

                        res = f"\n{N_batch_x},{NN_batch},{learning_rate},{gamma},{N_updates},{N_drop},{N_features_},{N_layers},{energy_norm_mean},{energy_norm_std},{relative_error_mean},{relative_error_std},{final_loss_mean},{final_loss_std},{training_time}"
                        with open(save_here, "a") as f:
                            f.write(res)