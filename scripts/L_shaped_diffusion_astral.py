import argparse
import os.path

import jax.numpy as jnp
import equinox as eqx
import time
import optax

from jax.nn import gelu
from jax.lax import scan
from jax import random, jit, vmap, grad

@jit
def distance_function(x, R):
    r = (x - R)
    w = 0
    for i in range(6):
        ri, rip1 = jnp.linalg.norm(r[i]), jnp.linalg.norm(r[i+1])
        t = (r[i][0] * r[i+1][1] - r[i][1]*r[i+1][0]) / (ri*rip1 + jnp.sum(r[i]*r[i+1]))
        w = w + t * (1 / ri + 1 / rip1)
    w = 1 / w
    return w

class PiNN(eqx.Module):
    matrices: list
    biases: list
    
    def __init__(self, N_features, N_layers, key, s0=10):
        keys = random.split(key, N_layers+1)
        features = [N_features[0],] + [N_features[1],]*(N_layers-1) + [N_features[-1],]
        self.matrices = [random.uniform(key, (f_in, f_out), minval=-1, maxval=1)*jnp.sqrt(6/f_in) for f_in, f_out, key in zip(features[:-1], features[1:], keys)]
        keys = random.split(keys[-1], N_layers)
        self.biases = [jnp.zeros((f_out,)) for f_in, f_out, key in zip(features[:-1], features[1:], keys)]
        self.matrices[0] = self.matrices[0]*s0

    def __call__(self, x, R):
        f = x @ self.matrices[0] + self.biases[0]
        for i in range(1, len(self.matrices)):
            f = jnp.sin(f)
            f = f @ self.matrices[i] + self.biases[i]
        return f[0] * distance_function(x, R)

class PiNNu(eqx.Module):
    matrices: list
    biases: list

    def __init__(self, N_features, N_layers, key, s0=10):
        keys = random.split(key, N_layers+1)
        features = [N_features[0],] + [N_features[1],]*(N_layers-1) + [N_features[-1],]
        self.matrices = [random.uniform(key, (f_in, f_out), minval=-1, maxval=1)*jnp.sqrt(6/f_in) for f_in, f_out, key in zip(features[:-1], features[1:], keys)]
        keys = random.split(keys[-1], N_layers)
        self.biases = [jnp.zeros((f_out,)) for f_in, f_out, key in zip(features[:-1], features[1:], keys)]
        self.matrices[0] = self.matrices[0]*s0

    def __call__(self, x, R):
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
        self.beta = jnp.array([1.0,])

    def __call__(self, x, R, i):
        return self.models[i](x, R)

def get_flux(model, x, R):
    return grad(model, argnums=0)(x, R, 2)

def get_laplacian(model, x, R):
    return grad(lambda x: grad(model, argnums=0)(x, R, 2)[0])(x)[0] + grad(lambda x: grad(model, argnums=0)(x, R, 2)[1])(x)[1]

def compute_loss(model, coordinates, b2, rhs, R):
    C_F = 1 / (2*jnp.pi)
    flux = vmap(get_flux, in_axes=(None, 0, None))(model, coordinates, R)
    dx_y = [vmap(grad(lambda x: model(x, R, i), argnums=0), in_axes=0)(coordinates)[:, i] for i in [0, 1]]
    y1 = vmap(model, in_axes=(0, None, None))(coordinates, R, 0)
    y2 = vmap(model, in_axes=(0, None, None))(coordinates, R, 1)
    u = vmap(model, in_axes=(0, None, None))(coordinates, R, 2)
    loss = (1 + model.beta[0]**2)*(C_F**2*(rhs - b2*u + dx_y[0] + dx_y[1])**2 / (C_F**2*b2*(1 + model.beta[0]**2) + 1) +  ((flux[:, 0] - y1)**2 + (flux[:, 1] - y2)**2) / (model.beta[0]**2))
    return jnp.mean(loss)

def compute_upper_bound(model, coordinates, weights, b2, rhs, R, N_batch=10):
    C_F = 1 / (2*jnp.pi)
    flux = []
    dx_y_0 = []
    dx_y_1 = []
    y1 = []
    y2 = []
    u = []
    coordinates = coordinates.reshape(N_batch, -1, 2)
    for j in range(N_batch):
        flux_ = vmap(get_flux, in_axes=(None, 0, None), out_axes=1)(model, coordinates[j], R)
        dx_y_ = [vmap(grad(lambda x: model(x, R, i), argnums=0), in_axes=0)(coordinates[j])[:, i] for i in [0, 1]]
        y1_ = vmap(model, in_axes=(0, None, None))(coordinates[j], R, 0)
        y2_ = vmap(model, in_axes=(0, None, None))(coordinates[j], R, 1)
        u_ = vmap(model, in_axes=(0, None, None))(coordinates[j], R, 2)
        u.append(u_)
        flux.append(flux_)
        dx_y_0.append(dx_y_[0])
        dx_y_1.append(dx_y_[1])
        y1.append(y1_)
        y2.append(y2_)
    flux = jnp.concatenate(flux, 1)
    dx_y = [jnp.concatenate(dx_y_0, 0), jnp.concatenate(dx_y_1, 0)]
    y1 = jnp.concatenate(y1, 0)
    y2 = jnp.concatenate(y2, 0)
    u = jnp.concatenate(u, 0)
    integrand = (1 + model.beta[0]**2)*(C_F**2*(rhs - b2*u + dx_y[0] + dx_y[1])**2 / (C_F**2*b2*(1 + model.beta[0]**2) + 1) +  ((flux[0] - y1)**2 + (flux[1] - y2)**2) / (model.beta[0]**2))
    l = jnp.sum(jnp.sum(integrand.reshape(weights.size, weights.size)*weights, axis=1) * weights[0]) / 16
    return l

def compute_error_energy_norm(model, coordinates, weights, b2, rhs, sol, R, N_batch=10):
    laplacian = []
    u = []
    coordinates = coordinates.reshape(N_batch, -1, 2)
    for i in range(N_batch):
        laplacian_ = vmap(get_laplacian, in_axes=(None, 0, None))(model, coordinates[i], R)
        u_ = vmap(model, in_axes=(0, None, None))(coordinates[i], R, 2)
        laplacian.append(laplacian_)
        u.append(u_)
    laplacian = jnp.concatenate(laplacian, 0)
    u = jnp.concatenate(u, 0)
    r = rhs - b2*u + laplacian
    integrand = jnp.abs((sol - u) * r)
    l = jnp.sum(jnp.sum(integrand.reshape(weights.size, weights.size)*weights, axis=1) * weights[0]) / 16
    return l


compute_loss_and_grads = eqx.filter_value_and_grad(compute_loss)

def make_step_scan(carry, ind, optim):
    model, coordinates, b2, rhs, R, opt_state = carry
    loss, grads = vmap(compute_loss_and_grads, in_axes=(0, None, 0, 0, None))(model, coordinates[ind], b2[:, ind], rhs[:, ind], R)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, coordinates, b2, rhs, R, opt_state], loss

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
    header = "N_batch,NN_batch,learning_rate,gamma,N_updates,N_drop,N_features,N_layers,energy_norm_mean,energy_norm_std,relative_error_mean,relative_error_std,final_loss_mean,final_loss_std,training_time,upper_bound_mean,upper_bound_std"
    save_here = results_path + "astral.csv"
    if not os.path.isfile(save_here):
        with open(save_here, "w") as f:
            f.write(header)
            
    data = jnp.load(dataset_path)
    solution_train = data["solution_train"]
    solution_validation = data["solution_validation"]
    solution_leg = data["solution_leg"]
    coords_train = data["coords_train"]
    coords_leg = data["coords_leg"]
    coords_validation = data["coords_validation"]
    b2_train = data["b2_train"]
    b2_leg = data["b2_leg"]
    f_train = data["f_train"]
    f_leg = data["f_leg"]
    weights = data["weights"]
    
    eps = 1e-4
    R = jnp.array([
        [-eps, -eps],
        [1.0+eps, -eps],
        [1.0+eps, 0.5+eps],
        [0.5+eps, 0.5+eps],
        [0.5+eps, 1.0+eps],
        [-eps, 1.0+eps],
        [-eps, -eps]
    ])
    
    key = random.PRNGKey(23)
    keys = random.split(key, 2)
    for learning_rate in learning_rates:
        for gamma in gammas:
            for N_drop in Ns_drop:
                for N_features_ in Ns_features:
                    for N_layers in Ns_layers:
                        M = N_batch_x*N_batch_x
                        inds = random.choice(keys[0], coords_train.shape[0], (N_updates, M))

                        N_features = [2, N_features_, 1]
                        NN_batch = N_batch_NN
                        keys = random.split(keys[1], NN_batch)
                        model = vmap(PiNN3, in_axes=(None, None, 0))(N_features, N_layers, keys)

                        sc = optax.exponential_decay(learning_rate, N_drop, gamma)
                        optim = optax.lion(learning_rate=sc)
                        opt_state = optim.init(eqx.filter(model, eqx.is_array))

                        carry = [model, coords_train, b2_train[:NN_batch], f_train[:NN_batch], R, opt_state]

                        make_step_scan_ = lambda a, b: make_step_scan(a, b, optim)

                        start = time.time()
                        carry, loss = scan(make_step_scan_, carry, inds)
                        stop = time.time()
                        training_time = stop - start
                        model = carry[0]

                        predicted_ = []
                        n_batch = 9
                        coords_validation_ = coords_validation.reshape(n_batch, -1, 2)
                        eval_model = jit(lambda model, coordinates: vmap(model, in_axes=(0, None, None))(coordinates, R, 2))
                        for i in range(n_batch):
                            predicted = vmap(lambda model, coords: eval_model(model, coords), in_axes=(0, None))(model, coords_validation_[i])
                            predicted_.append(predicted)
                        predicted = jnp.concatenate(predicted_, 1)
                        relative_errors = jnp.linalg.norm(predicted - solution_validation[:NN_batch], axis=1) / jnp.linalg.norm(solution_validation[:NN_batch], axis=1)

                        energy_norms = 0
                        for i in range(3):
                            energy_norms += vmap(compute_error_energy_norm, in_axes=(0, None, None, 0, 0, 0, None))(model, coords_leg[i], weights, b2_leg[:NN_batch, i], f_leg[:NN_batch, i], solution_leg[:NN_batch, i], R)
                        energy_norms = jnp.sqrt(energy_norms)

                        upper_bounds = 0
                        for i in range(3):
                            upper_bounds += vmap(compute_upper_bound, in_axes=(0, None, None, 0, 0, None))(model, coords_leg[i], weights, b2_leg[:NN_batch, i], f_leg[:NN_batch, i], R)
                        upper_bounds = jnp.sqrt(upper_bounds)

                        final_loss_mean = jnp.mean(loss, 1)[-1]
                        final_loss_std = jnp.sqrt(jnp.var(loss, 1))[-1]
                        relative_error_mean = jnp.mean(relative_errors)
                        relative_error_std = jnp.sqrt(jnp.var(relative_errors))
                        energy_norm_mean = jnp.mean(energy_norms)
                        energy_norm_std = jnp.sqrt(jnp.var(energy_norms))
                        upper_bound_mean = jnp.mean(upper_bounds)
                        upper_bound_std = jnp.sqrt(jnp.var(upper_bounds))

                        res = f"\n{N_batch_x},{NN_batch},{learning_rate},{gamma},{N_updates},{N_drop},{N_features_},{N_layers},{energy_norm_mean},{energy_norm_std},{relative_error_mean},{relative_error_std},{final_loss_mean},{final_loss_std},{training_time},{upper_bound_mean},{upper_bound_std}"
                        with open(save_here, "a") as f:
                            f.write(res)