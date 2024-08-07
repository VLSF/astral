import jax.numpy as jnp
import equinox as eqx
import optax
import itertools
import time
import os
import sys

from jax.nn import gelu
from jax.lax import scan
from jax import random, jit, vmap, grad, jacfwd

from jax.tree_util import tree_map
from collections import defaultdict

def compute_solution(coords, c1, c2):
    projector = jnp.prod(jnp.sin(jnp.pi*coords), axis=1)
    t1 = coords @ c1
    t2 = coords @ c2
    return jnp.sin(2*jnp.pi*t1 + jnp.cos(2*jnp.pi*t2))*projector

def compute_d_solution(coords, c1, c2, i):
    projector = jnp.prod(jnp.sin(jnp.pi*coords), axis=1)
    t1 = coords @ c1
    t2 = coords @ c2
    phi = jnp.sin(2*jnp.pi*t1 + jnp.cos(2*jnp.pi*t2))*projector
    d_phi = jnp.pi*phi / jnp.tan(jnp.pi*coords[:, i]) + (2*jnp.pi*c1[i] - 2*jnp.pi*c2[i]*jnp.sin(2*jnp.pi*t2))*projector*jnp.cos(2*jnp.pi*t1 + jnp.cos(2*jnp.pi*t2))
    return d_phi

def compute_f(coords, c1, c2):
    projector = jnp.prod(jnp.sin(jnp.pi*coords), axis=1)
    t1 = coords @ c1
    t2 = coords @ c2
    phi = jnp.sin(2*jnp.pi*t1 + jnp.cos(2*jnp.pi*t2))*projector
    f = jnp.pi**2*(c1.size + 4*jnp.sum((c1 - c2*jnp.sin(2*jnp.pi*jnp.expand_dims(t2, 1)))**2, axis=1))*phi
    f += (2*jnp.pi)**2*(jnp.sum(c2**2)*jnp.cos(2*jnp.pi*t2) - jnp.sum(1/jnp.tan(jnp.pi*coords)*(c1 - c2*jnp.sin(2*jnp.pi*jnp.expand_dims(t2, axis=1))), axis=1))*jnp.cos(2*jnp.pi*t1 + jnp.cos(2*jnp.pi*t2))*projector
    return f

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

    def __call__(self, x):
        f = x @ self.matrices[0] + self.biases[0]
        for i in range(1, len(self.matrices)):
            f = jnp.sin(f)
            f = f @ self.matrices[i] + self.biases[i]
        return jnp.prod(jnp.sin(jnp.pi*x)) * f[0]

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

    def __call__(self, x):
        f = x @ self.matrices[0] + self.biases[0]
        for i in range(1, len(self.matrices)):
            f = jnp.sin(f)
            f = f @ self.matrices[i] + self.biases[i]
        return f[0]

class PiNNd(eqx.Module):
    models: list
    beta: jnp.array

    def __init__(self, N_features, N_layers, key):
        keys = random.split(key, 1+N_features[0])
        self.models = [PiNN(N_features, N_layers, keys[0])]
        self.models += [PiNNu(N_features, N_layers, key) for key in keys[1:]]
        self.beta = jnp.ones((1,))

    def __call__(self, x, i):
        return self.models[i](x)


def Astral_loss(model, x, c1, c2):
    C = 1/(jnp.pi*len(c1))
    R = compute_f(x, c1, c2)
    m = []
    for i in range(x.shape[1]):
        R += vmap(grad(lambda x: model(x, i+1)))(x)[:, i]
        m.append(vmap(model, in_axes=(0, None))(x, i+1))
    m = jnp.stack(m, 1)
    L = jnp.mean(jnp.sum((m - vmap(grad(lambda x: model(x, 0)))(x))**2, axis=1))*(1+model.beta[0])/model.beta[0]
    L += jnp.mean(R**2)*C**2*(1+model.beta[0])
    return L

compute_loss_and_grads = eqx.filter_value_and_grad(Astral_loss)

def make_step_scan(carry, coord, optim):
    model, c1, c2, opt_state = carry
    loss, grads = compute_loss_and_grads(model, coord, c1, c2)
    updates, opt_state = optim.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)
    return [model, c1, c2, opt_state], loss

def compute_relative_error(model, x, c1, c2):
    return jnp.sqrt(jnp.sum((vmap(model, in_axes=(0, None))(x, 0) - compute_solution(x, c1, c2))**2) / jnp.sum(compute_solution(x, c1, c2)**2))

def compute_energy_norm(model, x, c1, c2):
    d_sol = jnp.stack([compute_d_solution(x, c1, c2, i) for i in range(len(c1))], 1)
    d_approximate_sol = vmap(grad(lambda x: model(x, 0)))(x)
    return jnp.sqrt(jnp.mean(jnp.sum((d_sol - d_approximate_sol)**2, axis=1)))

def train(key, d, N_features_, N_layers, N_batch, N_run, N_drop, gamma_, learning_rate, N_estimate, eps):
    keys = random.split(key, 6)
    c1 = random.normal(keys[0], (d,))
    c2 = random.normal(keys[1], (d,))
    model = PiNNd([d, N_features_, 1], N_layers, keys[2])
    coords = random.uniform(key, (N_run, N_batch, d))*(1-eps) + eps
    sc = optax.exponential_decay(learning_rate, N_drop, gamma_)
    optim = optax.lion(learning_rate=sc)
    opt_state = optim.init(eqx.filter(model, eqx.is_array))

    carry = [model, c1, c2, opt_state]

    make_step_scan_ = lambda a, b: make_step_scan(a, b, optim)

    start = time.time()
    carry, loss = scan(make_step_scan_, carry, coords)
    stop = time.time()
    training_time = stop - start
    model = carry[0]

    coords = random.uniform(keys[3], (N_estimate, d))*(1-eps) + eps
    relative_error = compute_relative_error(model, coords, c1, c2)
    
    coords = random.uniform(keys[4], (N_estimate, d))*(1-eps) + eps
    energy_norm = compute_energy_norm(model, coords, c1, c2)
    
    coords = random.uniform(keys[5], (N_estimate, d))*(1-eps) + eps
    upper_bound = jnp.sqrt(Astral_loss(model, coords, c1, c2))
    return training_time, model, loss, relative_error, energy_norm, upper_bound

if __name__ == "__main__":
    d = int(sys.argv[1])
    N_run = 50000
    N_batch = 16*16
    gamma_ = 0.5
    N_trials = 7
    eps = 1e-5
    N_estimate = 5000

    N_F = [50, 100]
    N_L = [3, 4, 5]
    N_D = [1000, 25000, 50000]
    L_R = [1e-4, 5e-3, 1e-3]
    
    key = random.PRNGKey(33)
    if not os.path.isfile("large_d_results/astral_split.csv"):
        with open("large_d_results/astral_split.csv", "w") as f:
            f.write("d,N_features,N_layers,N_drop,learning_rate,final_loss,relative_error,energy_norm,upper_bound,training_time,npz")
            
    for N_features_, N_layers, N_drop, learning_rate in itertools.product(N_F, N_L, N_D, L_R):
        npz = hash("".join(map(str, [d, N_features_, N_layers, N_drop, learning_rate]))+"astral_split")
        training_time = []
        loss = []
        relative_error = []
        energy_norm = []
        upper_bound = []
        for _ in range(N_trials):
            key = random.split(key)[0]
            training_time_, _, loss_, relative_error_, energy_norm_, upper_bound_ = train(key, d, N_features_, N_layers, N_batch, N_run, N_drop, gamma_, learning_rate, N_estimate, eps)
            training_time.append(training_time_)
            loss.append(loss_)
            relative_error.append(relative_error_)
            energy_norm.append(energy_norm_)
            upper_bound.append(upper_bound_)
        training_time = jnp.array(training_time)
        loss = jnp.array(loss)
        relative_error = jnp.array(relative_error)
        energy_norm = jnp.array(energy_norm)
        upper_bound = jnp.array(upper_bound)
        with open("large_d_results/astral_split.csv", "a") as f:
            f.write(f"\n{d},{N_features_},{N_layers},{N_drop},{learning_rate},{jnp.mean(loss[:, -1])},{jnp.mean(relative_error)},{jnp.mean(energy_norm)},{jnp.mean(upper_bound)},{jnp.mean(training_time)},{npz}")
        data = {
            "training_time": training_time,
            "loss": loss,
            "relative_error": relative_error,
            "energy_norm": energy_norm,
            "upper_bound": energy_norm
        }
        jnp.savez(f"large_d_results/{npz}.npz", **data)