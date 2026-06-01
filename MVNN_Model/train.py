"""
Optimized DeepMD code script.
"""

# Import common packages
from pathlib import Path
from typing import Sequence, Callable
import numpy as np
# Import JAX related packages
import jax
import jax.numpy as jnp
from jax import random, vmap
# Debugging packages; uncomment if needed
from jax.config import config
config.update("jax_enable_x64", True)
# config.update("jax_disable_jit", True)
# config.update("jax_debug_nans", True)
import pdb
# Flax imports
import flax
from flax import linen as nn
from flax.training import train_state
from flax import jax_utils

# Import Optax for optimization
import optax
# import pdb
# Import numpy and other utility libraries
import numpy as onp
import functools
import glob
import sys
from model import create_model


# Network settings
DIM = 1
TRAIN_ITERATION = 10000
BATCH_SIZE = 500
DT = 1e-3
FEATURE_NET_LAYER = [12,24, 48]
DESCRIBE_NET_LAYER = [128, 128, 128, DIM]
SAVE_PATH = Path(f"model_save_path")
SAVE_PATH.mkdir(parents=True, exist_ok=True)




x_worker_save = None
v_worker_save = None
x_manager_save = None
v_manager_save = None
x_ceo_save = None
v_ceo_save = None

for i in range(1,500):
    print(i)
    data_workers = np.load(f"../DATA_GENERATION3/data/opinion_workers_mpi_{i}.npy")[...,None]
    data_managers = np.load(f"../DATA_GENERATION3/data/opinion_managers_mpi_{i}.npy")[...,None]
    data_ceos = np.load(f"../DATA_GENERATION3/data/opinion_ceos_mpi_{i}.npy")[...,None]
    x_worker = data_workers[:-1]
    v_worker = (data_workers[1:]-data_workers[:-1])/DT
    x_manager = data_managers[:-1]
    v_manager = (data_managers[1:]-data_managers[:-1])/DT
    x_ceo = data_ceos[:-1]
    v_ceo = (data_ceos[1:]-data_ceos[:-1])/DT
    if x_worker_save is None:
        x_worker_save = x_worker
        v_worker_save = v_worker
        x_manager_save = x_manager
        v_manager_save = v_manager
        x_ceo_save = x_ceo
        v_ceo_save = v_ceo
    else:
        x_worker_save = np.concatenate([x_worker_save, x_worker], axis=0)
        v_worker_save = np.concatenate([v_worker_save, v_worker], axis=0)
        x_manager_save = np.concatenate([x_manager_save, x_manager], axis=0)
        v_manager_save = np.concatenate([v_manager_save, v_manager], axis=0)
        x_ceo_save = np.concatenate([x_ceo_save, x_ceo], axis=0)
        v_ceo_save = np.concatenate([v_ceo_save, v_ceo], axis=0)
        
rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)        
perms = jax.random.permutation(init_rng, x_worker_save.shape[0])
 
        
TRAIN_DATA = {
    "x_worker_save":x_worker_save[perms[:-BATCH_SIZE]],
    "v_worker_save":v_worker_save[perms[:-BATCH_SIZE]],
    "x_manager_save":x_manager_save[perms[:-BATCH_SIZE]],
    "v_manager_save":v_manager_save[perms[:-BATCH_SIZE]],
    "x_ceo_save":x_ceo_save[perms[:-BATCH_SIZE]],
    "v_ceo_save":v_ceo_save[perms[:-BATCH_SIZE]]
}
TEST_DATA = {
    "x_worker_save":x_worker_save[perms[-BATCH_SIZE:]],
    "v_worker_save":v_worker_save[perms[-BATCH_SIZE:]],
    "x_manager_save":x_manager_save[perms[-BATCH_SIZE:]],
    "v_manager_save":v_manager_save[perms[-BATCH_SIZE:]],
    "x_ceo_save":x_ceo_save[perms[-BATCH_SIZE:]],
    "v_ceo_save":v_ceo_save[perms[-BATCH_SIZE:]]
}




# Initialize the training state
def create_train_state(rng):
    learning_rate_fn = optax.exponential_decay(
        init_value=1e-3,
        transition_steps=100000,
        decay_rate=0.9,
        end_value =1e-7
    )

    cf_init,fcn_f = create_model(FEATURE_NET_LAYER, DESCRIBE_NET_LAYER, DIM)
    params = cf_init(rng)

    tx = optax.adam(learning_rate=learning_rate_fn)
    state = train_state.TrainState.create(
        apply_fn = fcn_f,
        params = params,
        tx = tx
    )

    return state



def update_model(state, grads):
    return state.apply_gradients(grads=grads)

# Define the function to apply the model
def apply_model(state,x_worker_save_,v_worker_save_,x_manager_save_,v_manager_save_,x_ceo_save_,v_ceo_save_):
    def loss_fn(params):
        # input = jnp.concatenate([fft_data_,grad_fft_data_],axis=-1)
        f_save_1, f_save_2, f_save_3 = state.apply_fn(params, x_worker_save_,x_manager_save_,x_ceo_save_)
        loss1 = jnp.sqrt(jnp.sum((v_worker_save_ - f_save_1)**2))/jnp.sqrt(jnp.sum(v_worker_save_**2))
        loss2 =  jnp.sqrt(jnp.sum((v_manager_save_ - f_save_2)**2))/jnp.sqrt(jnp.sum(v_manager_save_**2))
        loss3 =  jnp.sqrt(jnp.sum((v_ceo_save_ - f_save_3)**2))/jnp.sqrt(jnp.sum(v_ceo_save_**2))
        return (loss1 + loss2 + loss3) / 3, (loss1, loss2, loss3)

    # Compute loss and gradients
    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss_val,aux), grads,  = loss_grad_fn(state.params)

    return loss_val, grads, aux



def train_epoch(state, train_ds, batch_size, rng):
    train_ds_size = train_ds["x_worker_save"].shape[0]
    steps_per_epoch = train_ds_size // batch_size
    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size].reshape((steps_per_epoch, batch_size))
    epoch_loss = []
    for perm in perms:
        batch_data = {
            f"{key}_": train_ds[key][perm]
            for key in train_ds
        }
        loss_val, grads, _ = apply_model(state, **batch_data)
        state = update_model(state, grads)
        epoch_loss.append(loss_val)

    return state, onp.mean(epoch_loss)

def test_epoch(state, train_ds, batch_size, rng):
    train_ds_size = train_ds["x_worker_save"].shape[0]
    steps_per_epoch = train_ds_size // batch_size
    perms = jax.random.permutation(rng, train_ds_size)
    perms = perms[:steps_per_epoch * batch_size].reshape((steps_per_epoch, batch_size))
    epoch_loss = []
    epoch_loss1 = []
    epoch_loss2 = []
    epoch_loss3 = []
    for perm in perms:
        batch_data = {
            f"{key}_": train_ds[key][perm]
            for key in train_ds
        }
        loss_val, grads, aux = apply_model(state, **batch_data)
        epoch_loss.append(loss_val)
        epoch_loss1.append(aux[0])
        epoch_loss2.append(aux[1])
        epoch_loss3.append(aux[2])
    return onp.mean(epoch_loss), onp.mean(epoch_loss1), onp.mean(epoch_loss2), onp.mean(epoch_loss3)


# Initialize the random number generator


state = create_train_state(init_rng)

error_save = onp.zeros(TRAIN_ITERATION)

# Start training
for t in range(TRAIN_ITERATION):
    rng, input_rng = jax.random.split(rng)
    state, loss_val = train_epoch(state, TRAIN_DATA,BATCH_SIZE,input_rng)
    error_save[t] = loss_val
    if t % 10 == 0:
        onp.save(SAVE_PATH/ "error.npy",error_save)
        ckpt_filename = SAVE_PATH / f'jax_ckpt_{t:06d}.npz'
        params = jax.device_get(state.params)
        with open(ckpt_filename, 'wb') as f:
            jnp.savez(f,t=t,params=params)
        error_val, error_val1, error_val2, error_val3 = test_epoch(state, TEST_DATA,BATCH_SIZE,input_rng)   
        print('step: {}, loss: {}, error_total: {}, error_worker: {}, error_manager: {}, error_ceo: {}'.format(state.step,loss_val,error_val,error_val1,error_val2,error_val3))
