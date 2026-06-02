import jax.numpy as jnp
x1 = jnp.ones((2, 10, 1))
mu1 = jnp.ones((2, 10, 5))
mu2 = jnp.ones((2, 4, 5))
try:
    jnp.concatenate([x1, mu1, mu2], axis=-1)
    print("Success")
except Exception as e:
    print("Error:", e)
