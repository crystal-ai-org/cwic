import os
import jax
from jax.experimental import mesh_utils
from jax.sharding import Mesh

def process_is_leader():
    return jax.process_index() == 0

devices = mesh_utils.create_device_mesh(
    (jax.device_count(), 1),
)
global_mesh = Mesh(devices, axis_names=("dp", "mp"))


def set_global_mesh(x):
    global global_mesh
    global_mesh = x
