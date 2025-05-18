from flax import nnx
import jax
from jax import export
import jax.numpy as jnp
import numpy as np

from google.cloud import storage

class Linear(nnx.Module):
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    key = rngs.params()
    self.w = nnx.Param(jax.random.uniform(key, (din, dout)))
    self.b = nnx.Param(jnp.zeros((dout,)))
    self.din, self.dout = din, dout

  def __call__(self, x: jax.Array):
    return x @ self.w + self.b

linear=Linear(2, 5, rngs=nnx.Rngs(params=0))

a, b = export.symbolic_shape("a, b")
x_shape = (3, 2)

exported: export.Exported = export.export(jax.jit(linear))(
        jax.ShapeDtypeStruct(x_shape, np.float32))

print("exported.fun_name: {0}".format(exported.fun_name))

print("exported.mlir_module()")

serialized: bytearray = exported.serialize()

print("serialized: {0}".format(serialized))

storage_client = storage.Client()

bucket_name = "qianminj-bucket"

bucket = storage_client.bucket(bucket_name)
blob = bucket.blob("exported_linear1012")

with blob.open('wb') as f:
    f.write(serialized)
