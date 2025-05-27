from flax import nnx
import jax
from jax import export
import jax.numpy as jnp
import numpy as np
import tensorstore as ts

from google.cloud import storage

class Linear(nnx.Module):
  def __init__(self, din: int, dout: int, *, rngs: nnx.Rngs):
    key = rngs.params()
    self.w = jax.random.uniform(key, (din, dout))
    self.b = jnp.zeros((dout,))
    self.din, self.dout = din, dout

  def upload_weight(self):
    dataset = ts.open({
      'driver': 'n5',
      'kvstore': {
        'driver': 'gcs',
        'bucket': 'qianminj-bucket',
        'path': 'tmp/dataset3/',
      },
      'metadata': {
        'compression': {
          'type': 'gzip'
        },
        'dataType': 'float32',
        'dimensions': [self.din, self.dout],
        'blockSize': [100, 100],
       },
       'create': True,
       'delete_existing': True,
    }, read=False, write=True).result()
    print("w to write:")
    print(self.w)
    write_future = dataset.write(self.w)
    print(write_future.result())

  def model(self, w:jax.Array, x:jax.Array):
      w = nnx.Param(w)
      return x @ w + self.b

  def export_model(self):
    x_shape = (3, self.din)
    w_shape = (self.din, self.dout)
    exported: export.Exported = export.export(jax.jit(self.model))(
        jax.ShapeDtypeStruct(w_shape, np.float32),
        jax.ShapeDtypeStruct(x_shape, np.float32))
    print("exported.fun_name: {0}".format(exported.fun_name))
    print("exported.mlir_module()")
    serialized: bytearray = exported.serialize()

    print("serialized: {0}".format(serialized))

    storage_client = storage.Client()
    bucket_name = "qianminj-bucket"
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob("exported_tpu_linear1015")

    with blob.open('wb') as f:
        f.write(serialized)

linear=Linear(2, 5, rngs=nnx.Rngs(params=0))
linear.export_model()
linear.upload_weight()
