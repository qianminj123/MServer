import numpy as np
import jax
from jax import export

from google.cloud import storage

def f(x): return 2 * x * x

exported: export.Exported = export.export(jax.jit(f))(
        jax.ShapeDtypeStruct((), np.float32))

print("exported.fun_name: {0}".format(exported.fun_name))

print("exported.mlir_module()")

#print(exported.mlir_module())


serialized: bytearray = exported.serialize()

print("serialized: ")
print(serialized)


storage_client = storage.Client()

bucket_name="qianminj-bucket"

bucket = storage_client.bucket(bucket_name)
blob = bucket.blob("exported_function1037")

with blob.open('wb') as f:
    f.write(serialized)


