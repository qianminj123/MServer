import tensorstore as ts
import jax
from flax import nnx

dataset = ts.open({
     'driver': 'n5',
     'kvstore': {
         'driver': 'gcs',
         'bucket': 'qianminj-bucket',
         'path': 'tmp/dataset2/',
     },
     'metadata': {
         'compression': {
             'type': 'gzip'
         },
         'dataType': 'float32',
         'dimensions': [2, 5],
         'blockSize': [100, 100],
     },
     'create': True,
     'delete_existing': True,
 }, read=False, write=True).result()

val = jax.random.uniform(nnx.Rngs(params=0).params(), (2,5))
print(val)
write_future = dataset.write(val)
x = write_future.result()
print(x)
#print(dataset.read().result())
