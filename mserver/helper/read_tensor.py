import tensorstore as ts
# import numpy as np

dataset = ts.open({
     'driver': 'n5',
     'kvstore': {
         'driver': 'file',
         'path': 'tmp/dataset/',
     },
}, read=True, write=False).result()

print(dataset)
#read_future = dataset.read()
#x = read_future.result()
#print(x)

print(dataset[:, :].read().result())
