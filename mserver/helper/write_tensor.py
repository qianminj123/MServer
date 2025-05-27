import tensorstore as ts
# import numpy as np

dataset = ts.open({
     'driver': 'n5',
     'kvstore': {
         'driver': 'file',
         'path': 'tmp/dataset/',
     },
     'metadata': {
         'compression': {
             'type': 'gzip'
         },
         'dataType': 'uint32',
         'dimensions': [20, 20],
         'blockSize': [100, 100],
     },
     'create': True,
     'delete_existing': True,
 }, read=False, write=True).result()

write_future = dataset[10:12, 16:19].write([[1,2,3],[4,5,6]])
x = write_future.result()
print(x)
#print(dataset.read().result())
