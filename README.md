# MServer

A simple server built by FastAPI. 

```
export PYTHONPATH=/home/qianminj/qianminj123/MServer
```

To start the server loading a Flax Linear model with multi-processing:
```
fastapi run main_mp.py
```
To run a sample query:
```
curl -X GET  -H "Accept: Application/json" -H "Content-Type: application/json" http://127.0.0.1:8000/predictions/353 -d '{"input_val":[[1.0,1.0],[1.0,1.0],[1.0,1.0]]}'
```
The model is stored in a GCS bucket, the detailed path is in the config.py file.

The Flax linear model is exported by helper/export_linear_with_weight.py
