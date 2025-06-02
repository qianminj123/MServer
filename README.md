# MServer

A simple server built by FastAPI. 

```
export PYTHONPATH=/home/qianminj/qianminj123/MServer
```

To start the server:
```
fastapi run main.py
```

To run a sample query:
```
curl -X GET  -H "Accept: Application/json" -H "Content-Type: application/json" http://127.0.0.1:8000/predictions/353 -d '{"input_val":2.0}'
{"prediction_id":353,"output_val":"8.0"}
```

To start the server loading a Flax Linear model:
```
fastapi run main_linear.py
```
To run a sample query:
```
curl -X GET  -H "Accept: Application/json" -H "Content-Type: application/json" http://127.0.0.1:8000/predictions/353 -d '{"input_val":[[1.0,1.0],[1.0,1.0],[1.0,1.0]]}'
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
