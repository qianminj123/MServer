# MServer

A simple server built by FastAPI. 

To start the server:
```
fastapi dev main.py
```

To run a sample query:
```
curl -X GET  -H "Accept: Application/json" -H "Content-Type: application/json" http://127.0.0.1:8000/models/353 -d '{"name":"simpleModel","bucket":"qianminj-bucket","blob":"exported_function1037","input_val":4}'
```

To start the server loading a Flax Linear model:
```
fastapi dev main_linear.py
```
To run a sample query:
```
curl -X GET  -H "Accept: Application/json" -H "Content-Type: application/json" http://127.0.0.1:8000/predictions/353 -d '{"input_val":[[1.0,1.0],[1.0,1.0],[1.0,1.0]]}'
```
