from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

from google.cloud import storage

from mserver.tpu_manager import TpuManager
from mserver.model_manager import ModelManager
from mserver.model_runner import ModelRunner

import time

import os

os.environ['JAX_PLATFORMS']="tpu"
os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"
os.environ["TPU_PROCESS_BOUNDS"] = "1,1,1"
os.environ["TPU_VISIBLE_DEVICES"] = "0"

model_runner = ModelRunner()

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None

class Prediction(BaseModel):
    input_val: list


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}


@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}

@app.get("/predictions/{prediction_id}")
def run_model(prediction_id: int, prediction: Prediction):
    x = model_runner.run(prediction.input_val)
    return {"prediction_id": prediction_id, "output_val": str(x)}
