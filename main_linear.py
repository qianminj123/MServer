from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

from google.cloud import storage

import jax
from jax import device_put, export
import jax.numpy as jnp

from model_manager import ModelManager

from tpu_manager import TpuManager

model_manager = ModelManager()
model_manager.load_model("qianminj-bucket", "exported_tpu_linear0238")
tpu_manager = TpuManager(jax.device_count())
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

    device_id = tpu_manager.acquire_one_tpu()
    
    a = device_put(jnp.array(prediction.input_val), jax.devices()[device_id])

    x = model_manager.get_exported().call(a)

    print("output_val: {0}".format(x))

    tpu_manager.release_one_tpu(device_id)

    return {"prediction_id": prediction_id, "output_val": str(x)}
