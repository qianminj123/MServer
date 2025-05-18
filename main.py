from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

from google.cloud import storage

from jax import export

def loadModel(bucket: str, blob: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(blob)

    with blob.open("rb") as f:
        ba = f.read()

    rehydrated_exp: export.Exported = export.deserialize(ba)
    return rehydrated_exp

rehydrated_exp = loadModel("qianminj-bucket", "exported_function1037")
app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None

class Prediction(BaseModel):
    input_val: int


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

    x = rehydrated_exp.call(float(prediction.input_val))

    print("output_val: {0}".format(x))

    return {"prediction_id": prediction_id, "output_val": str(x)}
