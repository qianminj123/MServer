from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel

from google.cloud import storage

from jax import export

app = FastAPI()


class Item(BaseModel):
    name: str
    price: float
    is_offer: Union[bool, None] = None

class Model(BaseModel):
    name: str
    bucket: str
    blob: str
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

@app.get("/models/{model_id}")
def run_model(model_id: int, model: Model):
    storage_client = storage.Client()
    bucket = storage_client.bucket(model.bucket)
    blob = bucket.blob(model.blob)

    with blob.open("rb") as f:
        ba = f.read()

    rehydrated_exp: export.Exported = export.deserialize(ba)

    print("input_val in float: {0}".format(float(model.input_val)))

    x = rehydrated_exp.call(float(model.input_val))

    print("output_val: {0}".format(x))

    model.input_val=x

    return {"model_name": model.name, "output_val": str(x)}
