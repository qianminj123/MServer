from google.cloud import storage

import jax
from jax import export

def run_model(gcs_path: str):
    storage_client = storage.Client()
    bucket = storage_client.bucket("qianminj-bucket")
    blob = bucket.blob("exported_function1037")

    with blob.open("rb") as f:
        ba = f.read()

    rehydrated_exp: export.Exported = export.deserialize(ba)

    x = rehydrated_exp.call(4.0)

    return x


