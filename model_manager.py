from google.cloud import storage
from jax import export

class ModelManager:
    _rehydrated_exp = None
    def load_model(self, gcs_bucket: str, blob:str):
        storage_client = storage.Client()
        blob = storage_client.bucket(gcs_bucket).blob(blob)

        with blob.open("rb") as f:
            byte_array = f.read()
        print("Loading a model")
        self._rehydrated_exp: export.Exported = export.deserialize(byte_array)

    def get_exported(self):
        return self._rehydrated_exp

    def export_model(exported: export.Exported, gcs_bucket: str, blob: str):
        serialized: bytearray = exported.serialize()
        storage_client = storage.Client()
        blob = storage_client.bucket(gcs_bucket).blob(blob)

        with blob.open('wb') as f:
            f.write(serialized)


