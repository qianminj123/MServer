from google.cloud import storage
from jax import export
import tensorstore as ts

class ModelManager:
    _rehydrated_exp = None
    _weight = None
    def load_model(self, gcs_bucket: str, model_blob:str, gcs_weight_path: str):
        storage_client = storage.Client()
        blob = storage_client.bucket(gcs_bucket).blob(model_blob)

        with blob.open("rb") as f:
            byte_array = f.read()
        
        self._rehydrated_exp: export.Exported = export.deserialize(byte_array)
        self.download_weight(gcs_bucket, gcs_weight_path)

    def get_exported(self):
        return self._rehydrated_exp

    def call_func(self, input_val):
        return self._rehydrated_exp.call(self._weight, input_val)

    def export_model(exported: export.Exported, gcs_bucket: str, blob: str):
        serialized: bytearray = exported.serialize()
        storage_client = storage.Client()
        blob = storage_client.bucket(gcs_bucket).blob(blob)

        with blob.open('wb') as f:
            f.write(serialized)

    def download_weight(self, gcs_bucket, gcs_path):
        dataset = ts.open({
            'driver': 'n5',
            'kvstore': {
            'driver': 'gcs',
            'bucket': gcs_bucket,
            'path': gcs_path,
            },
        }, read=True, write=False).result()
        print(dataset)
        self._weight = dataset[:, :].read().result()

