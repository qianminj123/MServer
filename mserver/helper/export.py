import numpy as np
import jax
from jax import export
from mserver.model_manager import ModelManager

def f(x): return 2 * x * x

exported: export.Exported = export.export(jax.jit(f))(
        jax.ShapeDtypeStruct((), np.float32))

ModelManager.export_model(exported, "qianminj-bucket", "exported_0908")
