import multiprocessing as mp

from mserver.model_manager import ModelManager
from mserver.tpu_manager import TpuManager

import jax
import jax.numpy as jnp

import os
import time

class ModelRunner:
    _tpu_device_count = 4
    _model_manager = None
    _tpu_manager = None
    _pool = None

    def __init__(self):
        print("process of __init__ is {0}".format(os.getpid()))
        pool = mp.get_context("spawn").Pool(processes = 3)
        pool.map(self.pool_process_initializer, [1, 2, 3])
        self._pool = pool
        self._model_manager = ModelManager()
        self._model_manager.load_model("qianminj-bucket", "exported_tpu_linear1015", "tmp/dataset3")
        self._tpu_manager = TpuManager(4)


    def __del__(self):
        if self._pool:
            self._pool.close()
            self._pool.join()
    
    @staticmethod
    def pool_process_initializer(tpu_id: int):
        os.environ["JAX_PLATFORMS"] = "tpu"
        os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"
        os.environ["TPU_PROCESS_BOUNDS"] = "1,1,1"
        os.environ["TPU_VISIBLE_DEVICES"] = str(tpu_id)
        print("process_id: {0}, device: {1}".format(os.getpid(), jax.devices()))
    
    @staticmethod
    def work(func, input_value):
        x = jnp.array(input_value)
        print("jax array device: {0} in process {1}".format(x.devices(), os.getpid()))
        y = func(x)
        time.sleep(100)
        return y
        
    def run(self, input_value):
        if self._model_manager is None or self._tpu_manager is None or self._tpu_device_count == -1:
            raise RuntimeError('The ModelRunner is not initialized')
        res = self._pool.apply_async(self.work, (self._model_manager.call_func, input_value, ))
        return res.get(101)


