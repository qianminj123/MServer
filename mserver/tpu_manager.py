import threading

class TpuManager:

    def __init__(self, device_count: int):
        self._device_count = device_count
        self._lock = threading.Lock()
        self._current = -1

    def acquire_one_tpu(self):
        self._lock.acquire()
        next_device = self._current + 1
        if next_device >= self._device_count:
            next_device = next_device - self._device_count
        self._current = next_device
        self._lock.release()
        return next_device
